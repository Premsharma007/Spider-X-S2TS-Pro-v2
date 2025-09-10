# S2TS/modules/tts.py

import time
import tempfile
import re
import os
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict
import numpy as np
import soundfile as sf
import torch

# Import the new helper function
from utils.helpers import load_emotion_references, read_text
import config

# --- Setup logging ---
log = logging.getLogger(__name__)

# --- Safe Model Loading (once at module level) ---
try:
    from transformers import AutoModel
    F5_MODEL_REPO = "6Morpheus6/IndicF5"
    log.info("Loading IndicF5 TTS model...")
    F5_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    F5_MODEL = AutoModel.from_pretrained(F5_MODEL_REPO, trust_remote_code=True).to(F5_DEVICE)
    log.info(f"TTS model loaded successfully on device: {F5_DEVICE}")
    TTS_AVAILABLE = True
except Exception as e:
    log.error(f"Failed to load the IndicF5 TTS model: {e}", exc_info=True)
    F5_MODEL, F5_DEVICE, TTS_AVAILABLE = None, None, False

# --- Core Worker Function ---
def _synthesize_single_segment(text: str, ref_audio_path: str, ref_text: str) -> np.ndarray:
    """Generates audio for a single, short text segment."""
    log.debug(f"Synthesizing segment: '{text[:40]}...'")
    audio = F5_MODEL(text, ref_audio_path=ref_audio_path, ref_text=ref_text)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    return audio

# --- Synthesis Strategies ---
def _synthesize_in_chunks(text: str, ref_audio_path: str, ref_text: str, progress_callback) -> np.ndarray:
    """Synthesizes long text by breaking it into sentences and using one reference."""
    progress_callback(15, "Splitting text into sentences for chunking.")
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if not any(s.strip() for s in sentences):
        sentences = [text.strip()]

    audio_segments = []
    total_sentences = len(sentences)
    for i, sentence in enumerate(sentences):
        if not sentence.strip():
            continue
        
        progress_percent = 20 + int((i / total_sentences) * 60)
        progress_callback(progress_percent, f"Synthesizing chunk {i+1}/{total_sentences}")
        
        segment = _synthesize_single_segment(sentence, ref_audio_path, ref_text)
        audio_segments.append(segment)
        # Add a small silence between sentences for natural pacing
        audio_segments.append(np.zeros(int(0.3 * 24000), dtype=np.float32)) # 300ms pause

    return np.concatenate(audio_segments) if audio_segments else np.array([], dtype=np.float32)

def _synthesize_multi_reference(tagged_script: str, ref_data: Dict[str, Tuple[str, str]], progress_callback) -> np.ndarray:
    """Synthesizes text using different reference clips for tagged segments."""
    progress_callback(15, "Parsing tagged script for multi-reference synthesis.")
    segments = re.findall(r'\[(.*?)\]\s*(.*?)(?=\[|$)', tagged_script, re.DOTALL)
    if not segments:
        raise ValueError("Multi-reference mode selected, but no tags like [tag] were found in the text.")

    audio_segments = []
    total_segments = len(segments)
    for i, (tag, text_segment) in enumerate(segments):
        tag = tag.strip().lower()
        text_segment = text_segment.strip()

        if not text_segment:
            continue
        
        if tag not in ref_data:
            log.warning(f"Reference for tag '[{tag}]' not found. Skipping segment.")
            continue

        ref_audio_path, ref_text = ref_data[tag]
        progress_percent = 20 + int((i / total_segments) * 60)
        progress_callback(progress_percent, f"Synthesizing segment {i+1}/{total_segments} with tag '[{tag}]'")
        
        segment = _synthesize_single_segment(text_segment, ref_audio_path, ref_text)
        audio_segments.append(segment)
        audio_segments.append(np.zeros(int(0.3 * 24000), dtype=np.float32)) # 300ms pause

    if not audio_segments:
        raise ValueError("No valid segments with matching reference tags were found to synthesize.")

    return np.concatenate(audio_segments)

# --- Main Pipeline Entry Point ---
def synthesize_tts(text: str, ref_audio_tuple: Optional[Tuple], ref_text: Optional[str], out_path: Path, use_emotion_refs: bool, progress_callback) -> float:
    """
    This is the main "bridge" function called by the pipeline.
    It prepares the data and calls the appropriate synthesis strategy.
    """
    start_time = time.time()
    tmp_ref_path = None
    final_audio = np.array([], dtype=np.float32)

    try:
        if not TTS_AVAILABLE:
            raise RuntimeError("TTS model is not available for synthesis.")
        if not text or not text.strip():
            raise ValueError("Input text for TTS cannot be empty.")

        if use_emotion_refs:
            # --- Multi-Reference Mode ---
            progress_callback(5, "Loading emotion references for tagged synthesis.")
            ref_data = load_emotion_references(config.EMOTION_REF_DIR)
            if not ref_data:
                raise ValueError("Emotion references enabled, but no valid references were found in the folder.")
            final_audio = _synthesize_multi_reference(text, ref_data, progress_callback)
        
        else:
            # --- Single-Reference Mode ---
            if ref_audio_tuple is None or not ref_text or not ref_text.strip():
                raise ValueError("Single reference audio and text are required for this mode.")
            
            progress_callback(5, "Preparing single reference audio.")
            sr, audio_np = ref_audio_tuple
            
            if np.issubdtype(audio_np.dtype, np.floating):
                audio_np = (audio_np * 32767).astype(np.int16)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_ref_path = tmp_file.name
                sf.write(tmp_ref_path, audio_np, sr)

            final_audio = _synthesize_in_chunks(text, tmp_ref_path, ref_text, progress_callback)
        
        progress_callback(95, "Finalizing and saving audio file.")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), final_audio, samplerate=24000)
        progress_callback(100, f"TTS audio saved to {out_path.name}")

    except Exception as e:
        log.error(f"TTS synthesis failed: {e}", exc_info=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), np.zeros(24000, dtype=np.float32), 24000)
        log.info(f"Created a silent fallback audio file at {out_path.name}")
    
    finally:
        if tmp_ref_path and os.path.exists(tmp_ref_path):
            os.remove(tmp_ref_path)
            log.debug(f"Cleaned up temporary reference file: {tmp_ref_path}")
            
    return time.time() - start_time