# S2TS/modules/asr.py

import time
import logging
from pathlib import Path
from typing import Tuple, Optional, Callable
import json
import subprocess

import config
from utils.helpers import read_text, secfmt

# --- Setup logging ---
log = logging.getLogger(__name__)


def run_asr(
    audio_path: Path,
    out_txt_path: Path, # Note: This path is for the final .txt, not the intermediate .json
    progress_callback: Optional[Callable[[int, str], None]] = None
) -> Tuple[str, float]:
    """
    Performs Automatic Speech Recognition using a command-line call to Whisper.cpp.

    Args:
        audio_path: The absolute path to the input audio file.
        out_txt_path: The path where the final, clean transcription text should be saved.
        progress_callback: A function to report progress updates to the UI.

    Returns:
        A tuple containing the full transcribed text and the time taken in seconds.
    """
    start_time = time.time()

    def _progress(percent: int, message: str):
        if progress_callback:
            progress_callback(percent, message)
        log.info(f"[ASR Progress] {percent}% - {message}")

    try:
        # --- 1. Validate Paths ---
        whisper_bin = Path(config.WHISPER_CLI_PATH)
        model_path = Path(config.WHISPER_MODEL_PATH)
        if not whisper_bin.exists() or not model_path.exists():
            raise FileNotFoundError(
                "Whisper.cpp executable or model not found. Please check paths in config.py"
            )
        if not audio_path.exists():
            raise FileNotFoundError(f"Input audio file not found: {audio_path}")

        # --- 2. Prepare Command ---
        # The JSON output will be created in the same directory as the final txt output.
        output_base = out_txt_path.with_suffix('')
        json_output_path = output_base.with_suffix(".json")

        cmd = [
            str(whisper_bin),
            "-m", str(model_path),
            "-f", str(audio_path.resolve()),
            "-oj",         
            "-l", "auto",  
            "-of", str(output_base) 
        ]

        # --- 3. Run Whisper.cpp Subprocess ---
        _progress(20, "Starting Whisper.cpp process...")
        log.info(f"Executing command: {' '.join(cmd)}")
        
        # Using capture_output=True to get stdout/stderr for better debugging
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
        
        log.debug(f"Whisper.cpp stdout: {result.stdout}")
        log.debug(f"Whisper.cpp stderr: {result.stderr}")
        _progress(80, "Whisper.cpp process completed.")

        # --- 4. Parse JSON Transcript ---
        if not json_output_path.exists():
            raise FileNotFoundError(f"Whisper.cpp did not create the expected JSON output at {json_output_path}")

        with open(json_output_path, "r", encoding="utf-8") as f:
            transcript_data = json.load(f)

        # Try both possible key names: "segments" (old) or "transcription" (new)
        segments = transcript_data.get("segments") or transcript_data.get("transcription", [])
        full_text = " ".join([seg["text"].strip() for seg in segments])

        detected_lang = transcript_data.get("result", {}).get("language") or transcript_data.get("language", "unknown")
        
        _progress(95, f"Transcription parsed. Detected language: {detected_lang}")
        log.info(f"[ASR] Detected language: {detected_lang}")
        log.info(f"[ASR] Full text length: {len(full_text)} chars")

        # --- 5. Cleanup and Return ---
        duration = time.time() - start_time
        _progress(100, f"ASR completed in {secfmt(duration)}.")
        
        # Optional: Delete the intermediate JSON file if you don't need it
        # json_output_path.unlink()

        return full_text, duration

    except subprocess.CalledProcessError as e:
        log.error(f"Whisper.cpp failed with exit code {e.returncode}.")
        log.error(f"Stderr: {e.stderr}")
        raise RuntimeError(f"Whisper.cpp failed. Check logs for details.")
    except Exception as e:
        log.error(f"An unexpected error occurred during ASR: {e}", exc_info=True)
        raise