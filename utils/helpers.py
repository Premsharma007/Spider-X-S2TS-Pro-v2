# S2TS/utils/helpers.py

import os
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def ensure_dir(path: Path) -> None:
    """Ensure a directory exists, creating it if necessary."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log.error(f"Failed to create directory {path}: {e}")
        raise


def read_text(path: Path, default: str = "") -> str:
    """Read a text file with robust error handling."""
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return f.read()
    except (IOError, UnicodeDecodeError) as e:
        log.error(f"Failed to read text from {path}: {e}")
        return default


def write_text(path: Path, text: str) -> None:
    """Write text to a file, ensuring parent directories exist."""
    try:
        ensure_dir(path.parent)
        with path.open("w", encoding="utf-8") as f:
            f.write(text or "")
    except IOError as e:
        log.error(f"Failed to write text to {path}: {e}")
        raise


def secfmt(seconds: float) -> str:
    """Format seconds into a human-readable string (h:m:s or m:s)."""
    if seconds is None:
        return "N/A"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    mins, sec = divmod(seconds, 60)
    if mins < 60:
        return f"{mins}m {sec}s"
    hrs, mins = divmod(mins, 60)
    return f"{hrs}h {mins}m {sec}s"


def now_hhmmss() -> str:
    """Return the current time as an HH:MM:SS string."""
    return time.strftime("%H:%M:%S")


def make_project_folder(audio_file: Path, projects_dir: Path) -> Path:
    """Create a unique, timestamped project folder for an audio file."""
    base = audio_file.stem
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    project_path = projects_dir / f"Proj-{base}"
    ensure_dir(project_path)
    return project_path


def stage_filenames(proj_dir: Path, base_name: str, lang: str = "") -> Dict[str, Path]:
    """Generate standardized filenames for each pipeline stage."""
    # This function is now updated for the final Translate+TTS workflow
    filenames = {
        "asr": proj_dir / f"{base_name}-Manual-ASR.txt", # Used for the copied input .txt
        "clean": proj_dir / f"{base_name}-Clean.txt",
    }
    if lang:
        # Corrected naming for translation and TTS outputs
        filenames["trans"] = proj_dir / f"{base_name}-{lang}.txt"
        filenames["tts"] = proj_dir / f"{base_name}-{lang}-TTS.wav"
    return filenames

def get_clean_base_name(input_file: Path) -> str:
    """
    Takes an input file path and returns a clean base name,
    stripping all known stage suffixes like -ASR and -Cleaned.
    Example: Path("Sathish-Cleaned.txt") -> "Sathish"
    """
    base_name = input_file.stem
    suffixes_to_remove = ["-ASR", "-Cleaned", "-Manual-ASR"]
    for suffix in suffixes_to_remove:
        if base_name.endswith(suffix):
            base_name = base_name[:-len(suffix)]
    return base_name


def load_default_references(ref_text_file: Path, ref_audio_file: Path) -> Tuple[str, Optional[str]]:
    """Load the default reference text and audio file path."""
    ref_text = read_text(ref_text_file, default="")
    if ref_text:
        log.info(f"✅ Loaded default reference text: {len(ref_text)} characters")

    ref_audio_path = str(ref_audio_file) if ref_audio_file.exists() else None
    if ref_audio_path:
        log.info(f"✅ Loaded default reference audio: {ref_audio_file.name}")

    return ref_text, ref_audio_path


def get_supported_audio_files(folder_path: Path, formats: List[str]) -> List[Path]:
    """Scan a folder and return a list of all supported audio files."""
    audio_files = []
    if not folder_path.is_dir():
        log.warning(f"Batch folder not found: {folder_path}")
        return []

    for fmt in formats:
        # Case-insensitive search
        audio_files.extend(folder_path.glob(f"*{fmt.lower()}"))
        audio_files.extend(folder_path.glob(f"*{fmt.upper()}"))

    return sorted(list(set(audio_files))) # Sort and remove duplicates


def load_engines(engines_file: Path = Path("engines.json")) -> Dict[str, Any]:
    """Load AI engine configurations from a JSON file."""
    if not engines_file.exists():
        log.warning(f"Engine configuration file not found: {engines_file}")
        return {}
    try:
        with engines_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("engines", {})
    except (json.JSONDecodeError, IOError) as e:
        log.error(f"Error loading {engines_file}: {e}")
        return {}
    

def load_emotion_references(directory: Path) -> Dict[str, Tuple[str, str]]:
    """
    Scans a directory for audio files and matching .txt transcripts to build a
    reference dictionary for multi-reference TTS.

    Args:
        directory: The path to the folder containing emotion references.

    Returns:
        A dictionary mapping emotion tags (filenames) to a tuple of
        (audio_path, transcript_text).
    """
    ref_data = {}
    log.info(f"Scanning for emotion references in: {directory}")
    if not directory.is_dir():
        log.warning(f"Emotion reference directory not found: {directory}")
        return {}

    audio_files = list(directory.glob('*.wav')) + list(directory.glob('*.mp3'))
    for audio_file in audio_files:
        emotion_tag = audio_file.stem.lower()
        text_file = audio_file.with_suffix('.txt')

        if text_file.exists():
            transcript = read_text(text_file)
            if transcript:
                ref_data[emotion_tag] = (str(audio_file), transcript)
                log.info(f"  > Loaded emotion reference: '{emotion_tag}'")
            else:
                log.warning(f"  > Skipping '{emotion_tag}': Transcript file is empty.")
        else:
            log.warning(f"  > Skipping '{emotion_tag}': Missing transcript file '{text_file.name}'.")
    
    log.info(f"Found {len(ref_data)} valid emotion references.")
    return ref_data

def get_supported_text_files(folder_path: Path, formats: List[str]) -> List[Path]:
    """Scan a folder and return a list of all supported text files."""
    text_files = []
    if not folder_path.is_dir():
        log.warning(f"Text file folder not found: {folder_path}")
        return []

    for fmt in formats:
        text_files.extend(folder_path.glob(f"*{fmt.lower()}"))
        text_files.extend(folder_path.glob(f"*{fmt.upper()}"))

    return sorted(list(set(text_files)))

