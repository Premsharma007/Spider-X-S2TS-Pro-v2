# S2TS/config.py

import os
from pathlib import Path

# --- Core Directories ---
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "Data"
LOG_DIR = BASE_DIR / "Logs"
MODELS_DIR = BASE_DIR / "Models"
PROMPTS_DIR = BASE_DIR / "Prompts"
REFERENCE_DIR = BASE_DIR / "Reference-Txt-Aud"
EMOTION_REF_DIR = REFERENCE_DIR / "Emotions"

# --- Project & Input/Output Directories ---
INPUT_DIR = BASE_DIR / "Input"
INCOMING_DIR = INPUT_DIR / "Incoming_audio"
PROJECTS_DIR = DATA_DIR / "Projects"
ASR_TRANSCRIPTIONS_DIR = BASE_DIR / "ASR-Transcriptions"

# --- Model-Specific Directories ---

# --- Prompt Files ---
CORRECTOR_PROMPT_FILE = PROMPTS_DIR / "Corrector-Prompt.txt"
TRANSLATOR_PROMPT_FILE = PROMPTS_DIR / "Translator-Prompt.txt"

# --- ASR (Automatic Speech Recognition) Configuration ---
WHISPER_CLI_PATH = r"C:\AI\Whisper.cpp\whisper-cli.exe"
WHISPER_MODEL_PATH = r"C:\AI\Whisper.cpp\ggml-large-v2.bin"
# --- Browser Path
CHROME_EXECUTABLE_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

# --- Default Reference Files for TTS ---
# These are used for voice cloning if no other reference is provided.
DEFAULT_REF_TEXT_FILE = REFERENCE_DIR / "Reference_Text.txt"
DEFAULT_REF_AUDIO_FILE = REFERENCE_DIR / "Reference_Audio.MP3"


# --- GUI Automation Settings ---
PAGE_READY_DELAY = 5  # Seconds to wait for a webpage to load.
RESPONSE_TIMEOUT = 180  # Max seconds to wait for a response from the GUI.
SAMPLE_INTERVAL = 1.2  # Seconds between checks for a stable GUI response.
MIN_STREAM_TIME = 35  # Minimum time to wait before considering a response stable.
STABLE_ROUNDS = 3  # Number of identical responses needed to confirm stability.

# --- Language Options ---
# Supported target languages for translation.
LANG_LABELS = {
    "Hindi": "Hindi",
    "Kannada": "Kannada",
    "Telugu": "Telugu",
}

# --- Default Prompts ---
# Fallback prompts if the corresponding files are not found.
DEFAULT_CORRECTOR_PROMPT = (
    "You are a meticulous Tamil copy-editor for ASR output. "
    "Fix mishears, punctuation, casing, numerals, and spacing. "
    "Do NOT add or omit meaning. Return only cleaned Tamil text."
)

DEFAULT_TRANSLATOR_PROMPT = (
    "You are a professional translator. Translate the following **Tamil** text to the target language. "
    "Use natural register, preserve proper nouns, avoid code-mixing, and return only the translation."
)

# --- UI (User Interface) Settings ---
APP_TITLE = "Spider『X』 Speech → Translated Speech (S2TS) - Professional"
APP_TAG = "ASR (Tamil) → Clean Tamil → Translate (Hindi/Kannada/Telugu) → TTS (Indic-F5)"
DARK_BG_GRADIENT = "linear-gradient(135deg, #0f172a, #1e293b)"
THEME_PRIMARY = "cyan"
THEME_SECONDARY = "blue"
THEME_NEUTRAL = "gray"

# --- Performance Settings ---
MAX_PARALLEL_JOBS = 1  # Number of parallel jobs for batch processing.
TIMEOUT_PER_STAGE = 3600  # Max seconds per pipeline stage (1 hour).
AUTO_RETRY_FAILED = True  # Whether to automatically retry failed jobs.
MAX_MEMORY_MB = 4096  # Max memory to allocate (for future use).
GPU_MEMORY_LIMIT = 0.8  # GPU memory limit (for future use).

# --- Supported File Formats ---
SUPPORTED_AUDIO_FORMATS = [".wav", ".mp3", ".m4a", ".flac", ".ogg"]