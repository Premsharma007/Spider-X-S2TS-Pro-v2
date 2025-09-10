# S2TS/modules/text_cleaner.py

import time
import logging
from typing import Tuple, Optional, Callable

from utils.gui_automation import RobustGuiEngine, EngineConfig
from utils.helpers import read_text
from config import CORRECTOR_PROMPT_FILE, DEFAULT_CORRECTOR_PROMPT

# --- Setup logging ---
log = logging.getLogger(__name__)


def clean_text_gui(
    text: str,
    engine_config: EngineConfig,
    progress_callback: Optional[Callable[[int, str], None]] = None
) -> Tuple[str, float]:
    """
    Cleans ASR-generated text using a configured GUI automation engine.

    Args:
        text: The raw text to be cleaned.
        engine_config: The configuration for the GUI engine.
        progress_callback: A function to report progress updates.

    Returns:
        A tuple containing the cleaned text and the time taken in seconds.
        Returns the original text on failure.
    """
    start_time = time.time()

    def _progress(percent: int, message: str):
        if progress_callback:
            progress_callback(percent, message)
        log.info(f"[TextClean Progress] {percent}% - {message}")

    if not text.strip():
        log.warning("Input text for cleaning is empty. Skipping.")
        return "", 0.0

    engine = RobustGuiEngine(engine_config)
    try:
        # --- 1. Load Corrector Prompt ---
        _progress(10, "Loading text corrector prompt.")
        corrector_prompt = read_text(CORRECTOR_PROMPT_FILE, DEFAULT_CORRECTOR_PROMPT)

        # --- 2. Start GUI Engine ---
        _progress(25, "Starting browser for text cleaning.")
        #engine.start()

        # --- 3. Send Text and Get Cleaned Version ---
        _progress(50, "Sending text to AI engine for cleaning.")
        cleaned_text = engine.send_and_get(corrector_prompt, text)
        _progress(90, "Cleaned text received.")

        duration = time.time() - start_time
        _progress(100, f"Text cleaning completed in {duration:.1f}s.")
        return cleaned_text, duration

    except Exception as e:
        log.error(f"Text cleaning process failed: {e}", exc_info=True)
        
        # Return the original text as a fallback
        return text, time.time() - start_time