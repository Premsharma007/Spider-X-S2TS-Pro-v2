# S2TS/modules/translator.py

import time
import logging
from typing import Tuple, Optional, Callable

from utils.gui_automation import RobustGuiEngine, EngineConfig
from utils.helpers import read_text
from config import TRANSLATOR_PROMPT_FILE, DEFAULT_TRANSLATOR_PROMPT

# --- Setup logging ---
log = logging.getLogger(__name__)


def translate_text(
    text: str,
    target_lang: str,
    engine_config: EngineConfig,
    progress_callback: Optional[Callable[[int, str], None]] = None
) -> Tuple[str, float]:
    """
    Translates text to a single target language using an isolated browser session.

    Args:
        text: The source text to translate.
        target_lang: The target language (e.g., "Hindi").
        engine_config: The configuration for the GUI engine.
        progress_callback: A function to report progress updates.

    Returns:
        A tuple containing the translated text and the time taken in seconds.
    """
    start_time = time.time()

    def _progress(percent: int, message: str):
        if progress_callback:
            progress_callback(percent, message)
        log.info(f"[Translate-{target_lang} Progress] {percent}% - {message}")

    if not text.strip():
        log.warning(f"Input text for {target_lang} translation is empty. Skipping.")
        return "", 0.0

    # Each translation gets a fresh, isolated engine instance
    engine = RobustGuiEngine(engine_config)
    try:
        # --- 1. Load Translator Prompt ---
        _progress(10, f"Loading translator prompt for {target_lang}.")
        translator_prompt = read_text(TRANSLATOR_PROMPT_FILE, DEFAULT_TRANSLATOR_PROMPT)

        # --- 2. Start GUI Engine ---
        _progress(25, f"Starting browser for {target_lang} translation.")
        #engine.start()

        # --- 3. Send Text and Get Translation ---
        _progress(50, f"Sending text for {target_lang} translation.")
        translated_text = engine.send_and_get(translator_prompt, text, target_lang=target_lang)
        _progress(90, f"Received {target_lang} translation.")
        

        duration = time.time() - start_time
        _progress(100, f"{target_lang} translation completed in {duration:.1f}s.")
        return translated_text, duration

    except Exception as e:
        log.error(f"Translation process for {target_lang} failed: {e}", exc_info=True)
        return "", time.time() - start_time