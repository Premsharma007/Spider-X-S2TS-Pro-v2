# S2TS/utils/gui_automation.py

import time
import hashlib
import pyautogui
import pyperclip
import subprocess
import platform
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Tuple

from config import (
    PAGE_READY_DELAY, RESPONSE_TIMEOUT, SAMPLE_INTERVAL, 
    MIN_STREAM_TIME, STABLE_ROUNDS, CHROME_EXECUTABLE_PATH
)
from .helpers import ensure_dir

# --- Setup logging ---
log = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    """Configuration for a specific GUI automation engine."""
    url: str
    copy_btn_coords: Tuple[int, int] = (0, 0)
    copy_btn_image_path: Optional[str] = None


class RobustGuiEngine:
    """
    Manages robust GUI automation for interacting with web-based AI engines.
    
    Corrected Lifecycle:
    - This class no longer manages a persistent browser process. The OS handles it.
    - Each `send_and_get()` call is a self-contained task:
        1. It opens a new tab directly to the URL (using an existing browser if available).
        2. It performs the automation task.
        3. It closes the tab it opened.
    """

    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.screenshot_dir = Path("debug_screenshots")
        ensure_dir(self.screenshot_dir)

    def send_and_get(self, prompt: str, text: str, target_lang: Optional[str] = None) -> str:
        """
        Opens a new tab, sends a prompt, copies the response, and closes the tab.
        This is now the only public method.
        """
        try:
            # Step 1: Open and prepare a new tab efficiently.
            self._open_and_focus_tab()
            
            # Step 2: Compose and send the prompt.
            composed_prompt = self._compose_prompt(prompt, text, target_lang)
            self._paste_and_send(composed_prompt)
            pyautogui.hotkey("end")
            
            # Step 3: Monitor for a stable response.
            response = self._monitor_for_response(composed_prompt)
            if not response:
                log.warning("No valid response was captured before timeout.")
                self._capture_screenshot("empty_response_timeout")

            return response
            
        except Exception as e:
            log.error(f"A critical error occurred in send_and_get: {e}", exc_info=True)
            self._capture_screenshot("send_and_get_critical_error")
            raise
        finally:
            # Step 4: Always ensure the current tab is closed.
            self._close_current_tab()

    # --- Internal Automation Steps ---

    def _open_and_focus_tab(self):
        """
        Opens a new tab by calling the browser with the target URL directly.
        This is the efficient method from your original logic. It's the only
        place that launches a process.
        """
        log.info(f"Opening new tab directly to {self.cfg.url}...")
        try:
            command = [CHROME_EXECUTABLE_PATH, self.cfg.url]
            if platform.system() == "Darwin": # macOS
                command = ['open', '-a', 'Google Chrome', self.cfg.url]
            elif platform.system() != "Windows": # Linux
                command = ['google-chrome', self.cfg.url]
                
            subprocess.Popen(command)
            self._sleep(PAGE_READY_DELAY)

            log.info("Page loaded")
        except FileNotFoundError:
            log.error(f"Browser executable not found. Cannot open new tab.")
            raise

    def _paste_and_send(self, composed_prompt: str):
        """Pastes the composed prompt from the clipboard and sends it."""
        log.info("Sending prompt to GUI...")
        pyperclip.copy(composed_prompt)
        self._sleep(0.3)
        pyautogui.hotkey('ctrl', 'v')
        self._sleep(0.3)
        pyautogui.press('enter')
        log.info("✅ Prompt sent.")
        pyautogui.hotkey("end")

    def _monitor_for_response(self, sent_prompt: str) -> str:
        """
        Waits for an initial period, then polls the GUI for a stable response.
        """
        start_time = time.time()
        deadline = start_time + RESPONSE_TIMEOUT
        log.info(f"Giving the AI a {MIN_STREAM_TIME}s head start before polling...")
        self._sleep(MIN_STREAM_TIME)

        log.info("Now polling for a stable response...")
        last_digest = None
        stable_count = 0
        best_response = ""

        while time.time() < deadline:
            log.debug("Scrolling to bottom to see latest content...")
            pyautogui.hotkey("end")
            self._sleep(1)


            self._sleep(SAMPLE_INTERVAL)
            pyautogui.hotkey("end")
            page_text = self._copy_page_content()
            current_response = self._extract_reply(page_text, sent_prompt)

            if not current_response:
                continue

            if len(current_response) > len(best_response):
                best_response = current_response

            current_digest = hashlib.md5(current_response.encode()).hexdigest()
            if current_digest == last_digest:
                stable_count += 1
            else:
                stable_count = 0
                last_digest = current_digest
            
            if stable_count >= STABLE_ROUNDS:
                log.info(f"✅ Response stabilized after {time.time() - start_time:.1f} seconds.")
                return best_response
        
        log.warning(f"⚠️ Timeout reached after {RESPONSE_TIMEOUT}s. Returning best response found.")
        return best_response

    def _close_current_tab(self):
        """Closes the active browser tab."""
        log.info("Closing current tab...")
        if platform.system() == "Darwin":
            pyautogui.hotkey("command", "w")
        else:
            pyautogui.hotkey("ctrl", "w")
        self._sleep(0.5)

    def _copy_page_content(self) -> str:
        """
        Attempts to copy page content using a more robust sequence of actions.
        It prioritizes coordinate-based clicks with proper pauses and retries.
        """
        log.debug("Attempting to copy response...")
        pyperclip.copy("") # Clear the clipboard first

        # --- Method 1: Click by Coordinates (Primary Method) ---
        if self.cfg.copy_btn_coords and self.cfg.copy_btn_coords != (0, 0):
            # Try up to 3 times to ensure the page has settled
            for attempt in range(3):
                log.debug(f"Coordinate Click Attempt #{attempt + 1}")
                # 1. Press 'end' to scroll to the latest response
                pyautogui.press('end')
                # 2. IMPORTANT: Wait for the scroll to actually complete
                self._sleep(1.0) 
                
                # 3. Move to the button coordinates and click
                pyautogui.click(*self.cfg.copy_btn_coords)
                # 4. Wait for the click to register and the OS to copy to clipboard
                self._sleep(0.7)
                
                content = pyperclip.paste().strip()
                if content:
                    log.info("✅ Copy successful using coordinates.")
                    return content
                else:
                    log.warning("Coordinate click did not yield content, retrying...")
                    self._sleep(1.0) # Wait a bit longer before the next attempt

        # --- Method 2: Image Recognition (Fallback Method) ---
        if self.cfg.copy_btn_image_path:
            log.warning("Coordinate clicks failed. Falling back to image recognition...")
            try:
                # Scroll one last time to be sure
                pyautogui.press('end')
                self._sleep(1.0)
                
                button_location = pyautogui.locateOnScreen(self.cfg.copy_btn_image_path, confidence=0.8)
                if button_location:
                    pyautogui.click(pyautogui.center(button_location))
                    self._sleep(0.7)
                    content = pyperclip.paste().strip()
                    if content:
                        log.info("✅ Copy successful using image recognition.")
                        return content
                else:
                    log.warning(f"Could not find image '{self.cfg.copy_btn_image_path}' on screen.")
            except Exception as e:
                log.error(f"Image recognition for copy button failed: {e}")
                self._capture_screenshot("copy_image_recognition_error")

        log.error("All copy methods failed. Returning empty response.")
        return "" # Return empty if all methods fail
        '''
        # Method 3: Final Fallback (Select All + Copy)
        log.warning("Primary copy methods failed. Falling back to 'Select All + Copy'.")
        pyautogui.hotkey("ctrl", "a")
        self._sleep(0.1)
        pyautogui.hotkey("ctrl", "c")
        self._sleep(0.2)
        return pyperclip.paste().strip()'''

    def _compose_prompt(self, prompt: str, text: str, target_lang: Optional[str]) -> str:
        """Constructs the full string to be sent to the GUI."""
        composed = (prompt or "").strip()
        if target_lang:
            composed += f"\n\nTarget language: {target_lang}"
        composed += f"\n\nInput:\n{(text or '').strip()}"
        return composed

    def _extract_reply(self, whole_page_text: str, sent_text: str) -> str:
        """Extracts the AI's latest reply by removing the initial prompt text."""
        if not whole_page_text or not sent_text:
            return ""
        index = whole_page_text.rfind(sent_text)
        if index != -1:
            return whole_page_text[index + len(sent_text):].strip()
        return whole_page_text

    def _capture_screenshot(self, context: str):
        """Captures a screenshot for debugging purposes."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.screenshot_dir / f"{context}_{timestamp}.png"
            pyautogui.screenshot(filename)
            log.info(f"📸 Screenshot captured: {filename}")
        except Exception as e:
            log.error(f"Failed to capture screenshot for '{context}': {e}")

    def _sleep(self, duration: float):
        """A simple wrapper for time.sleep for consistency."""
        time.sleep(duration)