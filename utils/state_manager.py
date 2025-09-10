# S2TS/utils/state_manager.py

import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from datetime import datetime

# --- Setup logging ---
log = logging.getLogger(__name__)


@dataclass
class StageState:
    """Represents the state of a single pipeline stage."""
    status: str = "pending"  # pending, running, completed, failed, skipped
    output_text: Optional[str] = None
    output_file: Optional[str] = None
    error: Optional[str] = None
    time_taken: Optional[float] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None


@dataclass
class PipelineState:
    """Holds the complete state of a pipeline run."""
    session_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    audio_file: Optional[str] = None
    project_dir: Optional[str] = None
    settings: Dict[str, Any] = field(default_factory=dict)
    asr: StageState = field(default_factory=StageState)
    clean_text: StageState = field(default_factory=StageState)
    translation: Dict[str, StageState] = field(default_factory=dict)
    tts: Dict[str, StageState] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class StateManager:
    """Manages the pipeline's state for persistence and resuming."""

    def __init__(self, state_file: Path = Path("pipeline_state.json")):
        """
        Initializes the StateManager.

        Args:
            state_file: The path to the JSON file where the state is stored.
        """
        self.state_file = state_file
        self.state = self._load_state()

    def _load_state(self) -> PipelineState:
        """Loads state from the file or creates a new one if it doesn't exist or is invalid."""
        if self.state_file.exists():
            try:
                with self.state_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Re-instantiate nested dataclasses correctly
                    data['asr'] = StageState(**data.get('asr', {}))
                    data['clean_text'] = StageState(**data.get('clean_text', {}))
                    data['translation'] = {lang: StageState(**s) for lang, s in data.get('translation', {}).items()}
                    data['tts'] = {lang: StageState(**s) for lang, s in data.get('tts', {}).items()}
                    return PipelineState(**data)
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                log.error(f"Error loading state file '{self.state_file}', creating a new one. Error: {e}")
                # Backup corrupted file
                corrupted_file = self.state_file.with_suffix(f".corrupted.{int(time.time())}.json")
                self.state_file.rename(corrupted_file)
                log.info(f"Backed up corrupted state file to {corrupted_file}")

        return PipelineState()

    def save_state(self) -> None:
        """Saves the current state to the JSON file."""
        self.state.updated_at = time.time()
        try:
            with self.state_file.open("w", encoding="utf-8") as f:
                json.dump(asdict(self.state), f, indent=4)
        except IOError as e:
            log.error(f"Failed to save state to {self.state_file}: {e}")

    def reset_state(self) -> None:
        """Resets the state to its initial, empty configuration."""
        log.info("Resetting pipeline state.")
        self.state = PipelineState()
        self.save_state()

    def can_resume(self) -> bool:
        """Checks if there's a valid, resumable state."""
        has_project = self.state.project_dir and Path(self.state.project_dir).exists()
        has_audio = self.state.audio_file and Path(self.state.audio_file).exists()
        is_not_manual = self.state.audio_file is not None

        return (is_not_manual and has_project and has_audio) or (not is_not_manual and has_project)

    def get_next_stage(self) -> str:
        """Determines the next pipeline stage to be executed based on the current state."""
        settings = self.state.settings

        if settings.get('enable_asr', False) and self.state.asr.status not in ["completed", "skipped"]:
            return "asr"
        if settings.get('enable_clean', False) and self.state.clean_text.status not in ["completed", "skipped"]:
            return "clean_text"

        target_langs = settings.get('target_langs', [])
        if settings.get('enable_translate', False):
            for lang in target_langs:
                if self.state.translation.get(lang, StageState()).status not in ["completed", "skipped"]:
                    return f"translation_{lang}"

        if settings.get('enable_tts', False):
            for lang in target_langs:
                if self.state.tts.get(lang, StageState()).status not in ["completed", "skipped"]:
                    return f"tts_{lang}"

        return "completed"

    def get_completed_stages(self) -> List[str]:
        """Returns a list of all stages that have been successfully completed."""
        completed = []
        if self.state.asr.status == "completed":
            completed.append("asr")
        if self.state.clean_text.status == "completed":
            completed.append("clean_text")
        for lang, state in self.state.translation.items():
            if state.status == "completed":
                completed.append(f"translation_{lang}")
        for lang, state in self.state.tts.items():
            if state.status == "completed":
                completed.append(f"tts_{lang}")
        return completed

    def update_stage(self, stage_name: str, **kwargs: Any) -> None:
        """
        Updates the state of a specific stage and saves the change.

        Args:
            stage_name: The name of the stage (e.g., 'asr', 'translation_Hindi').
            **kwargs: Key-value pairs of attributes to update on the stage's state.
        """
        stage_obj = None
        if stage_name == "asr":
            stage_obj = self.state.asr
        elif stage_name == "clean_text":
            stage_obj = self.state.clean_text
        elif stage_name.startswith("translation_"):
            lang = stage_name.replace("translation_", "")
            if lang not in self.state.translation:
                self.state.translation[lang] = StageState()
            stage_obj = self.state.translation[lang]
        elif stage_name.startswith("tts_"):
            lang = stage_name.replace("tts_", "")
            if lang not in self.state.tts:
                self.state.tts[lang] = StageState()
            stage_obj = self.state.tts[lang]

        if stage_obj:
            for key, value in kwargs.items():
                setattr(stage_obj, key, value)
            self.save_state()
        else:
            log.warning(f"Attempted to update an unknown stage: {stage_name}")