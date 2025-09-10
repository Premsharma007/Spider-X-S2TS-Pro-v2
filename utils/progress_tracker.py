import time
from typing import Dict, Any, Callable, Optional

class StageProgress:
    def __init__(self, total_stages: int = 4):  # ASR, Clean, Translate, TTS
        self.total_stages = total_stages
        self.current_stage = 0
        self.stage_progress = 0
        self.stage_name = ""
        self.start_time = time.time()
        self.stage_times = {}
    
    def set_stage(self, stage_idx: int, stage_name: str):
        """Set the current stage."""
        self.current_stage = stage_idx
        self.stage_name = stage_name
        self.stage_progress = 0
        self.stage_start_time = time.time()
    
    def update_progress(self, progress: int, message: str = ""):
        """Update progress within the current stage."""
        self.stage_progress = progress
    
    def complete_stage(self):
        """Mark the current stage as completed."""
        stage_time = time.time() - self.stage_start_time
        self.stage_times[self.stage_name] = stage_time
    
    def get_overall_progress(self) -> int:
        """Calculate overall progress across all stages."""
        stage_progress = self.stage_progress / 100.0 if self.stage_progress > 0 else 0
        completed_stages = max(0, self.current_stage - 1)
        return int((completed_stages + stage_progress) / self.total_stages * 100)
    
    def get_elapsed_time(self) -> float:
        """Get total elapsed time."""
        return time.time() - self.start_time

def create_progress_callback(progress_tracker: StageProgress, 
                           stage_idx: int, 
                           stage_name: str,
                           ui_callback: Optional[Callable] = None) -> Callable:
    """Create a progress callback function for a specific stage."""
    progress_tracker.set_stage(stage_idx, stage_name)
    
    def callback(progress: int, message: str):
        progress_tracker.update_progress(progress, message)
        if ui_callback:
            overall = progress_tracker.get_overall_progress()
            ui_callback(overall, f"{stage_name}: {message}")
    
    return callback