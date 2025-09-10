# S2TS/utils/batch_processor.py

import time
import logging
from queue import Queue, Empty
from pathlib import Path
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass, field

# --- Setup logging ---
log = logging.getLogger(__name__)


@dataclass
class BatchJob:
    """Represents a single job within a batch process."""
    audio_path: Path
    settings: Dict[str, Any]
    status: str = "queued"  # queued, running, completed, failed, cancelled
    result: Optional[Any] = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


class BatchProcessor:
    """
    Handles the processing of multiple audio files in a sequential batch.
    """

    def __init__(self, pipeline_fn: Callable):
        self.pipeline_fn = pipeline_fn
        self.job_queue: Queue[BatchJob] = Queue()
        self.jobs: Dict[Path, BatchJob] = {}
        self._is_processing = False
        self._stop_requested = False

    def add_jobs(self, audio_files: List[Path], settings: Dict[str, Any]) -> None:
        """Adds a list of audio files to the processing queue."""
        for audio_path in audio_files:
            if audio_path not in self.jobs:
                job = BatchJob(audio_path=audio_path, settings=settings)
                self.jobs[audio_path] = job
                self.job_queue.put(job)
        log.info(f"Added {len(audio_files)} jobs to the batch queue.")

    def stop_processing(self) -> None:
        """Requests a graceful stop of the batch processing loop."""
        if self._is_processing:
            log.info("Stop request received. Finishing current job before stopping.")
            self._stop_requested = True

    def process_batch(
        self, progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Dict[Path, BatchJob]:
        """
        Processes all jobs in the queue sequentially.
        """
        self._is_processing = True
        self._stop_requested = False
        processed_count = 0
        # CRITICAL: Get the queue size *after* jobs have been added.
        total_count = self.job_queue.qsize()

        if progress_callback:
            progress_callback(processed_count, total_count, "Starting batch...")

        while not self.job_queue.empty():
            if self._stop_requested:
                log.info("Batch processing stopped by user.")
                break

            try:
                job = self.job_queue.get_nowait()
            except Empty:
                break # Queue is empty

            try:
                job.status = "running"
                job.start_time = time.time()
                current_file_msg = f"({processed_count + 1}/{total_count}) Processing: {job.audio_path.name}"

                if progress_callback:
                    progress_callback(processed_count, total_count, current_file_msg)

                # --- Execute the main pipeline function for this job ---
                job.result = self.pipeline_fn(
                    audio_path=str(job.audio_path),
                    settings=job.settings,
                    progress_callback=None # Individual progress handled inside pipeline
                )
                job.status = "completed"

            except Exception as e:
                log.error(f"Job for {job.audio_path.name} failed: {e}", exc_info=True)
                job.status = "failed"
                job.error = str(e)

            finally:
                job.end_time = time.time()
                processed_count += 1
                # Update progress after each file is done
                if progress_callback:
                    progress_callback(processed_count, total_count, f"Finished: {job.audio_path.name}")


        self._is_processing = False
        return self.jobs