# ==============================================================================
# S2TS Pipeline - Main Application (Revised)
# ------------------------------------------------------------------------------
# Changes:
# - Fixed UI wiring and event handler signatures
# - Removed invalid yields from inner callbacks
# - Robust mapping from language → output widgets
# - Safer batch directory handling
# - Clearer comments and logging
# ==============================================================================

import time
import logging
import random
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
import gradio as gr
import numpy as np  # noqa: F401 (used via settings; kept for clarity)
from dataclasses import asdict
from queue import Queue, Empty
import shutil

# --- Local Imports from Refactored Modules ---
import config
from utils.helpers import (
    ensure_dir, read_text, write_text, secfmt, now_hhmmss,
    load_default_references, get_supported_audio_files, load_engines, stage_filenames, get_supported_text_files, get_clean_base_name
)
from utils.state_manager import StateManager, StageState
from utils.resource_monitor import ResourceMonitor
from utils.batch_processor import BatchProcessor
from modules.asr import run_asr
from modules.text_cleaner import clean_text_gui, EngineConfig
from modules.translator import translate_text
from modules.tts import synthesize_tts

# --- Global Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

ENGINES = load_engines()
resource_monitor = ResourceMonitor()


class PipelineController:
    """
    Central orchestrator for the S2TS pipeline.
    Handles state, stage execution, and batch coordination.
    """

    def __init__(self):
        self.state_manager = StateManager()
        self.batch_processor = BatchProcessor(self.run_pipeline)
        self._is_processing = False
        self._stop_requested = False
        self._temp_ref_audio = None 

    # -------------------- Lifecycle / Control --------------------

    def request_stop(self):
        """Signal the pipeline/batch to stop gracefully."""
        log.info("Stop requested by user.")
        self._stop_requested = True
        self.batch_processor.stop_processing()

    def _check_for_stop(self):
        """Raise if a stop has been requested."""
        if self._stop_requested:
            raise InterruptedError("Pipeline execution was stopped by the user.")

# In New S2TS/main.py, inside class PipelineController

    def initialize_pipeline(self, input_path: Optional[str], settings: Dict[str, Any]) -> str:
        """
        Initializes the pipeline, creating a consolidated project folder for Translate/TTS runs.
        """
        self._stop_requested = False
        self.state_manager.reset_state()
        state = self.state_manager.state

        is_asr_run = settings.get('enable_asr', False)
        is_translate_run = settings.get('enable_translate', False) or settings.get('enable_tts', False)

        project_dir = ""
        if not input_path:
            raise ValueError("Input path cannot be empty for pipeline initialization.")

        input_file = Path(input_path)
        # Use our new helper to get a reliable, clean base name
        base_name = get_clean_base_name(input_file)
        state.audio_file = str(input_file)

        # --- MODE 2: Translate/TTS Run ---
        if not is_asr_run and is_translate_run:
            # 1. Create the final, consolidated project folder
            project_dir_path = config.PROJECTS_DIR / f"Proj-{base_name}-{now_hhmmss().replace(':', '')}"
            ensure_dir(project_dir_path)
            project_dir = str(project_dir_path)
            state.project_dir = project_dir

            # 2. Consolidate Files: Copy original ASR and Cleaned TXT files into the project folder
            cleaned_txt_path = input_file
            original_asr_txt_path = config.ASR_TRANSCRIPTIONS_DIR / f"{base_name}-ASR.txt"

            # Copy the cleaned file (the input for this stage)
            if cleaned_txt_path.exists():
                shutil.copy(cleaned_txt_path, project_dir_path / cleaned_txt_path.name)
            
            # Copy the original ASR transcript for a complete record
            if original_asr_txt_path.exists():
                shutil.copy(original_asr_txt_path, project_dir_path / original_asr_txt_path.name)

            # 3. Pre-fill the state with the content of the input (cleaned) text file
            text_content = read_text(cleaned_txt_path)
            state.clean_text = StageState(status="skipped", output_text=text_content)
            state.asr = StageState(status="skipped", output_text=read_text(original_asr_txt_path))


        self._temp_ref_audio = settings.get("ref_audio_numpy")
        settings_copy = settings.copy()
        settings_copy.pop("ref_audio_numpy", None)
        state.settings = settings_copy
        self.state_manager.save_state()
        
        log.info(f"Pipeline initialized. Project Dir: {'N/A (ASR/Clean Mode)' if not project_dir else project_dir}")
        return project_dir
    # -------------------- Run Orchestration --------------------

    def run_full_pipeline_from_resume(self, progress_callback: Optional[Callable] = None):
        """
        Resume through remaining stages from current state.
        Suitable for the 'Resume' button.
        """
        if self._is_processing:
            log.warning("Cannot resume, a process is already running.")
            return

        self._is_processing = True
        self._stop_requested = False
        try:
            while (next_stage := self.state_manager.get_next_stage()) != "completed":
                self._check_for_stop()
                self.run_stage(next_stage, progress_callback)
        except InterruptedError as e:
            log.warning(str(e))
            if progress_callback:
                progress_callback(4, 100, "Pipeline stopped by user.")
        except Exception as e:
            log.error(f"Pipeline execution failed during resume: {e}", exc_info=True)
            if progress_callback:
                progress_callback(4, 100, f"PIPELINE FAILED: {e}")
            raise
        finally:
            self._is_processing = False

# In New S2TS/main.py, inside class PipelineController

    def run_pipeline(
        self,
        input_path: Optional[str],
        settings: Dict[str, Any],
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Run the relevant pipeline stages based on the detected mode from settings.
        """
        if self._is_processing:
            log.warning("Pipeline is already running.")
            return {}

        project_dir = self.initialize_pipeline(input_path, settings)
        log.info(f"Starting pipeline run for: {input_path or 'manual input'}")

        # --- Run stages based on settings ---
        if settings.get('enable_asr'):
            self.run_stage('asr', progress_callback)
        
        # The 'clean_text' stage can run right after ASR in the same pass
        if settings.get('enable_clean'):
            self.run_stage('clean_text', progress_callback)

        # For Translate/TTS runs, the state is already set up to continue
        if settings.get('enable_translate') or settings.get('enable_tts'):
             if not settings.get('enable_asr'): # Only run this if ASR was not part of the run
                self.run_full_pipeline_from_resume(progress_callback)

        log.info("Pipeline run finished.")
        return self.get_summary()

    # -------------------- Stage Execution --------------------

    def run_stage(self, stage: str, progress_callback: Optional[Callable] = None) -> bool:
        """
        Execute a single stage with a reliably clean base_name.
        """
        state = self.state_manager.state
        project_dir = Path(state.project_dir) if state.project_dir else Path.cwd()
        
        # --- KEY CHANGE: Use the helper function for a clean base_name ---
        base_name = get_clean_base_name(Path(state.audio_file)) if state.audio_file else "manual"

        stage_map = {
            "asr": (0, self._run_asr_stage),
            "clean_text": (1, self._run_clean_stage),
            "translation": (2, self._run_translation_stage),
            "tts": (3, self._run_tts_stage),
        }

        # Robust stage parsing
        stage_key = None
        lang = None

        if stage in stage_map:
            stage_key = stage
        elif '_' in stage:
            parts = stage.split('_', 1)
            if parts[0] in stage_map:
                stage_key = parts[0]
                lang = parts[1]

        if stage_key is None:
            log.error(f"Unknown stage: '{stage}'.")
            return False  # Prevent looping

        stage_idx, stage_func = stage_map[stage_key]

        def stage_progress_callback(percent: int, message: str):
            if progress_callback:
                # Relay progress for the current stage (0..3)
                progress_callback(stage_idx, percent, message)

        try:
            log.info(f"--- Running Stage: {stage.upper()} ---")
            if stage.startswith("translation_") or stage.startswith("tts_"):
                lang = stage.split('_', 1)[1]
                stage_func(lang, project_dir, base_name, stage_progress_callback)
            else:
                stage_func(project_dir, base_name, stage_progress_callback)
            log.info(f"--- Stage {stage.upper()} COMPLETED ---")
            return True
        except Exception as e:
            log.error(f"Failed to run stage '{stage}': {e}", exc_info=True)
            self.state_manager.update_stage(stage, status="failed", error=str(e))
            raise

    # -------------------- Stage Implementations --------------------

    def _run_asr_stage(self, project_dir: Path, base_name: str, progress_cb: Callable):
        """
        ASR stage. Saves transcript to a central folder instead of a project folder.
        """
        self.state_manager.update_stage("asr", status="running", start_time=time.time())
        
        audio_file = self.state_manager.state.audio_file
        if not audio_file:
            raise ValueError("No input audio found for ASR stage.")

        # --- KEY CHANGE: Define output path in the dedicated ASR folder ---
        ensure_dir(config.ASR_TRANSCRIPTIONS_DIR)
        asr_output_path = config.ASR_TRANSCRIPTIONS_DIR / f"{Path(audio_file).stem}-ASR.txt"
        
        asr_text, asr_time = run_asr(Path(audio_file), asr_output_path, progress_cb)

        write_text(asr_output_path, asr_text or "")
        self.state_manager.update_stage(
            "asr",
            status="completed",
            output_text=asr_text or "",
            output_file=str(asr_output_path),
            time_taken=asr_time,
            end_time=time.time()
        )

    def _run_clean_stage(self, project_dir: Path, base_name: str, progress_cb: Callable):
        """
        Text cleaning stage. Saves output based on the run mode.
        """
        # In an ASR+Clean run, the ASR stage has just completed.
        # In a Translate/TTS run, this stage is loaded from the input file.
        text_to_clean = self.state_manager.state.asr.output_text
        if text_to_clean is None:
            raise ValueError("Input text not available for cleaning. ASR must be run or a text file provided.")
        
        # Determine if we are in an ASR/Clean only run
        settings = self.state_manager.state.settings
        is_pre_process_run = settings.get('enable_asr', False) or not (settings.get('enable_translate') or settings.get('enable_tts'))

        # Clean up the base_name for the output file
        clean_base_name = Path(self.state_manager.state.audio_file).stem.replace("-ASR", "")

        self.state_manager.update_stage("clean_text", status="running", start_time=time.time())
        engine_cfg = self._get_engine_config()
        cleaned_text, clean_time = clean_text_gui(text_to_clean, engine_cfg, progress_cb)

        if is_pre_process_run:
            # --- MODE 1: Save to the central ASR-Transcriptions folder ---
            ensure_dir(config.ASR_TRANSCRIPTIONS_DIR)
            output_path = config.ASR_TRANSCRIPTIONS_DIR / f"{clean_base_name}-Cleaned.txt"
        else:
            # --- MODE 2: Save to the project-specific folder ---
            files = stage_filenames(project_dir, base_name)
            output_path = files["clean"]

        write_text(output_path, cleaned_text or "")
        self.state_manager.update_stage(
            "clean_text",
            status="completed",
            output_text=cleaned_text or "",
            output_file=str(output_path),
            time_taken=clean_time,
            end_time=time.time()
        )

    def _run_translation_stage(self, lang: str, project_dir: Path, base_name: str, progress_cb: Callable):
        """Translation stage for a given target language."""
        # Prefer cleaned text if available, else fall back to raw ASR
        text = self.state_manager.state.clean_text.output_text or self.state_manager.state.asr.output_text
        if not text:
            raise ValueError("No input text available for translation.")
        files = stage_filenames(project_dir, base_name, lang)
        engine_cfg = self._get_engine_config()
        stage_name = f"translation_{lang}"
        self.state_manager.update_stage(stage_name, status="running", start_time=time.time())
        translated_text, trans_time = translate_text(text, lang, engine_cfg, progress_cb)
        write_text(files["trans"], translated_text or "")
        self.state_manager.update_stage(
            stage_name,
            status="completed",
            output_text=translated_text or "",
            output_file=str(files["trans"]),
            time_taken=trans_time,
            end_time=time.time()
        )

    def _run_tts_stage(self, lang: str, project_dir: Path, base_name: str, progress_cb: Callable):
        """TTS stage for a given target language."""
        # Normalize human-readable names → ISO codes
        lang_map = {
            "Hindi": "hi",
            "English": "en",
            "Tamil": "ta",
            "Telugu": "te",
            "Kannada": "kn",
        }
        code = lang_map.get(lang, lang)

        if lang not in self.state_manager.state.translation:
            raise KeyError(f"Translation for {lang} ({code}) not found in state.")

        text = self.state_manager.state.translation[lang].output_text
        if not text:
            raise ValueError(f"Translated text for {lang} not available for TTS.")

        files = stage_filenames(project_dir, base_name, lang)
        settings = self.state_manager.state.settings
        stage_name = f"tts_{lang}"
        self.state_manager.update_stage(stage_name, status="running", start_time=time.time())

        tts_time = synthesize_tts(
            text=text,
            ref_audio_tuple=self._temp_ref_audio,
            ref_text=settings.get('ref_text'),
            out_path=files["tts"],
            use_emotion_refs=settings.get('use_emotion_refs', False),
            progress_callback=progress_cb
        )

        self.state_manager.update_stage(
            stage_name,
            status="completed",
            output_file=str(files["tts"]),
            time_taken=tts_time,
            end_time=time.time()
        )

    # -------------------- Config / Summary --------------------

    def _get_engine_config(self) -> EngineConfig:
        """Build EngineConfig from current settings."""
        engine_name = self.state_manager.state.settings.get('engine_name')
        cfg_raw = ENGINES.get(engine_name, list(ENGINES.values())[0] if ENGINES else {})
        return EngineConfig(
            url=cfg_raw.get("url", ""),
            #login_required=cfg_raw.get("login_required", True),
            copy_btn_coords=tuple(cfg_raw.get("copy_btn_coords", (0, 0))),
            copy_btn_image_path=cfg_raw.get("copy_btn_image_path")
        )

    def get_summary(self) -> Dict[str, Any]:
        """Produce a concise summary for UI."""
        state = self.state_manager.state
        summary = {
            "Project Folder": state.project_dir,
            "Total Time": secfmt(time.time() - state.created_at if state.created_at else 0),
            "ASR Time": secfmt(state.asr.time_taken),
            "Cleaning Time": secfmt(state.clean_text.time_taken),
            "Languages": {}
        }
        for lang in state.settings.get('target_langs', []):
            trans_state = state.translation.get(lang, StageState())
            tts_state = state.tts.get(lang, StageState())
            summary["Languages"][lang] = {
                "Translation Time": secfmt(trans_state.time_taken),
                "TTS Time": secfmt(tts_state.time_taken)
            }
        return summary

    # -------------------- Batch Processing --------------------

    def run_batch_processing(
        self,
        batch_folder_path: str,
        settings: Dict[str, Any],
        max_files: int,
        shuffle: bool,
        progress_callback: Callable
    ):
        """Configure and run batch jobs for all supported audio files in a folder."""
        audio_files = get_supported_audio_files(Path(batch_folder_path), config.SUPPORTED_AUDIO_FORMATS)
        if shuffle:
            random.shuffle(audio_files)
        audio_files = audio_files[: int(max_files)]

        if not audio_files:
            gr.Warning("No supported audio files found in the selected folder.")
            return {}

        self.batch_processor.add_jobs(audio_files, settings)
        return self.batch_processor.process_batch(progress_callback)


# Create global pipeline controller
pipeline_controller = PipelineController()


def create_ui():
    """
    Build and wire the Gradio UI for the S2TS pipeline.
    """

    custom_css = """
    .dark-neon {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
        line-height: 1.5;
    }
    .dark-neon .gr-block {
        background: rgba(30, 41, 59, 0.75);
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, box-shadow 0.3s ease;
    }
    .dark-neon .gr-block:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.35);
    }
    .dark-neon .gr-button-primary {
        background: linear-gradient(135deg, #0ea5e9, #0369a1);
        border: none;
        color: white;
        border-radius: 10px;
        font-weight: 600;
        padding: 10px 18px;
        transition: all 0.25s ease;
        box-shadow: 0 0 6px rgba(14, 165, 233, 0.6);
    }
    .dark-neon .gr-button-primary:hover {
        background: linear-gradient(135deg, #38bdf8, #0284c7);
        box-shadow: 0 0 12px rgba(14, 165, 233, 0.9);
    }
    .dark-neon .gr-button-secondary {
        background: rgba(30, 41, 59, 0.95);
        border: 1px solid #334155;
        color: #e2e8f0;
        border-radius: 10px;
        padding: 10px 18px;
        transition: all 0.25s ease;
    }
    .dark-neon .gr-button-secondary:hover {
        border-color: #0ea5e9;
        color: #0ea5e9;
        box-shadow: 0 0 8px rgba(14, 165, 233, 0.6);
    }
    .dark-neon .gr-button-stop {
        background: linear-gradient(135deg, #ef4444, #b91c1c);
        border: none;
        color: white;
        border-radius: 10px;
        font-weight: 600;
        padding: 10px 18px;
        transition: all 0.25s ease;
        box-shadow: 0 0 6px rgba(239, 68, 68, 0.6);
    }
    .dark-neon .gr-button-stop:hover {
        background: linear-gradient(135deg, #f87171, #dc2626);
        box-shadow: 0 0 12px rgba(239, 68, 68, 0.9);
    }
    .dark-neon .gr-tabs { border-bottom: 1px solid #334155; padding-bottom: 4px; }
    .dark-neon .gr-tab-item {
        background: rgba(30, 41, 59, 0.65);
        border: 1px solid #334155;
        border-bottom: none;
        border-radius: 10px 10px 0 0;
        margin-right: 6px;
        padding: 8px 18px;
        transition: all 0.2s ease;
    }
    .dark-neon .gr-tab-item:hover { color: #38bdf8; border-color: #0ea5e9; }
    .dark-neon .gr-tab-item.selected {
        background: rgba(14, 165, 233, 0.25);
        border-color: #0ea5e9;
        color: #0ea5e9;
        font-weight: 600;
    }
    .dark-neon h1, .dark-neon h2, .dark-neon h3 {
        color: #0ea5e9;
        font-weight: 700;
        text-shadow: 0 0 6px rgba(14, 165, 233, 0.6);
    }
    .dark-neon .progress-bar {
        background: linear-gradient(90deg, #0ea5e9, #22d3ee);
        height: 8px;
        border-radius: 6px;
        box-shadow: 0 0 6px rgba(14, 165, 233, 0.7);
    }
    .dark-neon .resource-meter { background: #1e293b; border-radius: 6px; height: 16px; overflow: hidden; margin-top: 4px; }
    .dark-neon .resource-meter-fill { height: 100%; border-radius: 6px; transition: width 0.4s ease; }
    .dark-neon .log-container {
        background: #0f172a; border: 1px solid #334155; border-radius: 10px;
        padding: 14px; font-family: 'Fira Code', monospace; font-size: 13px;
        line-height: 1.4; max-height: 320px; overflow-y: auto;
        box-shadow: inset 0 0 8px rgba(14, 165, 233, 0.2);
    }
    .dark-neon .log-container::-webkit-scrollbar { width: 8px; }
    .dark-neon .log-container::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
    """

    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue=config.THEME_PRIMARY,
            secondary_hue=config.THEME_SECONDARY,
            neutral_hue=config.THEME_NEUTRAL
        ).set(
            body_background_fill=config.DARK_BG_GRADIENT,
            block_background_fill="#111827",
            block_title_text_color="#38bdf8",
            block_border_color="#334155",
            body_text_color="#d1d5db",
            button_primary_background_fill="#0ea5e9",
            button_primary_text_color="white",
            button_secondary_background_fill="#1e293b",
            button_secondary_text_color="#e5e7eb"
        ),
        css=custom_css,
        title=config.APP_TITLE
    ) as demo:

        # ===== Header =====
        gr.Markdown(
            f"""
            <div style="text-align:center; padding:24px; border-radius:18px;
                        background:rgba(15,23,42,0.92);
                        box-shadow:0 0 24px rgba(14,165,233,0.35);
                        margin-bottom:28px;">
                <h1 style="color:#38bdf8; margin:0; font-size:2.6em; font-weight:700; letter-spacing:0.5px;
                           text-shadow:0 0 14px rgba(14,165,233,0.7);">
                    {config.APP_TITLE}
                </h1>
                <p style="color:#94a3b8; margin-top:12px; font-size:1.15em; border-top:1px solid #334155; padding-top:10px;">
                    {config.APP_TAG}
                </p>
            </div>
            """
        )

        # ===== Resource Monitoring =====
        with gr.Accordion("📊 Resource Monitoring", open=True):
            with gr.Row():
                with gr.Column(scale=1):
                    cpu_usage = gr.HTML()
                with gr.Column(scale=1):
                    gpu_usage = gr.HTML()
                with gr.Column(scale=1):
                    memory_usage = gr.HTML()
                with gr.Column(scale=1):
                    disk_usage = gr.HTML()

        # ===== Mode Selection =====
        processing_mode = gr.Radio(
            choices=["Single File", "Batch Folder"],
            value="Single File",
            label="⚙️ Processing Mode",
            elem_classes="glow-radio"
        )

        # ===== Single File Mode =====
        with gr.Group(visible=True) as single_file_group:
            with gr.Row():
                with gr.Column(scale=1):
                    # --- Inputs ---
                    audio_in = gr.Audio(label="🎵 Upload Input Audio (Tamil)", type="filepath")

                    # Manual text shown only when ASR disabled
                    manual_text_group = gr.Group(visible=False)
                    with manual_text_group:
                        manual_text = gr.Textbox(lines=3, label="📝 Or Enter Tamil Text Manually")

                    # --- Stage Toggles ---
                    with gr.Row():
                        enable_asr = gr.Checkbox(value=True, label="Enable ASR")
                        enable_clean = gr.Checkbox(value=True, label="Enable Text Cleaning")
                    with gr.Row():
                        enable_translate = gr.Checkbox(value=True, label="Enable Translation")
                        enable_tts = gr.Checkbox(value=True, label="Enable TTS")

                    # --- Language & Engine ---
                    enable_langs = gr.CheckboxGroup(
                        ["Hindi", "Kannada", "Telugu"], value=["Hindi"], label="Target Languages"
                    )
                    engine = gr.Dropdown(
                        list(ENGINES.keys()), value=list(ENGINES.keys())[0] if ENGINES else None, label="AI Engine"
                    )
                    reload_btn = gr.Button("🔄 Reload Engines")

                    # --- TTS Reference Voice ---
                    with gr.Accordion("🎙️ Reference Voice (for TTS)", open=False) as ref_voice_group:
                        load_default_ref = gr.Checkbox(
                            label="Load Default Reference", value=False,
                            info="Auto-load reference from Reference-Txt-Aud folder"
                        )
                        enable_emotion_refs = gr.Checkbox(
                            label="Enable Emotion References", value=False,
                            info="Use the 'emotions' subfolder for tagged TTS (e.g., [happy] text...)"
                        )    
                        ref_audio = gr.Audio(type="numpy", label="Reference Audio (voice to clone)")
                        ref_text = gr.Textbox(lines=2, label="Reference Audio Transcript")

                with gr.Column(scale=1):
                    # --- Progress & Logs ---
                    overall_progress = gr.Slider(label="Overall Progress", interactive=False)
                    stage_progress = gr.Slider(label="Current Stage Progress", interactive=False)
                    log_display = gr.Textbox(label="📜 Progress Log", lines=15, interactive=False)

                    # --- Action Buttons ---
                    with gr.Row():
                        run_btn = gr.Button("🚀 Run Pipeline", variant="primary", scale=3)
                        stop_btn = gr.Button("⛔ Stop", elem_classes="gr-button-stop", scale=1)

        # ===== Batch Mode =====
        with gr.Group(visible=False) as batch_group:
            with gr.Row():
                with gr.Column(scale=1):
                    batch_folder = gr.File(
                        label="📂 Select Folder with Audio Files",
                        file_count="directory"
                    )

                    with gr.Row():
                        batch_max_files = gr.Slider(
                            minimum=1, maximum=500, value=100,
                            label="Maximum Files to Process"
                        )
                        batch_shuffle = gr.Checkbox(label="Shuffle Processing Order", value=False)

                    with gr.Accordion("⚙️ Batch Processing Settings", open=False):
                        with gr.Row():
                            batch_enable_asr = gr.Checkbox(value=True, label="Enable Transcription")
                            batch_enable_clean = gr.Checkbox(value=True, label="Enable Text Cleaning")
                        with gr.Row():
                            batch_enable_translate = gr.Checkbox(value=True, label="Enable Text Translation")
                            batch_enable_tts = gr.Checkbox(value=True, label="Enable TTS")
                        with gr.Row():
                            batch_langs = gr.CheckboxGroup(
                                ["Hindi", "Kannada", "Telugu"], value=["Hindi"], label="Target Languages"
                            )

                        batch_engine = gr.Dropdown(
                            list(ENGINES.keys()),
                            value=list(ENGINES.keys())[0] if ENGINES else None,
                            label="AI Engine"
                        )
                        with gr.Row():
                            batch_load_default_ref = gr.Checkbox(
                                label="Load Default Reference", value=False,
                                    info="Auto-load reference from Reference-Txt-Aud folder"
                            )
                            batch_enable_emotion_refs = gr.Checkbox(
                                label="Enable Emotion References", value=False,
                                info="Use the 'emotions' subfolder for tagged TTS (e.g., [happy] text...)"
                            ) 
                        with gr.Row():    
                            batch_ref_audio = gr.Audio(type="numpy", label="Reference Audio (voice to clone)")
                        with gr.Row():    
                            batch_ref_text = gr.Textbox(lines=2, label="Reference Audio Transcript")
                        

                with gr.Column(scale=1):
                    batch_overall_progress = gr.Slider(0, 100, value=0, label="Batch Progress", interactive=False)
                    batch_current_file = gr.Textbox(label="Currently Processing", value="", interactive=False)

                    with gr.Row():
                        batch_processed = gr.Number(label="Processed", value=0, interactive=False)
                        batch_total = gr.Number(label="Total", value=0, interactive=False)

                    batch_log = gr.Textbox(
                        label="📜 Batch Log",
                        value="",
                        interactive=False,
                        lines=10,
                        elem_classes="log-container"
                    )

                    with gr.Row():
                        batch_run_btn = gr.Button("🚀 Run Batch Processing", variant="primary", scale=3)
                        batch_stop_btn = gr.Button("⛔ Stop Batch", elem_classes="gr-button-stop", scale=1)

        # ===== Outputs, Summary, and Pipeline Controls =====
        with gr.Accordion("📤 Outputs & Controls", open=True):
            with gr.Row():
                resume_btn = gr.Button("🔁 Resume Pipeline", variant="primary")
                reset_btn = gr.Button("🔄 Reset State")
            status_display = gr.JSON(label="Current Pipeline Status")
            summary = gr.JSON(label="⏱️ Timing Summary")

            # Per-language outputs
            with gr.Tabs():
                with gr.TabItem("Texts"):
                    with gr.Row():
                        asr_txt_file = gr.File(label="ASR Text")
                        clean_txt_file = gr.File(label="Cleaned Tamil Text")
                with gr.TabItem("Hindi"):
                    with gr.Row():
                        hi_trans_file = gr.File(label="Hindi Translation")
                        hi_tts_file = gr.Audio(label="Hindi TTS Audio", type="filepath")
                with gr.TabItem("Kannada"):
                    with gr.Row():
                        kn_trans_file = gr.File(label="Kannada Translation")
                        kn_tts_file = gr.Audio(label="Kannada TTS Audio", type="filepath")
                with gr.TabItem("Telugu"):
                    with gr.Row():
                        te_trans_file = gr.File(label="Telugu Translation")
                        te_tts_file = gr.Audio(label="Telugu TTS Audio", type="filepath")

        # ========= UI Logic =========

        def toggle_processing_mode(mode):
            """Switch between Single-file and Batch modes, and show/hide reference-voice section."""
            if mode == "Single File":
                return [
                    gr.update(visible=True),   # single_file_group
                    gr.update(visible=False),  # batch_group
                    gr.update(visible=True)    # ref_voice_group
                ]
            else:
                return [
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=False)
                ]

        processing_mode.change(
            fn=toggle_processing_mode,
            inputs=[processing_mode],
            outputs=[single_file_group, batch_group, ref_voice_group]
        )

        # ========= UI Event Handlers =========

        def toggle_manual_input_handler(is_asr_enabled: bool):
            """Show manual-text input only when ASR is disabled."""
            return gr.update(visible=not is_asr_enabled)

        def load_default_references_handler():
            """Load default TTS reference files."""
            text, audio_path = load_default_references(
                config.DEFAULT_REF_TEXT_FILE, config.DEFAULT_REF_AUDIO_FILE
            )
            return text, audio_path

        def reload_engines_handler():
            """Reload engine configurations from engines.json."""
            global ENGINES
            ENGINES = load_engines()
            choices = list(ENGINES.keys())
            value = choices[0] if choices else None
            return gr.update(choices=choices, value=value), gr.update(choices=choices, value=value)

        def update_resource_metrics_handler():
            """Continuously update resource metrics in a loop."""
            while True:
                metrics = resource_monitor.get_metrics()
                # Yield a tuple of the new values for the outputs
                yield (
                    metrics["cpu_html"], 
                    metrics["gpu_html"], 
                    metrics["memory_html"], 
                    metrics["disk_html"]
                )
                time.sleep(2)

        def stop_pipeline_handler():
            """Stop current pipeline or batch gracefully."""
            pipeline_controller.request_stop()
            gr.Info("Stop request sent. Current task will finish before stopping.")

        def update_status_display():
            """Continuously update the pipeline status display."""
            while True:
                if not pipeline_controller.state_manager.state.project_dir:
                    yield {"status": "Not initialized. Run the pipeline to begin."}
                else:
                    yield asdict(pipeline_controller.state_manager.state)
                time.sleep(5) # Update every 5 seconds

        # ---- Streaming Handlers ----

        def resume_pipeline_handler(progress=gr.Progress(track_tqdm=True)):
            """
            Resume pipeline from last failed/pending stage.
            Note: progress updates are shown via gr.Progress; logs update on start/end.
            """
            if not pipeline_controller.state_manager.can_resume():
                gr.Warning("No valid state to resume from. Please start a new run.")
                return

            log_messages = [f"[{now_hhmmss()}] Resuming pipeline from last failed/pending stage..."]
            yield {log_display: "\n".join(log_messages)}

            def progress_callback(stage_idx: int, stage_prog: int, message: str):
                total_stages = 4
                overall_prog = int(((stage_idx * 100) + stage_prog) / total_stages)
                progress(overall_prog / 100, desc=message)
                # (We avoid yielding from the callback; Gradio expects yields only in the outer handler.)

            pipeline_controller.run_full_pipeline_from_resume(progress_callback)

            summary_result = pipeline_controller.get_summary()
            log_messages.append(f"[{now_hhmmss()}] Resumed pipeline finished.")
            yield {log_display: "\n".join(log_messages), summary: summary_result, overall_progress: 100}

        def reset_pipeline_handler():
            """Reset state and clear UI outputs."""
            pipeline_controller.state_manager.reset_state()
            gr.Info("Pipeline state has been reset.")
            return {
                log_display: "Pipeline state reset. Ready for a new run.",
                status_display: {"status": "Reset"},
                overall_progress: 0,
                stage_progress: 0,
                summary: {},
                asr_txt_file: None,
                clean_txt_file: None,
                hi_trans_file: None,
                hi_tts_file: None,
                kn_trans_file: None,
                kn_tts_file: None,
                te_trans_file: None,
                te_tts_file: None,
            }

        def execute_pipeline_handler(
            audio_path,
            manual_text_str,
            en_asr,
            en_clean,
            en_trans,
            en_tts,
            langs,
            engine_name,
            ref_audio_np,
            ref_text_str,
            en_emotion_refs,
            progress=gr.Progress(track_tqdm=True)
        ):
            """
            Main handler for Single-file 'Run Pipeline'.
            Streams an initial state and a final state; progress bar updates live.
            """
            settings = {
                'enable_asr': en_asr,
                'enable_clean': en_clean,
                'enable_translate': en_trans,
                'enable_tts': en_tts,
                'target_langs': langs or [],
                'engine_name': engine_name,
                'ref_audio_numpy': ref_audio_np,
                'ref_text': ref_text_str,
                'manual_text': manual_text_str if not en_asr else "",
                'use_emotion_refs': en_emotion_refs
            }

            log_messages = ["[INITIALIZING] Preparing pipeline..."]
            yield {
                log_display: "\n".join(log_messages),
                overall_progress: 0,
                stage_progress: 0,
                summary: {},
                asr_txt_file: None,
                clean_txt_file: None,
                hi_trans_file: None,
                hi_tts_file: None,
                kn_trans_file: None,
                kn_tts_file: None,
                te_trans_file: None,
                te_tts_file: None,
            }

            def progress_callback(stage_idx: int, stage_prog: int, message: str):
                total_stages = 4
                overall_prog = int(((stage_idx * 100) + stage_prog) / total_stages)
                progress(overall_prog / 100, desc=message)
                # (No yield here; we update live via progress and final yield below.)

            summary_result = pipeline_controller.run_pipeline(audio_path, settings, progress_callback)

            # Map language → components for safe updates (avoid globals())
            widget_map = {
                "Hindi": {"trans": hi_trans_file, "tts": hi_tts_file},
                "Kannada": {"trans": kn_trans_file, "tts": kn_tts_file},
                "Telugu": {"trans": te_trans_file, "tts": te_tts_file},
            }

            log_messages.append(f"[{now_hhmmss()}] Pipeline finished. Finalizing outputs.")
            final_updates = {
                log_display: "\n".join(log_messages),
                overall_progress: 100,
                stage_progress: 100,
                summary: summary_result
            }

            # Add produced files to outputs if present
            state = pipeline_controller.state_manager.state
            if state.asr.output_file:
                final_updates[asr_txt_file] = gr.update(value=state.asr.output_file, visible=True)
            if state.clean_text.output_file:
                final_updates[clean_txt_file] = gr.update(value=state.clean_text.output_file, visible=True)

            for lang, widgets in widget_map.items():
                if lang in state.translation and state.translation[lang].output_file:
                    final_updates[widgets["trans"]] = gr.update(
                        value=state.translation[lang].output_file, visible=True
                    )
                if lang in state.tts and state.tts[lang].output_file:
                    final_updates[widgets["tts"]] = gr.update(
                        value=state.tts[lang].output_file, visible=True
                    )

            yield final_updates

        def _extract_directory_path(dir_input) -> str:
            """
            Normalize Gradio directory file input across variants (list/dict/obj).
            Returns a filesystem path string or "".
            """
            if not dir_input:
                return ""
            # If it's a list, take the first element (some builds wrap dict/obj in a list)
            candidate = dir_input[0] if isinstance(dir_input, (list, tuple)) else dir_input
            # Dict-like
            if isinstance(candidate, dict):
                # For Gradio directory uploads, the 'name' is the path to a file inside the directory
                path_str = candidate.get("name", "") or candidate.get("orig_name", "") or ""
                if path_str:
                    return str(Path(path_str).parent)
                return ""
            # File-like (SimpleNamespace / tempfile)
            name = getattr(candidate, "name", "")
            if isinstance(name, str) and name:
                return str(Path(name).parent)
            # Raw string path
            if isinstance(candidate, str):
                # It's possible to get a direct path to a file, so get its parent
                return str(Path(candidate).parent)
            return ""

# In New S2TS/main.py

        def execute_batch_handler(
            batch_folder_input, # This is a list of file-like objects from Gradio
            max_files,
            shuffle,
            batch_en_asr,
            batch_en_clean,
            batch_en_trans,
            batch_en_tts,
            batch_langs_list,
            batch_engine_name,
            batch_ref_audio_np,      
            batch_ref_text_str,  
            batch_en_emotion_refs,    
            progress=gr.Progress(track_tqdm=True)
        ):
            settings = {
                'enable_asr': bool(batch_en_asr),
                'enable_clean': bool(batch_en_clean),
                'enable_translate': bool(batch_en_trans),
                'enable_tts': bool(batch_en_tts),
                'target_langs': batch_langs_list or [],
                'engine_name': batch_engine_name,
                'ref_audio_numpy': batch_ref_audio_np, 
                'ref_text': batch_ref_text_str,  
                'use_emotion_refs': bool(batch_en_emotion_refs),      
                'manual_text': ""
            }

            if not batch_folder_input:
                gr.Warning("No files uploaded for batch processing.")
                return
            
            is_asr_enabled = bool(batch_en_asr)
            all_files = [Path(f.name) for f in batch_folder_input]
            
            if is_asr_enabled:
                log.info("ASR enabled, filtering for supported audio files.")
                files_to_process = [f for f in all_files if f.suffix.lower() in config.SUPPORTED_AUDIO_FORMATS]
            else:
                log.info("ASR disabled, filtering for supported text files (.txt).")
                files_to_process = [f for f in all_files if f.suffix.lower() in ['.txt', '.text']]
            
            if not files_to_process:
                gr.Warning("No files of the required type (audio or .txt) were found in the upload.")
                return


            # 2. Apply user settings for shuffle and max files
            if shuffle:
                random.shuffle(files_to_process)
            files_to_process = files_to_process[:int(max_files)]
            
            total_count = len(files_to_process)
            log_messages = [f"[{now_hhmmss()}] [BATCH INIT] Found {total_count} files. Starting processing..."]
            processed_count = 0
            
            # 3. Initial UI update with the correct total count
            yield {
                batch_log: "\n".join(log_messages),
                batch_total: total_count,
                batch_processed: 0,
                batch_overall_progress: 0
            }

            # 4. Main processing loop, controlled directly by the Gradio handler
            for file_path in files_to_process:
                # Check for stop request before processing each file
                if pipeline_controller._stop_requested:
                    log.warning("Batch processing was stopped by the user.")
                    log_messages.append(f"[{now_hhmmss()}] --- BATCH STOPPED BY USER ---")
                    break

                processed_count += 1
                message = f"File {processed_count}/{total_count}: Processing {file_path.name}"
                progress(processed_count / total_count, desc=message)
                log_messages.append(f"[{now_hhmmss()}] START: {file_path.name}")
                
                # Update UI to show which file is currently being processed
                yield {
                    batch_log: "\n".join(log_messages),
                    batch_current_file: file_path.name,
                    batch_processed: processed_count - 1,
                    batch_overall_progress: int((processed_count / total_count) * 100)
                }

                try:
                    # --- CORE LOGIC: Run the main pipeline for a SINGLE file ---
                    pipeline_controller.run_pipeline(
                        input_path=str(file_path),
                        settings=settings,
                        progress_callback=None # Inner progress bar is not needed for batch mode
                    )
                    log_messages.append(f"[{now_hhmmss()}] DONE:  {file_path.name}")

                except Exception as e:
                    log.error(f"Error processing {file_path.name}: {e}", exc_info=True)
                    log_messages.append(f"[{now_hhmmss()}] ERROR: Failed to process {file_path.name}: {e}")

            # 5. Final UI update after the loop finishes
            log_messages.append(f"[{now_hhmmss()}] BATCH COMPLETE.")
            yield {
                batch_log: "\n".join(log_messages),
                batch_overall_progress: 100,
                batch_current_file: "Batch processing finished.",
                batch_processed: processed_count
            }

        # --- Wire UI events to handlers ---

        enable_asr.change(fn=toggle_manual_input_handler, inputs=enable_asr, outputs=manual_text_group)

        run_btn.click(
            fn=execute_pipeline_handler,
            inputs=[
                audio_in, manual_text, enable_asr, enable_clean, enable_translate, enable_tts,
                enable_langs, engine, ref_audio, ref_text, enable_emotion_refs
            ],
            outputs=[
                log_display, overall_progress, stage_progress, summary,
                asr_txt_file, clean_txt_file,
                hi_trans_file, hi_tts_file,
                kn_trans_file, kn_tts_file,
                te_trans_file, te_tts_file
            ]
        )

        batch_run_btn.click(
            fn=execute_batch_handler,
            inputs=[
                batch_folder, batch_max_files, batch_shuffle,
                batch_enable_asr, batch_enable_clean, batch_enable_translate, batch_enable_tts,
                batch_langs, batch_engine, batch_ref_audio, batch_ref_text, batch_enable_emotion_refs
            ],
            outputs=[batch_log, batch_overall_progress, batch_current_file, batch_processed, batch_total]
        )

        stop_btn.click(fn=stop_pipeline_handler)
        batch_stop_btn.click(fn=stop_pipeline_handler)

        load_default_ref.change(fn=load_default_references_handler, outputs=[ref_text, ref_audio])
        batch_load_default_ref.change(fn=load_default_references_handler, outputs=[batch_ref_text, batch_ref_audio])
        reload_btn.click(fn=reload_engines_handler, outputs=[engine, batch_engine])

        resume_btn.click(fn=resume_pipeline_handler, outputs=[log_display, summary, overall_progress])

        reset_btn.click(
            fn=reset_pipeline_handler,
            outputs=[
                log_display, status_display, overall_progress, stage_progress, summary,
                asr_txt_file, clean_txt_file,
                hi_trans_file, hi_tts_file,
                kn_trans_file, kn_tts_file,
                te_trans_file, te_tts_file
            ]
        )

        demo.load(fn=update_resource_metrics_handler, outputs=[cpu_usage, gpu_usage, memory_usage, disk_usage,])
        demo.load(fn=update_status_display, outputs=[status_display ])

    return demo


# --- Application Entry Point ---
if __name__ == "__main__":
    # Ensure required directories/files exist
    ensure_dir(config.PROJECTS_DIR)
    ensure_dir(config.PROMPTS_DIR)
    ensure_dir(config.REFERENCE_DIR)
    ensure_dir(config.LOG_DIR)
    ensure_dir(config.ASR_TRANSCRIPTIONS_DIR)

    if not config.CORRECTOR_PROMPT_FILE.exists():
        write_text(config.CORRECTOR_PROMPT_FILE, config.DEFAULT_CORRECTOR_PROMPT)
    if not config.TRANSLATOR_PROMPT_FILE.exists():
        write_text(config.TRANSLATOR_PROMPT_FILE, config.DEFAULT_TRANSLATOR_PROMPT)

    # Launch resource monitoring and Gradio app
    resource_monitor.start_monitoring(interval=5)
    app_ui = create_ui()
    app_ui.launch(server_port=7860, inbrowser=False, show_error=True)
