"""
Transliterate JP→ES: Real-time Japanese audio to Spanish subtitle translator.

Captures system audio (loopback), transcribes Japanese speech using local
faster-whisper or OpenAI Whisper API, and translates to Spanish using
OpenAI GPT with streaming output.
"""

from __future__ import annotations

import io
import os
import threading
import time
import warnings
import wave
from collections import deque
from pathlib import Path

# Suppress huggingface_hub symlink warning on Windows
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Suppress soundcard "data discontinuity" warnings (normal during loopback capture)
warnings.filterwarnings("ignore", message="data discontinuity", module="soundcard")

import customtkinter as ctk
import numpy as np

try:
    import soundcard as sc
    _sc_error = None
except Exception as _e:
    sc = None
    _sc_error = str(_e)

# ---------------------------------------------------------------------------
# Register NVIDIA DLL directories so Windows can find cublas64_12.dll, etc.
# pip-installed nvidia-* packages place DLLs under site-packages/nvidia/…/bin
# which is NOT on the default DLL search path.
# ---------------------------------------------------------------------------
if os.name == "nt":
    import importlib.util as _ilu
    for _pkg in ("nvidia.cublas", "nvidia.cudnn"):
        _spec = _ilu.find_spec(_pkg)
        if _spec and _spec.submodule_search_locations:
            for _loc in _spec.submodule_search_locations:
                _bin = os.path.join(_loc, "bin")
                if os.path.isdir(_bin):
                    os.add_dll_directory(_bin)

try:
    from faster_whisper import WhisperModel
    _fw_available = True
except ImportError:
    _fw_available = False

from openai import OpenAI


# ---------------------------------------------------------------------------
# Device detection for faster-whisper (CUDA / CPU)
# ---------------------------------------------------------------------------

def _detect_device() -> tuple[str, str]:
    """Return (device, compute_type) for faster-whisper."""
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


# ---------------------------------------------------------------------------
# STT / Translation method constants
# ---------------------------------------------------------------------------

STT_METHODS = [
    "Whisper Local (GPU)",
    "Whisper Local (CPU)",
    "OpenAI Whisper API",
]
WHISPER_MODELS = ["large-v3", "medium", "small", "base", "tiny"]

_DEFAULT_TRANSLATION_MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4.1",
    "gpt-5-mini-2025-08-07",
    "gpt-5.4-2026-03-05",
]


def _load_translation_models() -> list[str]:
    """Load translation models from models.txt next to this script, falling back to defaults."""
    models_path = Path(__file__).parent / "models.txt"
    if models_path.is_file():
        models = [
            line.strip()
            for line in models_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if models:
            return models
    return list(_DEFAULT_TRANSLATION_MODELS)


TRANSLATION_MODELS = _load_translation_models()


# ---------------------------------------------------------------------------
# Audio capture (system loopback via soundcard)
# ---------------------------------------------------------------------------

class AudioCapture:
    """Captures system audio using loopback recording.

    Pipe workers call ``record_chunk()`` directly.  An internal lock
    ensures only one thread records at a time, but the lock is released
    as soon as the chunk is captured so the next pipe can start
    immediately while the previous one processes.
    """

    SAMPLE_RATE = 16000
    CHANNELS = 1
    CHUNK_SECONDS = 4

    def __init__(self):
        self._running = False
        self._loopback = None
        self._recorder = None
        self._mic_lock = threading.Lock()
        self._chunk_count = 0

    @staticmethod
    def list_loopback_devices() -> list[str]:
        if sc is None:
            return []
        try:
            speakers = sc.all_speakers()
            return [s.name for s in speakers]
        except Exception:
            return []

    def start(self, device_name: str | None = None):
        """Open the loopback mic so pipe workers can call record_chunk()."""
        if self._running:
            return

        if sc is None:
            raise RuntimeError(
                "soundcard no está disponible. Instálalo con: pip install soundcard"
            )

        # Find the speaker, then get its loopback microphone
        if device_name:
            speakers = sc.all_speakers()
            match = [s for s in speakers if s.name == device_name]
            speaker = match[0] if match else sc.default_speaker()
        else:
            speaker = sc.default_speaker()

        self._loopback = sc.get_microphone(
            speaker.id, include_loopback=True
        )

        self._recorder = self._loopback.recorder(
            samplerate=self.SAMPLE_RATE,
            channels=self.CHANNELS,
            blocksize=1024,
        )
        self._recorder.__enter__()
        self._running = True
        self._chunk_count = 0

    def stop(self):
        self._running = False
        if self._recorder:
            try:
                self._recorder.__exit__(None, None, None)
            except Exception:
                pass
            self._recorder = None

    def record_chunk(self, on_start=None):
        """Record one chunk (blocking).  Called by pipe worker threads.

        *on_start* is an optional callback invoked with ``(rec_start,)``
        the instant this thread acquires the mic and begins recording,
        so the caller can light up its semaphore / start its timer.

        Returns ``(wav_bytes, rec_start, rms)`` or ``None`` if stopped
        or the audio is silent.
        """
        num_frames = self.SAMPLE_RATE * self.CHUNK_SECONDS

        with self._mic_lock:
            if not self._running:
                return None
            rec_start = time.monotonic()
            if on_start:
                on_start(rec_start)
            try:
                data = self._recorder.record(numframes=num_frames)
            except Exception:
                if not self._running:
                    return None
                raise

        self._chunk_count += 1
        audio_float = data[:, 0] if data.ndim > 1 else data
        rms = float(np.sqrt(np.mean(audio_float ** 2)))

        if rms < 0.001:
            return None  # silence

        wav_bytes = self._float_to_wav(audio_float)
        return wav_bytes, rec_start, rms

    def _float_to_wav(self, audio: np.ndarray) -> bytes:
        audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(self.SAMPLE_RATE)
            wf.writeframes(audio_int16.tobytes())
        buf.seek(0)
        return buf.read()


# ---------------------------------------------------------------------------
# Services: STT + OpenAI GPT streaming translation
# ---------------------------------------------------------------------------

class TranscriptionService:
    """Handles STT (local or API) + OpenAI GPT streaming translation."""

    def __init__(self, api_key: str, whisper_model: str, translation_model: str,
                 stt_method: str = "Whisper Local (GPU)"):
        self.client = OpenAI(api_key=api_key)
        self.whisper_model_name = whisper_model
        self.translation_model = translation_model
        self.stt_method = stt_method
        self._context: deque[str] = deque(maxlen=5)
        self._whisper: WhisperModel | None = None

    def load_whisper(self) -> str:
        """Load the STT backend. Returns a status string describing what was loaded."""
        if self.stt_method == "OpenAI Whisper API":
            return "OpenAI Whisper API (nube)"

        # Local faster-whisper
        if not _fw_available:
            raise RuntimeError(
                "faster-whisper no está instalado. "
                "Instálalo con: pip install faster-whisper"
            )

        if self.stt_method == "Whisper Local (GPU)":
            device, compute_type = _detect_device()
            if device == "cuda":
                try:
                    self._whisper = WhisperModel(
                        self.whisper_model_name,
                        device="cuda",
                        compute_type="float16",
                    )
                    return f"faster-whisper {self.whisper_model_name} (cuda, float16)"
                except RuntimeError:
                    pass
            # GPU requested but unavailable — fail explicitly so user knows
            raise RuntimeError(
                "No se pudo inicializar CUDA. Verifica que tienes GPU compatible "
                "y las librerías CUDA instaladas (pip install nvidia-cublas-cu12 "
                "nvidia-cudnn-cu12). O selecciona 'Whisper Local (CPU)'."
            )

        # CPU mode
        self._whisper = WhisperModel(
            self.whisper_model_name,
            device="cpu",
            compute_type="int8",
        )
        return f"faster-whisper {self.whisper_model_name} (cpu, int8)"

    def update_translation_model(self, translation_model: str):
        self.translation_model = translation_model

    def transcribe(self, wav_bytes: bytes) -> str:
        if self.stt_method == "OpenAI Whisper API":
            return self._transcribe_api(wav_bytes)
        return self._transcribe_local(wav_bytes)

    def _transcribe_local(self, wav_bytes: bytes) -> str:
        segments, _info = self._whisper.transcribe(
            io.BytesIO(wav_bytes),
            language="ja",
            beam_size=5,
        )
        return "".join(s.text for s in segments).strip()

    def _transcribe_api(self, wav_bytes: bytes) -> str:
        wav_file = io.BytesIO(wav_bytes)
        wav_file.name = "audio.wav"
        response = self.client.audio.transcriptions.create(
            model="whisper-1",
            file=wav_file,
            language="ja",
        )
        return response.text.strip()

    def translate_stream(self, japanese_text: str):
        """Yield translation tokens as they stream from GPT."""
        if not japanese_text:
            return

        context_str = "\n".join(self._context) if self._context else "(sin contexto previo)"

        stream = self.client.chat.completions.create(
            model=self.translation_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Eres un traductor profesional de japonés a español. "
                        "Traduce el texto japonés al español de forma natural y fluida. "
                        "Mantén el tono y registro del original. "
                        "Si el texto contiene onomatopeyas o expresiones culturales japonesas, "
                        "adapta al equivalente más cercano en español. "
                        "Responde SOLO con la traducción, sin explicaciones."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Contexto previo de la conversación:\n{context_str}\n\n"
                        f"Traduce al español:\n{japanese_text}"
                    ),
                },
            ],
            temperature=0.3,
            max_completion_tokens=500,
            stream=True,
        )

        full_text: list[str] = []
        for chunk in stream:
            delta = chunk.choices[0].delta
            token = delta.content if delta and delta.content else ""
            if token:
                full_text.append(token)
                yield token

        translation = "".join(full_text)
        self._context.append(f"JP: {japanese_text}\nES: {translation}")


# ---------------------------------------------------------------------------
# Main Application UI
# ---------------------------------------------------------------------------

class SubtitleApp(ctk.CTk):
    """Main application window."""

    def __init__(self):
        super().__init__()

        self.title("Transliterate JP→ES")
        self.geometry("900x700")
        self.minsize(700, 550)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self._audio = AudioCapture()
        self._service: TranscriptionService | None = None
        self._is_running = False
        self._debug_enabled = False

        # 3 parallel full-chain pipelines (rec → STT → translate)
        self.NUM_PIPES = 3
        self._pipe_threads: list[threading.Thread] = []

        # Locks for shared resources
        self._stt_lock = threading.Lock()       # Whisper model is not thread-safe
        self._display_lock = threading.Lock()   # serialize subtitle display output

        # Per-pipe timer state
        self._pipe_timer_starts: list[float | None] = [None] * self.NUM_PIPES
        self._timer_after_id: str | None = None

        # Flag: STT backend loaded (pipes wait for this before processing)
        self._stt_ready = threading.Event()

        self._build_ui()
        self._load_api_key()

    # -- UI Construction ---------------------------------------------------

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(4, weight=1)

        # -- Config frame (API Key) --
        config_frame = ctk.CTkFrame(self)
        config_frame.grid(row=0, column=0, padx=12, pady=(12, 6), sticky="ew")
        config_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(config_frame, text="API Key:").grid(
            row=0, column=0, padx=(12, 6), pady=8, sticky="w"
        )
        self._api_key_entry = ctk.CTkEntry(
            config_frame, placeholder_text="sk-... (OpenAI)", show="•", width=350
        )
        self._api_key_entry.grid(row=0, column=1, padx=6, pady=8, sticky="ew")

        self._show_key_var = ctk.BooleanVar(value=False)
        self._show_key_btn = ctk.CTkCheckBox(
            config_frame,
            text="Mostrar",
            variable=self._show_key_var,
            command=self._toggle_key_visibility,
            width=80,
        )
        self._show_key_btn.grid(row=0, column=2, padx=(0, 12), pady=8)

        # -- STT + Translation method frame --
        method_frame = ctk.CTkFrame(self)
        method_frame.grid(row=1, column=0, padx=12, pady=6, sticky="ew")
        method_frame.grid_columnconfigure(1, weight=1)
        method_frame.grid_columnconfigure(3, weight=1)

        # STT method selector
        ctk.CTkLabel(method_frame, text="Transcripción:").grid(
            row=0, column=0, padx=(12, 6), pady=8, sticky="w"
        )
        self._stt_combo = ctk.CTkComboBox(
            method_frame, values=STT_METHODS, state="readonly", width=200,
            command=self._on_stt_method_changed,
        )
        self._stt_combo.set(STT_METHODS[0])
        self._stt_combo.grid(row=0, column=1, padx=6, pady=8, sticky="w")

        # Translation model (GPT)
        ctk.CTkLabel(method_frame, text="Traducción (GPT):").grid(
            row=0, column=2, padx=(24, 6), pady=8, sticky="w"
        )
        self._trans_combo = ctk.CTkComboBox(
            method_frame, values=TRANSLATION_MODELS, state="readonly", width=180
        )
        self._trans_combo.set(TRANSLATION_MODELS[0])
        self._trans_combo.grid(row=0, column=3, padx=6, pady=8, sticky="w")

        # -- Whisper model selector (row 2, only for local modes) --
        self._whisper_frame = ctk.CTkFrame(self)
        self._whisper_frame.grid(row=2, column=0, padx=12, pady=6, sticky="ew")
        self._whisper_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(self._whisper_frame, text="Modelo Whisper:").grid(
            row=0, column=0, padx=(12, 6), pady=8, sticky="w"
        )
        self._whisper_combo = ctk.CTkComboBox(
            self._whisper_frame, values=WHISPER_MODELS, state="readonly", width=180
        )
        self._whisper_combo.set(WHISPER_MODELS[0])
        self._whisper_combo.grid(row=0, column=1, padx=6, pady=8, sticky="w")

        self._whisper_hint = ctk.CTkLabel(
            self._whisper_frame,
            text="",
            font=ctk.CTkFont(size=11),
            text_color="gray",
        )
        self._whisper_hint.grid(row=0, column=2, padx=(12, 12), pady=8, sticky="w")
        self._update_whisper_hint()

        # -- Audio device + controls --
        ctrl_frame = ctk.CTkFrame(self)
        ctrl_frame.grid(row=3, column=0, padx=12, pady=6, sticky="ew")
        ctrl_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(ctrl_frame, text="Dispositivo:").grid(
            row=0, column=0, padx=(12, 6), pady=8, sticky="w"
        )
        devices = AudioCapture.list_loopback_devices()
        device_display = devices if devices else ["(no disponible)"]
        self._device_combo = ctk.CTkComboBox(
            ctrl_frame, values=device_display, state="readonly", width=350
        )
        if devices:
            self._device_combo.set(devices[0])
        else:
            self._device_combo.set(device_display[0])
        self._device_combo.grid(row=0, column=1, padx=6, pady=8, sticky="ew")

        self._refresh_btn = ctk.CTkButton(
            ctrl_frame, text="⟳", width=36, command=self._refresh_devices
        )
        self._refresh_btn.grid(row=0, column=2, padx=4, pady=8)

        self._start_btn = ctk.CTkButton(
            ctrl_frame,
            text="▶  Iniciar",
            width=120,
            fg_color="#2d8a4e",
            hover_color="#236b3e",
            command=self._toggle_capture,
        )
        self._start_btn.grid(row=0, column=3, padx=(12, 12), pady=8)

        # -- Subtitle display area --
        subtitle_frame = ctk.CTkFrame(self)
        subtitle_frame.grid(row=4, column=0, padx=12, pady=(6, 12), sticky="nsew")
        subtitle_frame.grid_columnconfigure(0, weight=1)
        subtitle_frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            subtitle_frame,
            text="Subtítulos",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=0, column=0, padx=12, pady=(8, 0), sticky="w")

        self._subtitle_box = ctk.CTkTextbox(
            subtitle_frame,
            font=ctk.CTkFont(size=22),
            wrap="word",
            state="disabled",
        )
        self._subtitle_box.grid(row=1, column=0, padx=8, pady=8, sticky="nsew")

        # -- Pipeline semaphore + timer bar (3 pipes) --
        pipeline_frame = ctk.CTkFrame(self)
        pipeline_frame.grid(row=5, column=0, padx=12, pady=(0, 4), sticky="ew")
        pipeline_frame.grid_columnconfigure(7, weight=1)

        sem_font = ctk.CTkFont(size=11)
        dot_font = ctk.CTkFont(size=13)
        header_font = ctk.CTkFont(size=10, weight="bold")

        # Column headers
        for col, label in [(0, ""), (1, "Grab"), (3, "STT"), (5, "Trad"), (7, "Tiempo")]:
            ctk.CTkLabel(pipeline_frame, text=label, font=header_font).grid(
                row=0, column=col, padx=2, pady=(4, 0)
            )

        # Per-pipe semaphore rows (colored square indicators)
        self._pipe_sem_lights: list[dict[str, ctk.CTkFrame]] = []
        self._pipe_timer_labels: list[ctk.CTkLabel] = []

        _OFF_COLOR = "#555555"  # dark gray = inactive

        for i in range(self.NUM_PIPES):
            row = i + 1
            pipe_label = ctk.CTkLabel(
                pipeline_frame, text=f"P{i + 1}", font=sem_font, width=24
            )
            pipe_label.grid(row=row, column=0, padx=(8, 2), pady=3)

            lights: dict[str, ctk.CTkFrame] = {}
            for col, key in [(1, "rec"), (3, "stt"), (5, "trans")]:
                light = ctk.CTkFrame(
                    pipeline_frame,
                    width=16, height=16,
                    corner_radius=8,
                    fg_color=_OFF_COLOR,
                )
                light.grid(row=row, column=col, padx=4, pady=3)
                light.grid_propagate(False)
                lights[key] = light
                # Arrow separator between stages
                if col < 5:
                    ctk.CTkLabel(pipeline_frame, text="→", font=sem_font).grid(
                        row=row, column=col + 1, padx=0, pady=3
                    )

            self._pipe_sem_lights.append(lights)

            timer_lbl = ctk.CTkLabel(
                pipeline_frame,
                text="⏱ --",
                font=ctk.CTkFont(size=11, weight="bold"),
                width=80,
            )
            timer_lbl.grid(row=row, column=7, padx=(4, 8), pady=2, sticky="e")
            self._pipe_timer_labels.append(timer_lbl)

        # -- Status bar --
        status_frame = ctk.CTkFrame(self, fg_color="transparent")
        status_frame.grid(row=6, column=0, padx=16, pady=(0, 8), sticky="ew")
        status_frame.grid_columnconfigure(0, weight=1)

        self._status_label = ctk.CTkLabel(
            status_frame,
            text="Estado: Detenido",
            font=ctk.CTkFont(size=12),
            anchor="w",
        )
        self._status_label.grid(row=0, column=0, sticky="w")

        self._debug_var = ctk.BooleanVar(value=False)
        self._debug_check = ctk.CTkCheckBox(
            status_frame,
            text="Debug",
            variable=self._debug_var,
            command=self._toggle_debug,
            width=70,
            font=ctk.CTkFont(size=11),
        )
        self._debug_check.grid(row=0, column=1, sticky="e")

    # -- STT method change handler -----------------------------------------

    def _on_stt_method_changed(self, _value: str = ""):
        stt = self._stt_combo.get()
        if stt == "OpenAI Whisper API":
            self._whisper_frame.grid_remove()
        else:
            self._whisper_frame.grid()
            self._update_whisper_hint()

    def _update_whisper_hint(self):
        stt = self._stt_combo.get()
        if stt == "Whisper Local (GPU)":
            self._whisper_hint.configure(text="Requiere GPU NVIDIA + CUDA")
        elif stt == "Whisper Local (CPU)":
            self._whisper_hint.configure(text="Sin GPU, más lento")
        else:
            self._whisper_hint.configure(text="")

    # -- Actions -----------------------------------------------------------

    def _load_api_key(self):
        """Load API key from apikey.txt if it exists next to the script."""
        key_path = Path(__file__).parent / "apikey.txt"
        if key_path.is_file():
            key = key_path.read_text(encoding="utf-8").strip()
            if key:
                self._api_key_entry.insert(0, key)

    def _toggle_debug(self):
        self._debug_enabled = self._debug_var.get()

    def _toggle_key_visibility(self):
        if self._show_key_var.get():
            self._api_key_entry.configure(show="")
        else:
            self._api_key_entry.configure(show="•")

    def _refresh_devices(self):
        devices = AudioCapture.list_loopback_devices()
        if devices:
            self._device_combo.configure(values=devices)
            self._device_combo.set(devices[0])
        else:
            self._device_combo.configure(values=["(no disponible)"])
            self._device_combo.set("(no disponible)")

    def _toggle_capture(self):
        if self._is_running:
            self._stop_capture()
        else:
            self._start_capture()

    def _start_capture(self):
        try:
            api_key = self._api_key_entry.get().strip()
            if not api_key:
                self._set_status("⚠ Ingresa tu API Key de OpenAI")
                return

            device_name = self._device_combo.get()
            self._log(f"[DEBUG] Dispositivo seleccionado: '{device_name}'")
            self._log(f"[DEBUG] soundcard disponible: {sc is not None}")
            if _sc_error:
                self._log(f"[DEBUG] Error al importar soundcard: {_sc_error}")

            if device_name == "(no disponible)":
                self._set_status("⚠ No hay dispositivos de audio disponibles")
                return

            stt_method = self._stt_combo.get()
            self._log(f"[DEBUG] Método STT: {stt_method}")

            self._log("[DEBUG] Creando TranscriptionService...")
            self._service = TranscriptionService(
                api_key=api_key,
                whisper_model=self._whisper_combo.get(),
                translation_model=self._trans_combo.get(),
                stt_method=stt_method,
            )
            self._log("[DEBUG] TranscriptionService creado OK")

            # Disable selectors while running
            self._stt_combo.configure(state="disabled")
            self._whisper_combo.configure(state="disabled")

            self._is_running = True
            self._stt_ready.clear()

            # Open mic (pipe workers will call record_chunk directly)
            self._log("[DEBUG] Abriendo dispositivo de audio...")
            try:
                self._audio.start(device_name)
            except Exception as e:
                self._is_running = False
                self._set_status(f"⚠ Error de audio: {e}")
                self._log(f"[DEBUG] Error en audio.start: {type(e).__name__}: {e}")
                self._stt_combo.configure(state="readonly")
                self._whisper_combo.configure(state="readonly")
                return

            # Load STT in a background thread, then launch pipe workers
            self._pipe_threads = []
            loader = threading.Thread(target=self._load_stt_and_start_pipes, daemon=True)
            loader.start()

            self._start_btn.configure(
                text="⏹  Detener", fg_color="#c0392b", hover_color="#962d22"
            )
            # Reset all pipe semaphores
            for pid in range(self.NUM_PIPES):
                self._set_pipe_semaphore(pid, "rec", False)
                self._set_pipe_semaphore(pid, "stt", False)
                self._set_pipe_semaphore(pid, "trans", False)
                self._reset_pipe_timer(pid)
            self._log("[DEBUG] Captura de audio iniciada")
        except Exception as e:
            self._set_status(f"⚠ Error inesperado: {e}")
            self._log(f"[DEBUG] Excepción no controlada: {type(e).__name__}: {e}")

    def _stop_capture(self):
        self._is_running = False
        self._stt_ready.set()  # unblock any waiting pipes
        self._audio.stop()     # releases mic lock waiters
        self._start_btn.configure(
            text="▶  Iniciar", fg_color="#2d8a4e", hover_color="#236b3e"
        )
        # Re-enable selectors
        self._stt_combo.configure(state="readonly")
        self._whisper_combo.configure(state="readonly")
        # Reset all pipe semaphores and timers
        for pid in range(self.NUM_PIPES):
            self._set_pipe_semaphore(pid, "rec", False)
            self._set_pipe_semaphore(pid, "stt", False)
            self._set_pipe_semaphore(pid, "trans", False)
            self._reset_pipe_timer(pid)
        self._set_status("Estado: Detenido")

    # -- Pipeline workers --------------------------------------------------

    def _load_stt_and_start_pipes(self):
        """Thread: loads STT backend, then launches pipe workers."""
        stt_method = self._service.stt_method
        self.after(0, self._set_status, f"⏳ Cargando STT ({stt_method})...")
        self.after(0, self._log, f"[DEBUG] Cargando STT: {stt_method}...")

        try:
            status = self._service.load_whisper()
        except Exception as e:
            self.after(0, self._log,
                       f"[ERROR] No se pudo cargar STT: {type(e).__name__}: {e}")
            self.after(0, self._stop_capture)
            return

        self.after(0, self._log, f"[DEBUG] STT listo: {status}")
        self.after(0, self._set_status, "🎧 Capturando audio del sistema...")
        self._stt_ready.set()

        # Launch 3 pipe workers
        for pid in range(self.NUM_PIPES):
            t = threading.Thread(target=self._pipe_worker, args=(pid,), daemon=True)
            t.start()
            self._pipe_threads.append(t)

    def _pipe_worker(self, pipe_id: int):
        """Thread: full pipeline — record → STT → translate → display.

        Each pipe records its own audio chunk by calling
        ``self._audio.record_chunk()``.  An internal mic lock inside
        AudioCapture ensures only one pipe records at a time, but the
        lock is released the instant recording finishes so the next
        pipe can start recording while this one does STT / translation.
        """
        tag = f"P{pipe_id + 1}"
        use_local_whisper = self._service.stt_method != "OpenAI Whisper API"

        # Wait for STT backend to be ready
        self._stt_ready.wait()
        if not self._is_running:
            return

        self.after(0, self._log, f"[DEBUG] {tag} worker listo")

        while self._is_running:
            # -- Recording phase (blocks until mic lock acquired + 4s audio) --
            def on_rec_start(t0, _pid=pipe_id):
                self.after(0, self._set_pipe_semaphore, _pid, "rec", True)
                self.after(0, self._start_pipe_timer, _pid, t0)

            try:
                result = self._audio.record_chunk(on_start=on_rec_start)
            except Exception as e:
                if not self._is_running:
                    break
                self.after(0, self._log,
                           f"[ERROR] {tag} Grabación: {type(e).__name__}: {e}")
                self.after(0, self._stop_pipe_timer, pipe_id)
                break

            self.after(0, self._set_pipe_semaphore, pipe_id, "rec", False)

            if result is None:
                self.after(0, self._stop_pipe_timer, pipe_id)
                continue  # silence or stopped

            wav_bytes, t0, rms = result
            self.after(0, self._log,
                       f"[DEBUG] {tag} Chunk grabado - RMS: {rms:.6f} "
                       f"({len(wav_bytes)} bytes)")

            self._service.update_translation_model(self._trans_combo.get())

            # -- STT phase (runs in parallel with next pipe's recording) --
            self.after(0, self._set_pipe_semaphore, pipe_id, "stt", True)
            self.after(0, self._set_status, f"📝 {tag} Transcribiendo...")

            try:
                if use_local_whisper:
                    with self._stt_lock:
                        jp_text = self._service.transcribe(wav_bytes)
                else:
                    jp_text = self._service.transcribe(wav_bytes)
            except Exception as e:
                self.after(0, self._set_pipe_semaphore, pipe_id, "stt", False)
                self.after(0, self._stop_pipe_timer, pipe_id)
                self.after(0, self._log, f"[ERROR] {tag} STT: {type(e).__name__}: {e}")
                continue

            self.after(0, self._set_pipe_semaphore, pipe_id, "stt", False)
            self.after(0, self._log, f"[DEBUG] {tag} STT: '{jp_text}'")

            if not jp_text:
                self.after(0, self._stop_pipe_timer, pipe_id)
                continue

            # -- Translation phase (parallel: no lock during API call) --
            self.after(0, self._set_pipe_semaphore, pipe_id, "trans", True)
            self.after(0, self._set_status, f"🌐 {tag} Traduciendo...")

            try:
                tokens = []
                for token in self._service.translate_stream(jp_text):
                    tokens.append(token)
                translation = "".join(tokens)
            except Exception as e:
                self.after(0, self._set_pipe_semaphore, pipe_id, "trans", False)
                self.after(0, self._stop_pipe_timer, pipe_id)
                self.after(0, self._log,
                           f"[ERROR] {tag} Traducción: {type(e).__name__}: {e}")
                continue

            elapsed = time.monotonic() - t0

            # Display atomically (brief lock, only for textbox write)
            with self._display_lock:
                self.after(0, self._display_complete_subtitle,
                           jp_text, translation, tag, elapsed)

            self.after(0, self._set_pipe_semaphore, pipe_id, "trans", False)
            self.after(0, self._stop_pipe_timer, pipe_id)

        self.after(0, self._log, f"[DEBUG] {tag} worker finalizado")

    # -- Subtitle display helpers ------------------------------------------

    def _display_complete_subtitle(self, jp_text: str, translation: str,
                                    tag: str, elapsed: float):
        """Display a complete subtitle block atomically (no interleaving)."""
        timing = f"{elapsed:.1f}".replace(".", ",")
        block = (
            f"[{tag}] 🇯🇵  {jp_text}\n"
            f"      🇪🇸  {translation}\n"
            f"{'─' * 50} ({timing} s)\n\n"
        )
        self._subtitle_box.configure(state="normal")
        self._subtitle_box.insert("end", block)
        self._subtitle_box.see("end")
        self._subtitle_box.configure(state="disabled")

    def _log(self, message: str):
        """Append a message to the subtitle box. [DEBUG] messages only show if debug is on."""
        if message.startswith("[DEBUG]") and not self._debug_enabled:
            return
        self._subtitle_box.configure(state="normal")
        self._subtitle_box.insert("end", f"{message}\n")
        self._subtitle_box.see("end")
        self._subtitle_box.configure(state="disabled")

    def _set_status(self, text: str):
        self._status_label.configure(text=text)

    # -- Per-pipe semaphore & timer helpers --------------------------------

    def _set_pipe_semaphore(self, pipe_id: int, stage: str, active: bool):
        """Update a pipe's semaphore indicator. stage: 'rec', 'stt', 'trans'."""
        color = "#2ecc71" if active else "#555555"  # bright green / dark gray
        widget = self._pipe_sem_lights[pipe_id].get(stage)
        if widget:
            widget.configure(fg_color=color)

    def _start_pipe_timer(self, pipe_id: int, t0: float):
        """Start the live timer for a specific pipe."""
        self._pipe_timer_starts[pipe_id] = t0
        if self._timer_after_id is None:
            self._tick_timers()

    def _stop_pipe_timer(self, pipe_id: int):
        """Stop a pipe's timer and show final elapsed time."""
        t0 = self._pipe_timer_starts[pipe_id]
        if t0 is not None:
            elapsed = time.monotonic() - t0
            self._pipe_timer_labels[pipe_id].configure(
                text=f"⏱ {elapsed:.1f} s".replace(".", ",")
            )
        self._pipe_timer_starts[pipe_id] = None

    def _reset_pipe_timer(self, pipe_id: int):
        """Reset a pipe's timer display."""
        self._pipe_timer_starts[pipe_id] = None
        self._pipe_timer_labels[pipe_id].configure(text="⏱ --")

    def _tick_timers(self):
        """Periodic callback to update all active pipe timers."""
        any_active = False
        now = time.monotonic()
        for i in range(self.NUM_PIPES):
            t0 = self._pipe_timer_starts[i]
            if t0 is not None:
                elapsed = now - t0
                self._pipe_timer_labels[i].configure(
                    text=f"⏱ {elapsed:.1f} s".replace(".", ",")
                )
                any_active = True
        if any_active:
            self._timer_after_id = self.after(100, self._tick_timers)
        else:
            self._timer_after_id = None

    # -- Cleanup -----------------------------------------------------------

    def destroy(self):
        self._is_running = False
        self._stt_ready.set()  # unblock waiting pipes
        self._pipe_timer_starts = [None] * self.NUM_PIPES
        if self._timer_after_id is not None:
            self.after_cancel(self._timer_after_id)
            self._timer_after_id = None
        self._audio.stop()
        super().destroy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = SubtitleApp()
    app.mainloop()


if __name__ == "__main__":
    main()
