"""
Transliterate JP→ES: Real-time Japanese audio to Spanish subtitle translator.

Captures system audio (loopback), transcribes Japanese speech using local
faster-whisper or OpenAI Whisper API, and translates to Spanish using
OpenAI GPT with streaming output.
"""

from __future__ import annotations

import io
import os
import queue
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
    """Captures system audio using loopback recording."""

    SAMPLE_RATE = 16000
    CHANNELS = 1
    CHUNK_SECONDS = 4

    def __init__(self):
        self._running = False
        self._thread: threading.Thread | None = None
        self._on_chunk = None
        self._loopback = None

    @staticmethod
    def list_loopback_devices() -> list[str]:
        if sc is None:
            return []
        try:
            speakers = sc.all_speakers()
            return [s.name for s in speakers]
        except Exception:
            return []

    def start(self, on_chunk, device_name: str | None = None):
        """Start capturing. *on_chunk* receives a WAV bytes buffer each cycle."""
        if self._running:
            return
        self._on_chunk = on_chunk
        self._running = True

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

        # Get loopback mic for the selected speaker
        self._loopback = sc.get_microphone(
            speaker.id, include_loopback=True
        )

        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def _record_loop(self):
        num_frames = self.SAMPLE_RATE * self.CHUNK_SECONDS
        if self._on_chunk:
            self._on_chunk(None, None, "[DEBUG] _record_loop iniciado")
        try:
            mic = self._loopback.recorder(
                samplerate=self.SAMPLE_RATE,
                channels=self.CHANNELS,
                blocksize=1024,
            )
        except Exception as e:
            if self._on_chunk:
                self._on_chunk(None, f"Error creando recorder: {type(e).__name__}: {e}")
            return

        if self._on_chunk:
            self._on_chunk(None, None, "[DEBUG] Recorder creado, grabando...")

        with mic:
            chunk_count = 0
            while self._running:
                try:
                    data = mic.record(numframes=num_frames)
                    chunk_count += 1
                    audio_float = data[:, 0] if data.ndim > 1 else data
                    wav_bytes = self._float_to_wav(audio_float)

                    rms = np.sqrt(np.mean(audio_float ** 2))
                    if self._on_chunk:
                        self._on_chunk(None, None, f"[DEBUG] Chunk #{chunk_count} - RMS: {rms:.6f}")
                    if rms < 0.001:
                        continue

                    if self._on_chunk:
                        self._on_chunk(wav_bytes, None)
                except Exception as e:
                    if self._on_chunk and self._running:
                        self._on_chunk(None, f"Error en record: {type(e).__name__}: {e}")
                    break

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

        # Pipeline queues: items are (payload, capture_time) or None (poison pill)
        self._stt_queue: queue.Queue[tuple[bytes, float] | None] = queue.Queue(maxsize=5)
        self._translate_queue: queue.Queue[tuple[str, float] | None] = queue.Queue(maxsize=5)
        self._stt_thread: threading.Thread | None = None
        self._translate_thread: threading.Thread | None = None

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

        # -- Status bar --
        status_frame = ctk.CTkFrame(self, fg_color="transparent")
        status_frame.grid(row=5, column=0, padx=16, pady=(0, 8), sticky="ew")
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

            # Clear queues
            for q in (self._stt_queue, self._translate_queue):
                while not q.empty():
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        break

            self._is_running = True

            # Launch pipeline workers (STT worker loads Whisper model before processing)
            self._stt_thread = threading.Thread(target=self._stt_worker, daemon=True)
            self._stt_thread.start()
            self._translate_thread = threading.Thread(target=self._translate_worker, daemon=True)
            self._translate_thread.start()

            self._log("[DEBUG] Iniciando captura de audio...")
            try:
                self._audio.start(self._on_audio_chunk, device_name)
            except Exception as e:
                self._is_running = False
                self._stt_queue.put(None)
                self._set_status(f"⚠ Error de audio: {e}")
                self._log(f"[DEBUG] Error en audio.start: {type(e).__name__}: {e}")
                self._stt_combo.configure(state="readonly")
                self._whisper_combo.configure(state="readonly")
                return

            self._start_btn.configure(
                text="⏹  Detener", fg_color="#c0392b", hover_color="#962d22"
            )
            self._log("[DEBUG] Captura de audio iniciada")
        except Exception as e:
            self._set_status(f"⚠ Error inesperado: {e}")
            self._log(f"[DEBUG] Excepción no controlada: {type(e).__name__}: {e}")

    def _stop_capture(self):
        self._audio.stop()
        self._is_running = False
        # Send poison pills to stop workers
        self._stt_queue.put(None)
        self._translate_queue.put(None)
        self._start_btn.configure(
            text="▶  Iniciar", fg_color="#2d8a4e", hover_color="#236b3e"
        )
        # Re-enable selectors
        self._stt_combo.configure(state="readonly")
        self._whisper_combo.configure(state="readonly")
        self._set_status("Estado: Detenido")

    def _on_audio_chunk(self, wav_bytes: bytes | None, error: str | None, debug_msg: str | None = None):
        """Called from the audio thread — enqueues audio for the STT worker."""
        if debug_msg:
            self.after(0, self._log, debug_msg)
            if wav_bytes is None and error is None:
                return

        if error:
            self.after(0, self._set_status, f"⚠ Audio error: {error}")
            self.after(0, self._log, f"[ERROR] {error}")
            self.after(0, self._stop_capture)
            return

        # Enqueue with capture timestamp; drop if queue is full
        try:
            self._stt_queue.put_nowait((wav_bytes, time.monotonic()))
        except queue.Full:
            self.after(0, self._log, "[DEBUG] STT queue llena, chunk descartado")

    # -- Pipeline workers --------------------------------------------------

    def _stt_worker(self):
        """Thread: loads STT backend, then transcribes audio chunks."""
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

        while self._is_running:
            try:
                item = self._stt_queue.get(timeout=1)
            except queue.Empty:
                continue
            if item is None:
                break

            wav_bytes, t0 = item

            self._service.update_translation_model(self._trans_combo.get())

            self.after(0, self._set_status, "📝 Transcribiendo...")
            self.after(0, self._log,
                       f"[DEBUG] Transcribiendo audio ({len(wav_bytes)} bytes)...")

            try:
                jp_text = self._service.transcribe(wav_bytes)
            except Exception as e:
                self.after(0, self._log, f"[ERROR] STT: {type(e).__name__}: {e}")
                continue

            self.after(0, self._log, f"[DEBUG] STT resultado: '{jp_text}'")

            if not jp_text:
                self.after(0, self._set_status, "🎧 Capturando audio del sistema...")
                continue

            # Enqueue for translation, carrying the original capture timestamp
            try:
                self._translate_queue.put_nowait((jp_text, t0))
            except queue.Full:
                self.after(0, self._log, "[DEBUG] Cola traducción llena, texto descartado")

        self.after(0, self._log, "[DEBUG] STT worker finalizado")

    def _translate_worker(self):
        """Thread: streams GPT translation to the subtitle box."""
        while self._is_running:
            try:
                item = self._translate_queue.get(timeout=1)
            except queue.Empty:
                continue
            if item is None:
                break

            jp_text, t0 = item

            if not self._service:
                continue

            self.after(0, self._set_status, "🌐 Traduciendo...")
            self.after(0, self._begin_subtitle, jp_text)

            try:
                for token in self._service.translate_stream(jp_text):
                    self.after(0, self._stream_token, token)
            except Exception as e:
                self.after(0, self._log, f"[ERROR] Traducción: {type(e).__name__}: {e}")
                self.after(0, self._end_subtitle, None)
                continue

            elapsed = time.monotonic() - t0
            self.after(0, self._end_subtitle, elapsed)
            self.after(0, self._set_status, "🎧 Capturando audio del sistema...")

        self.after(0, self._log, "[DEBUG] Translate worker finalizado")

    # -- Subtitle display helpers ------------------------------------------

    def _begin_subtitle(self, jp_text: str):
        """Insert the JP line and start the ES line for streaming."""
        self._subtitle_box.configure(state="normal")
        self._subtitle_box.insert("end", f"🇯🇵  {jp_text}\n🇪🇸  ")
        self._subtitle_box.see("end")
        self._subtitle_box.configure(state="disabled")

    def _stream_token(self, token: str):
        """Append a single streamed token to the current ES line."""
        self._subtitle_box.configure(state="normal")
        self._subtitle_box.insert("end", token)
        self._subtitle_box.see("end")
        self._subtitle_box.configure(state="disabled")

    def _end_subtitle(self, elapsed: float | None):
        """Close the current subtitle block with separator and optional timing."""
        self._subtitle_box.configure(state="normal")
        if elapsed is not None:
            timing = f"{elapsed:.1f}".replace(".", ",")
            self._subtitle_box.insert("end", f"\n{'─' * 50} ({timing} s)\n\n")
        else:
            self._subtitle_box.insert("end", f"\n{'─' * 60}\n\n")
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

    # -- Cleanup -----------------------------------------------------------

    def destroy(self):
        self._is_running = False
        self._audio.stop()
        self._stt_queue.put(None)
        self._translate_queue.put(None)
        super().destroy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = SubtitleApp()
    app.mainloop()


if __name__ == "__main__":
    main()
