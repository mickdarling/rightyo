"""Explicit loopback speech lab; startup never enables microphone or hosted inference."""

from __future__ import annotations

import hmac
import json
import math
import queue
import secrets
import threading
import time
import uuid
import wave
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from rightyo.capture import CaptureError, MacMicrophoneCapture
from rightyo.contracts import ContractError, Turn
from rightyo.credentials import CredentialError
from rightyo.live_audio import LiveAudioError, LiveConfig, LiveProcessor
from rightyo.memory import TranscriptMemory
from rightyo.pipeline import ReplayRunner
from rightyo.providers import JevProvider, MockProvider, ProviderError

SESSION_SECONDS = 900
BROWSER_LEASE_SECONDS = 15


class PrototypeError(ValueError):
    """A safe configuration or control error."""


def _reject_constant(_value):
    raise ValueError("Invalid JSON constant")


@dataclass(frozen=True)
class PrototypeConfig:
    whisper_executable: Path
    whisper_model: Path
    diarization_library: Path
    diarization_model: Path
    microphone_helper: Path
    demo_audio: Path | None = None

    @classmethod
    def load(cls, path: Path) -> PrototypeConfig:
        try:
            with path.open("rb") as source:
                content = source.read(65537)
            if len(content) > 65536:
                raise ValueError
            raw = json.loads(content, parse_constant=_reject_constant)
            required = {
                "whisper_executable",
                "whisper_model",
                "diarization_library",
                "diarization_model",
                "microphone_helper",
            }
            if not isinstance(raw, dict) or not required <= raw.keys():
                raise ValueError
            if raw.keys() - required - {"demo_audio"}:
                raise ValueError
            values = {}
            for name, value in raw.items():
                if not isinstance(value, str) or not value or not Path(value).is_absolute():
                    raise ValueError
                values[name] = Path(value)
            config = cls(**values)
            if not all(value.is_file() for value in values.values()):
                raise ValueError
            return config
        except (OSError, ValueError, TypeError, UnicodeError, RecursionError):
            raise PrototypeError(
                "Prototype requires an existing local asset configuration"
            ) from None


class PrototypeController:
    def __init__(
        self,
        config: PrototypeConfig,
        *,
        processor_factory=LiveProcessor,
        capture_factory=MacMicrophoneCapture,
        provider_factory=JevProvider,
    ):
        self.config = config
        self.processor_factory = processor_factory
        self.capture_factory = capture_factory
        self.provider_factory = provider_factory
        self._lock = threading.RLock()
        self._generation = 0
        self._phase = "idle"
        self._mode = "microphone"
        self._error = None
        self._decision_status = "off"
        self._memory = TranscriptMemory()
        self._runner: ReplayRunner | None = None
        self._capture = None
        self._processor = None
        self._audio_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._decision_cancel = threading.Event()
        self._decision_queue: queue.Queue[Turn] = queue.Queue(maxsize=32)
        self._decisions: dict[str, dict[str, Any]] = {}
        self._pending = 0
        self._requests = 0
        self._request_limit = 20
        self._received_ms = 0
        self._received_bytes = 0
        self._audio_started: float | None = None
        self._completed_at: float | None = None
        self._started = 0.0
        self._last_browser = time.monotonic()
        self._closed = threading.Event()
        self._timer = threading.Thread(target=self._tick, daemon=True)
        self._timer.start()

    def _now_ms(self) -> int:
        if self._completed_at is not None:
            return self._received_ms + int((time.monotonic() - self._completed_at) * 1000)
        wall_ms = (
            int((time.monotonic() - self._audio_started) * 1000)
            if self._mode == "microphone" and self._audio_started is not None
            else 0
        )
        return max(self._received_ms, wall_ms)

    def _tick(self) -> None:
        while not self._closed.wait(0.5):
            with self._lock:
                active = self._phase in {"starting", "listening", "replaying", "finishing"}
                expired = active and (
                    time.monotonic() - self._last_browser > BROWSER_LEASE_SECONDS
                    or time.monotonic() - self._started > SESSION_SECONDS
                )
                if self._runner is not None:
                    self._runner.expire(self._now_ms())
                    self._prune_pending()
            if expired:
                self.stop()

    def start(self, options: dict[str, Any]) -> None:
        if set(options) - {"mode", "use_jev", "retention_seconds", "confidence", "max_requests"}:
            raise PrototypeError("Unrecognized prototype setting")
        mode = options.get("mode", "microphone")
        hosted = options.get("use_jev", False)
        retention = options.get("retention_seconds", 300)
        confidence = options.get("confidence", 0.7)
        budget = options.get("max_requests", 20)
        if (
            not isinstance(mode, str)
            or mode not in {"microphone", "demo"}
            or type(hosted) is not bool
            or type(retention) is not int
            or not 60 <= retention <= 600
            or type(budget) is not int
            or not 1 <= budget <= 100
            or type(confidence) not in {int, float}
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
        ):
            raise PrototypeError("Invalid prototype setting")
        if mode == "demo" and self.config.demo_audio is None:
            raise PrototypeError("No generated audio demo is configured")
        with self._lock:
            if self._phase in {"starting", "listening", "replaying", "finishing", "stopping"}:
                raise PrototypeError("Stop the active session before starting another")
            if self._audio_thread is not None and self._audio_thread.is_alive():
                raise PrototypeError("The previous audio runtime is still stopping")
            stop = threading.Event()
            decision_cancel = threading.Event()

            def cancelled():
                return stop.is_set() or decision_cancel.is_set()

            memory = TranscriptMemory(retention_ms=retention * 1000)
            try:
                provider = (
                    self.provider_factory(
                        allow_hosted=True,
                        max_requests=budget,
                        timeout_seconds=10,
                        min_confidence=confidence,
                        cancelled=cancelled,
                    )
                    if hosted
                    else MockProvider()
                )
            except (ProviderError, CredentialError):
                raise PrototypeError("Hosted decisions could not be initialized") from None
            runner = ReplayRunner(provider, memory=memory, cancelled=cancelled)
            session = "prototype-" + uuid.uuid4().hex
            runner.restart(session)
            self._generation += 1
            generation = self._generation
            self._stop = stop
            self._decision_cancel = decision_cancel
            self._memory = memory
            self._runner = runner
            self._decision_queue = work = queue.Queue(maxsize=32)
            self._decisions = {}
            self._pending = self._requests = self._received_ms = 0
            self._received_bytes = 0
            self._audio_started = None
            self._completed_at = None
            self._request_limit = budget
            self._mode = mode
            self._error = None
            self._decision_status = "ready" if hosted else "off"
            self._phase = "starting"
            self._started = self._last_browser = time.monotonic()
            threading.Thread(
                target=self._decide,
                args=(generation, stop, work, runner, hosted),
                daemon=True,
            ).start()
            self._audio_thread = threading.Thread(
                target=self._audio,
                args=(generation, stop, session, work, memory, mode),
                daemon=True,
            )
            self._audio_thread.start()

    def _accept(self, generation: int, work: queue.Queue, memory: TranscriptMemory, turn: Turn):
        with self._lock:
            if generation != self._generation or self._stop.is_set():
                return
            memory.append(turn)
            if self._decision_cancel.is_set():
                return
            try:
                work.put_nowait(turn)
            except queue.Full:
                self._decision_cancel.set()
                if self._decision_status != "off":
                    self._decision_status = "unavailable"
                self._discard_pending()
                return
            self._pending += 1

    def _audio(self, generation, stop, session, work, memory, mode):
        capture = processor = None
        try:
            processor = self.processor_factory(
                LiveConfig(
                    session_id=session,
                    whisper_executable=self.config.whisper_executable,
                    whisper_model=self.config.whisper_model,
                    diarization_library=self.config.diarization_library,
                    diarization_model=self.config.diarization_model,
                    provenance="live-microphone" if mode == "microphone" else "causal-replay",
                    cancelled=stop.is_set,
                ),
                lambda turn: self._accept(generation, work, memory, turn),
            )
            with self._lock:
                if generation != self._generation or stop.is_set():
                    return
                self._processor = processor
            if mode == "microphone":
                capture = self.capture_factory(self.config.microphone_helper)
                with self._lock:
                    if generation != self._generation or stop.is_set():
                        return
                    self._capture = capture
                    capture.start()
                while not stop.is_set():
                    pcm = capture.read(timeout=0.25)
                    if pcm:
                        self._feed(generation, processor, pcm, mode)
            else:
                with wave.open(str(self.config.demo_audio), "rb") as audio:
                    if (
                        audio.getnchannels() != 1
                        or audio.getframerate() != 16000
                        or audio.getsampwidth() != 2
                        or audio.getcomptype() != "NONE"
                        or not 0 < audio.getnframes() <= 180 * 16000
                    ):
                        raise PrototypeError("Demo requires bounded mono PCM16 16 kHz audio")
                    while not stop.is_set():
                        pcm = audio.readframes(3200)
                        if not pcm:
                            break
                        self._feed(generation, processor, pcm, mode)
                if not stop.is_set():
                    processor.finish()
                    with self._lock:
                        if generation == self._generation:
                            self._phase = "finishing" if self._pending else "complete"
                            self._completed_at = time.monotonic()
        except (CaptureError, LiveAudioError, PrototypeError, ContractError, OSError, wave.Error):
            with self._lock:
                if generation == self._generation and not stop.is_set():
                    self._phase = "error"
                    self._error = (
                        "Audio stopped. Check microphone permission and local model setup."
                    )
                    stop.set()
                    self._completed_at = time.monotonic()
                    self._discard_pending()
                    if self._runner is not None:
                        self._runner.clear(clear_memory=False)
        except Exception:
            with self._lock:
                if generation == self._generation and not stop.is_set():
                    self._phase = "error"
                    self._error = "Audio processing failed; the session is incomplete."
                    stop.set()
                    self._completed_at = time.monotonic()
                    self._discard_pending()
                    if self._runner is not None:
                        self._runner.clear(clear_memory=False)
        finally:
            if capture is not None:
                capture.stop()
            if processor is not None:
                processor.close()
            with self._lock:
                if generation == self._generation:
                    self._capture = self._processor = None
                elif generation + 1 == self._generation and self._phase == "stopping":
                    self._phase = "idle"

    def _feed(self, generation, processor, pcm, mode):
        with self._lock:
            if generation != self._generation or self._stop.is_set():
                return
            self._received_bytes += len(pcm)
            self._received_ms = self._received_bytes // 32
            if self._audio_started is None:
                self._audio_started = time.monotonic()
            self._phase = "listening" if mode == "microphone" else "replaying"
        processor.push_pcm16(pcm)

    def _decide(self, generation, stop, work, runner, hosted):
        enabled = hosted
        while not stop.is_set():
            try:
                with self._lock:
                    turn = work.get_nowait()
            except queue.Empty:
                with self._lock:
                    if generation != self._generation or self._phase == "complete":
                        return
                stop.wait(0.05)
                continue
            try:
                event = None
                with self._lock:
                    if generation != self._generation or stop.is_set():
                        continue
                    if self._decision_cancel.is_set():
                        enabled = False
                    runner.expire(self._now_ms())
                    if turn.utterance_id not in self._memory.retained_ids:
                        continue
                if not enabled:
                    if not hosted:
                        runner.process(turn)
                    continue
                if runner.provider.requests >= self._request_limit:
                    with self._lock:
                        if generation == self._generation:
                            self._decision_status = "budget-exhausted"
                    enabled = False
                    continue
                event = runner.process(turn)
                with self._lock:
                    if (
                        generation == self._generation
                        and not stop.is_set()
                        and not self._decision_cancel.is_set()
                        and event is not None
                    ):
                        self._requests = runner.provider.requests
                        self._decisions[turn.utterance_id] = {
                            **event.public_dict(),
                            "recipient_speaker_id": event.decision.recipient_speaker_id,
                        }
            except Exception:
                enabled = False
                with self._lock:
                    if generation == self._generation and not stop.is_set():
                        self._decision_status = "unavailable"
                        self._requests = getattr(runner.provider, "requests", 0)
            finally:
                event = None
                work.task_done()
                with self._lock:
                    if generation == self._generation:
                        self._pending = max(0, self._pending - 1)
                        if self._phase == "finishing" and self._pending == 0:
                            self._phase = "complete"
                del turn
        runner.clear(clear_memory=False)

    def _discard_pending(self):
        while True:
            try:
                self._decision_queue.get_nowait()
            except queue.Empty:
                break
            self._decision_queue.task_done()
            self._pending = max(0, self._pending - 1)

    def _prune_pending(self):
        """Release expired queued plaintext, including while hosted inference is slow."""
        retained = self._memory.retained_ids
        keep = []
        while True:
            try:
                turn = self._decision_queue.get_nowait()
            except queue.Empty:
                break
            self._decision_queue.task_done()
            if turn.utterance_id in retained:
                keep.append(turn)
            else:
                self._pending = max(0, self._pending - 1)
        for turn in keep:
            self._decision_queue.put_nowait(turn)

    def snapshot(self, *, heartbeat=True) -> dict[str, Any]:
        with self._lock:
            if heartbeat:
                self._last_browser = time.monotonic()
            now = self._now_ms()
            if self._runner is not None:
                self._runner.expire(now)
                self._prune_pending()
            snapshot = self._memory.snapshot(now)
            retained = {t["utterance_id"] for t in snapshot["turns"]}
            self._decisions = {k: v for k, v in self._decisions.items() if k in retained}
            return {
                **snapshot,
                "phase": self._phase,
                "mode": self._mode,
                "error": self._error,
                "decision_status": self._decision_status,
                "decisions": dict(self._decisions),
                "pending_decisions": self._pending,
                "jev_requests": self._requests,
                "jev_request_limit": self._request_limit,
                "demo_available": self.config.demo_audio is not None,
                "models": {
                    "diarization": "Nemotron 3 (configured GGUF)",
                    "asr": "Whisper (configured model)",
                    "decision": "Jev 1.13.0 (opt-in)",
                },
                "session_limit_seconds": SESSION_SECONDS,
            }

    def stop(self) -> None:
        with self._lock:
            if self._phase == "stopping":
                return
            self._generation += 1
            generation = self._generation
            self._stop.set()
            self._decision_cancel.set()
            capture, processor, runner = self._capture, self._processor, self._runner
            audio_thread = self._audio_thread
            self._capture = self._processor = self._runner = None
            self._phase = "stopping"
            self._error = None
            self._memory.clear()
            self._decisions.clear()
            self._pending = self._received_ms = 0
            self._received_bytes = 0
            self._audio_started = None
            self._completed_at = None
            self._decision_status = "off"
            while not self._decision_queue.empty():
                try:
                    self._decision_queue.get_nowait()
                    self._decision_queue.task_done()
                except queue.Empty:
                    break
        try:
            if runner is not None:
                runner.clear()
            if capture is not None:
                capture.stop()
            if processor is not None:
                processor.close()
            if audio_thread is not None and audio_thread is not threading.current_thread():
                audio_thread.join(timeout=5)
        finally:
            with self._lock:
                if generation == self._generation and (
                    audio_thread is None or not audio_thread.is_alive()
                ):
                    self._phase = "idle"

    def close(self):
        self._closed.set()
        self.stop()


class PrototypeServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, controller: PrototypeController, port: int = 0):
        super().__init__(("127.0.0.1", port), PrototypeHandler)
        self.controller = controller
        self.token = secrets.token_urlsafe(32)
        self.origin = f"http://127.0.0.1:{self.server_port}"

    @property
    def launch_url(self):
        return self.origin + "/#" + self.token


class PrototypeHandler(BaseHTTPRequestHandler):
    server: PrototypeServer

    def setup(self):
        self.request.settimeout(5)
        super().setup()

    def log_message(self, *_args):
        pass

    def _response(self, status: int, payload: bytes, kind="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
        )
        self.end_headers()
        try:
            self.wfile.write(payload)
        except OSError:
            pass

    def _json(self, status, value):
        self._response(status, json.dumps(value, allow_nan=False).encode())

    def _allowed(self, *, authenticated: bool) -> bool:
        host = self.server.origin.removeprefix("http://")
        if self.headers.get("Host") != host:
            return False
        origin = self.headers.get("Origin")
        if origin is not None and origin != self.server.origin:
            return False
        return not authenticated or hmac.compare_digest(
            self.headers.get("Authorization", "").encode("utf-8"),
            ("Bearer " + self.server.token).encode("utf-8"),
        )

    def do_GET(self):
        if not self._allowed(authenticated=self.path.startswith("/api/")):
            self._json(403, {"error": "Local session authorization required"})
            return
        if self.path == "/api/state":
            self._json(200, self.server.controller.snapshot())
            return
        assets = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/app.js": ("app.js", "text/javascript; charset=utf-8"),
            "/style.css": ("style.css", "text/css; charset=utf-8"),
        }
        if self.path not in assets:
            self._json(404, {"error": "Not found"})
            return
        name, kind = assets[self.path]
        self._response(200, (Path(__file__).parent / "web" / name).read_bytes(), kind)

    def do_POST(self):
        if not self._allowed(authenticated=True):
            self._json(403, {"error": "Local session authorization required"})
            return
        try:
            length = int(self.headers.get("Content-Length", "-1"))
            if not 0 <= length <= 4096 or self.headers.get("Transfer-Encoding") is not None:
                raise ValueError
            raw = json.loads(self.rfile.read(length), parse_constant=_reject_constant)
            if not isinstance(raw, dict):
                raise ValueError
        except (ValueError, UnicodeError, RecursionError, OSError):
            self._json(400, {"error": "Invalid control request"})
            return
        try:
            if self.path == "/api/start":
                self.server.controller.start(raw)
            elif self.path == "/api/stop" and not raw:
                self.server.controller.stop()
            else:
                self._json(404, {"error": "Not found"})
                return
        except PrototypeError as error:
            self._json(400, {"error": str(error)})
            return
        self._json(200, {"ok": True})


def serve(config_path: Path, port: int = 8765) -> None:
    if type(port) is not int or not 0 <= port <= 65535:
        raise PrototypeError("Invalid local port")
    controller = PrototypeController(PrototypeConfig.load(config_path))
    server = None
    try:
        server = PrototypeServer(controller, port)
        print("Open the local speech lab (idle): " + server.launch_url, flush=True)
        server.serve_forever(poll_interval=0.25)
    finally:
        controller.close()
        if server is not None:
            server.server_close()
