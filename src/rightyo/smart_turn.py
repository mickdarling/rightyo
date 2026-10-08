"""Optional Smart Turn v3 end-of-turn scoring in a private worker process (#117).

Smart Turn (pipecat-ai/smart-turn, BSD-2-Clause) reads the last 8 s of an utterance and
returns P(turn complete) from intonation and phrasing. RightyO itself stays free of
third-party dependencies: the worker runs under an explicitly configured interpreter
that has `numpy` and `onnxruntime`, with an explicitly configured local ONNX file. No
model is downloaded or provisioned here, and no audio is kept.

The worker is this file run as a script in isolated mode, so it imports nothing from
the repository. Its log-mel features reimplement the Whisper feature extractor that the
reference `inference.py` uses (16 kHz, 400-point periodic Hann window, hop 160, 80
Slaney mel bands, log10 with an 8-decade floor, `chunk_length=8`), and its input
handling follows that reference: keep the last 8 s and left-pad shorter audio.
"""

from __future__ import annotations

import base64
import json
import math
import os
import selectors
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 8 * SAMPLE_RATE
MAX_PCM_BYTES = 2 * WINDOW_SAMPLES
MAX_RESPONSE_BYTES = 4096
MAX_REQUEST_BYTES = 400000


class SmartTurnError(ValueError):
    """Sanitized failure: never embeds paths, audio, or worker output."""


def _features(np, samples):
    """Whisper log-mel features, shape (1, 80, 800), for exactly 8 s of float audio."""
    x = samples.astype(np.float64)
    x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
    n_fft, hop = 400, 160
    x = np.pad(x, n_fft // 2, mode="reflect")
    frames = 1 + (len(x) - n_fft) // hop
    index = np.arange(n_fft)[None, :] + hop * np.arange(frames)[:, None]
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
    power = np.abs(np.fft.rfft(x[index] * window, n=n_fft)) ** 2
    log_mel = np.log10(np.maximum(power @ _mel_filters(np), 1e-10)).T[:, :-1]
    log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
    return ((log_mel + 4.0) / 4.0)[None].astype(np.float32)


def _mel_filters(np, bands=80, n_fft=400):
    """Slaney-scale, Slaney-normalized triangular filters, shape (n_fft // 2 + 1, bands)."""
    f_sp, min_log_hz, min_log_mel = 200.0 / 3, 1000.0, 15.0
    logstep = np.log(6.4) / 27.0

    def to_mel(hz):
        hz = np.asarray(hz, dtype=np.float64)
        return np.where(
            hz >= min_log_hz,
            min_log_mel + np.log(np.maximum(hz, min_log_hz) / min_log_hz) / logstep,
            hz / f_sp,
        )

    def to_hz(mel):
        return np.where(
            mel >= min_log_mel, min_log_hz * np.exp(logstep * (mel - min_log_mel)), f_sp * mel
        )

    edges = to_hz(np.linspace(to_mel(0.0), to_mel(SAMPLE_RATE / 2), bands + 2))
    bins = np.linspace(0, SAMPLE_RATE // 2, n_fft // 2 + 1)
    slopes = edges[None, :] - bins[:, None]
    widths = np.diff(edges)
    down = -slopes[:, :-2] / widths[:-1]
    up = slopes[:, 2:] / widths[1:]
    filters = np.maximum(0.0, np.minimum(down, up))
    return filters * (2.0 / (edges[2 : bands + 2] - edges[:bands]))[None, :]


def _worker(model: str, threads: int) -> int:
    protocol = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    with open(os.devnull, "wb") as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
    try:
        import numpy as np
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = threads
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(model, sess_options=options)
        protocol.write('{"ok":true}\n')
        while True:
            line = sys.stdin.buffer.readline(MAX_REQUEST_BYTES)
            if not line or not line.endswith(b"\n"):
                break
            request = json.loads(line)
            if request.get("command") != "score":
                break
            pcm = base64.b64decode(request["pcm"], validate=True)
            if not pcm or len(pcm) > MAX_PCM_BYTES or len(pcm) % 2:
                raise SmartTurnError("Invalid audio")
            samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
            samples = np.concatenate(
                [np.zeros(WINDOW_SAMPLES - len(samples), dtype=np.float32), samples]
            )
            output = session.run(None, {"input_features": _features(np, samples)})
            probability = float(np.asarray(output[0]).reshape(-1)[0])
            protocol.write(json.dumps({"ok": True, "p": probability}) + "\n")
        return 0
    except Exception:
        protocol.write('{"ok":false}\n')
        return 1
    finally:
        protocol.close()


@dataclass(frozen=True)
class EndOfTurn:
    """The prototype's optional `end_of_turn` section; off unless present and enabled.

    `{"enabled": true, "python": "/abs/venv/bin/python", "model": "/abs/smart-turn.onnx"}`
    with optional `threshold` (0-1, default 0.5), `silence_ms` (20-1000 in 20 ms steps,
    default 200) and `threads` (1-8, default 4).
    """

    python: Path
    model: Path
    threshold: float = 0.5
    silence_ms: int = 200
    threads: int = 4

    @classmethod
    def from_dict(cls, value: Any) -> EndOfTurn | None:
        keys = {"enabled", "python", "model", "threshold", "silence_ms", "threads"}
        if not isinstance(value, dict) or set(value) - keys:
            raise ValueError("invalid end_of_turn section")
        enabled = value.get("enabled", True)
        if type(enabled) is not bool:
            raise ValueError("invalid end_of_turn section")
        if not enabled:
            return None
        paths = [value.get("python"), value.get("model")]
        if not all(isinstance(item, str) and Path(item).is_absolute() for item in paths):
            raise ValueError("invalid end_of_turn section")
        threshold = value.get("threshold", 0.5)
        silence = value.get("silence_ms", 200)
        threads = value.get("threads", 4)
        if (
            type(threshold) not in (int, float)
            or not math.isfinite(threshold)
            or not 0 < threshold <= 1
            or type(silence) is not int
            or not 20 <= silence <= 1000
            or silence % 20
            or type(threads) is not int
            or not 1 <= threads <= 8
        ):
            raise ValueError("invalid end_of_turn section")
        return cls(Path(paths[0]), Path(paths[1]), float(threshold), silence, threads)


class SmartTurn:
    """`score(pcm) -> P(complete)` through one persistent worker; single owner, synchronous."""

    model_id = "smart-turn-v3"

    def __init__(
        self,
        python: str | Path,
        model: str | Path,
        *,
        threads: int = 4,
        timeout_seconds: float = 5,
        cancelled: Callable[[], bool] | None = None,
    ):
        if not Path(python).is_file() or not Path(model).is_file():
            raise SmartTurnError("Explicit existing Smart Turn runtime and model are required")
        if type(threads) is not int or not 1 <= threads <= 8:
            raise SmartTurnError("Invalid Smart Turn thread count")
        self.timeout = timeout_seconds
        self.cancelled = cancelled if cancelled is not None else (lambda: False)
        self.process = subprocess.Popen(
            [
                str(python),
                "-I",
                str(Path(__file__).resolve()),
                "--worker",
                str(Path(model).resolve()),
                str(threads),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
            env={"PATH": os.defpath},
        )
        self.buffer = bytearray()
        try:
            # Loading the model can take longer than one score.
            self._receive(max(self.timeout, 30))
        except BaseException:
            self.close()
            raise

    def _receive(self, timeout: float) -> dict:
        deadline = time.monotonic() + timeout
        assert self.process.stdout is not None
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            while b"\n" not in self.buffer:
                if self.cancelled():
                    raise SmartTurnError("Audio session was stopped")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise SmartTurnError("Smart Turn timed out")
                if not selector.select(min(0.05, remaining)):
                    continue
                chunk = os.read(self.process.stdout.fileno(), 4096)
                if not chunk:
                    raise SmartTurnError("Smart Turn stopped unexpectedly")
                self.buffer.extend(chunk)
                if len(self.buffer) > MAX_RESPONSE_BYTES:
                    raise SmartTurnError("Smart Turn exceeded output limit")
        line, _, rest = self.buffer.partition(b"\n")
        self.buffer = bytearray(rest)
        try:
            result = json.loads(line)
            if not isinstance(result, dict) or result.get("ok") is not True:
                raise ValueError
        except (ValueError, RecursionError):
            raise SmartTurnError("Smart Turn failed") from None
        return result

    def score(self, pcm: bytes) -> float:
        """P(turn complete) for mono PCM16 at 16 kHz; only the last 8 s are sent."""
        if not isinstance(pcm, bytes) or not pcm or len(pcm) % 2:
            raise SmartTurnError("Invalid audio")
        pcm = pcm[-MAX_PCM_BYTES:]
        try:
            assert self.process.stdin is not None
            request = {"command": "score", "pcm": base64.b64encode(pcm).decode("ascii")}
            self.process.stdin.write((json.dumps(request) + "\n").encode())
            probability = self._receive(self.timeout).get("p")
        except SmartTurnError:
            self.close()
            raise
        except (BrokenPipeError, OSError, ValueError):
            self.close()
            raise SmartTurnError("Smart Turn stopped unexpectedly") from None
        if type(probability) is not float or not 0.0 <= probability <= 1.0:
            self.close()
            raise SmartTurnError("Smart Turn failed")
        return probability

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
        for stream in (self.process.stdin, self.process.stdout):
            if stream is not None:
                stream.close()


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker" and sys.argv[3].isdigit():
        raise SystemExit(_worker(sys.argv[2], int(sys.argv[3])))
    raise SystemExit(2)
