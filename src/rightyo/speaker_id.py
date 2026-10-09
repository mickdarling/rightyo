"""Speaker embeddings in a private worker process, for local enrollment (#137 step 3).

The model is WeSpeaker ResNet34-LM (ONNX, CC-BY-4.0), chosen by the voiceprint spike in
docs/voiceprint-evaluation.md. RightyO itself stays free of third-party dependencies: the
worker runs under an explicitly configured interpreter that has `numpy` and `onnxruntime`
(the Smart Turn interpreter is enough), with an explicitly configured local `.onnx` file
whose SHA-256 must match the documented download. Nothing is downloaded or provisioned
here, and the worker keeps no audio and writes nothing.

The worker is this file run as a script in isolated mode, so it imports nothing from the
repository. Its features reimplement the Kaldi-compatible fbank front end WeSpeaker uses
(25/10 ms frames, Hamming window, pre-emphasis 0.97, 80 log mel bands from 20 Hz to
Nyquist, int16-scaled samples, per-utterance mean removal), as in the spike harness.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import selectors
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SAMPLE_RATE = 16000
# One request carries at most 30 s of PCM16; a response is one short vector.
MAX_PCM_BYTES = 30 * SAMPLE_RATE * 2
MAX_REQUEST_BYTES = MAX_PCM_BYTES * 4 // 3 + 4096
MAX_RESPONSE_BYTES = 65536
MAX_EMBEDDING_DIM = 1024

# Model files RightyO knows how to drive, by SHA-256. An enrolled voiceprint records the
# model identity, and scores are only computed against voiceprints from the same model.
KNOWN_MODELS = {
    "7bb2f06e9df17cdf1ef14ee8a15ab08ed28e8d0ef5054ee135741560df2ec068": {
        "id": "Wespeaker/wespeaker-voxceleb-resnet34-LM",
        "revision": "f0c48c298fd835726c27956a5d617bad7115627e",
        "file": "voxceleb_resnet34_LM.onnx",
        "license": "CC-BY-4.0",
    }
}

DEFAULT_STORE = Path("~/Library/Application Support/RightyO/enrollment")


class SpeakerIdError(ValueError):
    """Sanitized failure: never embeds paths, audio, embeddings or worker output."""


def _kaldi_fbank(np, x, bins=80):
    """Kaldi-compatible fbank for float samples in [-1, 1], shape (frames, bins), with CMN."""
    x = x.astype(np.float64) * 32768.0
    frame_len, shift, n_fft = 400, 160, 512
    if len(x) < frame_len:
        x = np.pad(x, (0, frame_len - len(x)))
    count = 1 + (len(x) - frame_len) // shift
    index = np.arange(frame_len)[None, :] + shift * np.arange(count)[:, None]
    frames = x[index]
    frames = frames - frames.mean(axis=1, keepdims=True)
    frames[:, 1:] -= 0.97 * frames[:, :-1].copy()
    frames[:, 0] -= 0.97 * frames[:, 0]
    window = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(frame_len) / (frame_len - 1))
    power = np.abs(np.fft.rfft(frames * window, n=n_fft)) ** 2

    def mel(f):
        return 1127.0 * np.log(1.0 + np.asarray(f) / 700.0)

    low, high = mel(20.0), mel(SAMPLE_RATE / 2)
    centers = np.linspace(low, high, bins + 2)
    fft_mel = mel(np.arange(n_fft // 2) * SAMPLE_RATE / n_fft)
    banks = np.zeros((n_fft // 2 + 1, bins))
    for b in range(bins):
        left, center, right = centers[b : b + 3]
        up = (fft_mel - left) / (center - left)
        down = (right - fft_mel) / (right - center)
        banks[: n_fft // 2, b] = np.maximum(0.0, np.minimum(up, down))
    feats = np.log(np.maximum(power @ banks, np.finfo(np.float32).eps))
    return (feats - feats.mean(axis=0)).astype(np.float32)


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
        session = ort.InferenceSession(
            model, sess_options=options, providers=["CPUExecutionProvider"]
        )
        name = session.get_inputs()[0].name
        protocol.write('{"ok":true}\n')
        while True:
            line = sys.stdin.buffer.readline(MAX_REQUEST_BYTES)
            if not line or not line.endswith(b"\n"):
                break
            request = json.loads(line)
            if request.get("command") != "embed":
                break
            pcm = base64.b64decode(request["pcm"], validate=True)
            if not pcm or len(pcm) > MAX_PCM_BYTES or len(pcm) % 2:
                raise SpeakerIdError("Invalid audio")
            samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
            output = session.run(None, {name: _kaldi_fbank(np, samples)[None]})
            vector = np.asarray(output[0], dtype=np.float64).reshape(-1)
            protocol.write(json.dumps({"ok": True, "embedding": vector.tolist()}) + "\n")
        return 0
    except Exception:
        protocol.write('{"ok":false}\n')
        return 1
    finally:
        protocol.close()


def finite_number(value: Any) -> bool:
    """A JSON number that is finite as a float; a huge int is rejected, never raised on.

    `math.isfinite` raises OverflowError on an int too large for a float (a 400-digit
    literal parses fine), so ints are compared instead of converted.
    """
    if type(value) is float:
        return math.isfinite(value)
    return type(value) is int and -1e308 < value < 1e308


def _bounded(value: Any, low: float, high: float) -> bool:
    return finite_number(value) and low <= value <= high


@dataclass(frozen=True)
class SpeakerIdConfig:
    """The prototype's optional `speaker_id` section; off unless present and enabled.

    `{"python": "/abs/venv/bin/python", "model": "/abs/voxceleb_resnet34_LM.onnx"}` with
    optional `store` (absolute directory, default
    `~/Library/Application Support/RightyO/enrollment`), `bind_threshold` (default 0.60),
    `tentative_threshold` (default 0.45, below the bind threshold), `min_turn_seconds`
    (0.5-10, default 1.0), `threads` (1-8, default 4), `live` (default false),
    `bind_min_seconds` (0.5-120, default 3.0) and `enabled` (default true). `rightyo
    enroll` uses it; with `live` true, live sessions also run shadow identification
    (#137 step 4a, `rightyo.live_speaker_id`): scores are logged, nothing else changes.
    A label binds only once `bind_min_seconds` of its speech has been accumulated.

    `roles` (default false, requires `live`) applies the bindings (#137 step 4b): a label
    bound to an enrolled identifier carries the role the `speakers` section gives that
    identifier, and the session advertises `speakers: "enrolled"`. The optional
    `enrolled_follow_up_min_probability` (above 0, up to 1; requires `roles`) lowers the
    conversation-mode follow-up bar for an engaged owner or trusted speaker; it never
    raises it.
    """

    python: Path
    model: Path
    store: Path | None = None
    bind_threshold: float = 0.60
    tentative_threshold: float = 0.45
    min_turn_seconds: float = 1.0
    threads: int = 4
    live: bool = False
    bind_min_seconds: float = 3.0
    roles: bool = False
    enrolled_follow_up_min_probability: float | None = None

    @classmethod
    def from_dict(cls, value: Any) -> SpeakerIdConfig | None:
        keys = {
            "enabled",
            "python",
            "model",
            "store",
            "bind_threshold",
            "tentative_threshold",
            "min_turn_seconds",
            "threads",
            "live",
            "bind_min_seconds",
            "roles",
            "enrolled_follow_up_min_probability",
        }
        if not isinstance(value, dict) or set(value) - keys:
            raise ValueError("invalid speaker_id section")
        enabled = value.get("enabled", True)
        if type(enabled) is not bool:
            raise ValueError("invalid speaker_id section")
        if not enabled:
            return None
        paths = [value.get("python"), value.get("model")]
        store = value.get("store")
        if store is not None:
            paths.append(store)
        if not all(isinstance(item, str) and Path(item).is_absolute() for item in paths):
            raise ValueError("invalid speaker_id section")
        bind = value.get("bind_threshold", 0.60)
        tentative = value.get("tentative_threshold", 0.45)
        min_turn = value.get("min_turn_seconds", 1.0)
        threads = value.get("threads", 4)
        live = value.get("live", False)
        bind_min = value.get("bind_min_seconds", 3.0)
        roles = value.get("roles", False)
        follow_up = value.get("enrolled_follow_up_min_probability")
        if (
            type(roles) is not bool
            or (roles and not live)
            or (
                follow_up is not None
                and (not roles or not _bounded(follow_up, 0.0, 1.0) or follow_up == 0)
            )
        ):
            raise ValueError("invalid speaker_id section")
        if (
            not _bounded(bind, 0.0, 1.0)
            or not _bounded(tentative, 0.0, 1.0)
            or not 0 < tentative < bind
            or not _bounded(min_turn, 0.5, 10.0)
            or type(threads) is not int
            or not 1 <= threads <= 8
            or type(live) is not bool
            or not _bounded(bind_min, 0.5, 120.0)
        ):
            raise ValueError("invalid speaker_id section")
        return cls(
            Path(paths[0]),
            Path(paths[1]),
            None if store is None else Path(store),
            float(bind),
            float(tentative),
            float(min_turn),
            threads,
            live,
            float(bind_min),
            roles,
            None if follow_up is None else float(follow_up),
        )

    def band(self, score: float) -> str:
        """`bind` at or above the bind threshold, `tentative` above the lower one, else `below`."""
        if score >= self.bind_threshold:
            return "bind"
        if score >= self.tentative_threshold:
            return "tentative"
        return "below"


def model_identity(model: str | Path) -> dict[str, str]:
    """The documented identity of a known model file, by SHA-256; refuses any other file."""
    digest = hashlib.sha256()
    try:
        with open(model, "rb") as source:
            for block in iter(lambda: source.read(1 << 20), b""):
                digest.update(block)
    except OSError:
        raise SpeakerIdError("The speaker model file could not be read") from None
    known = KNOWN_MODELS.get(digest.hexdigest())
    if known is None:
        raise SpeakerIdError(
            "The speaker model file does not match the documented WeSpeaker ResNet34-LM download"
        )
    return {"id": known["id"], "revision": known["revision"], "sha256": digest.hexdigest()}


class SpeakerEmbedder:
    """`embed(pcm) -> list[float]` through one persistent worker; single owner, synchronous."""

    def __init__(
        self,
        python: str | Path,
        model: str | Path,
        *,
        threads: int = 4,
        timeout_seconds: float = 10,
        identity: dict[str, str] | None = None,
    ):
        if (
            not Path(python).is_file()
            or not os.access(python, os.X_OK)
            or not Path(model).is_file()
        ):
            raise SpeakerIdError("Explicit existing speaker model runtime and model are required")
        if type(threads) is not int or not 1 <= threads <= 8:
            raise SpeakerIdError("Invalid speaker model thread count")
        # Tests inject an identity for a stand-in worker; real use checks the file hash.
        self.model = model_identity(model) if identity is None else dict(identity)
        self.timeout = timeout_seconds
        try:
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
        except OSError:
            raise SpeakerIdError("The speaker model could not start") from None
        self.buffer = bytearray()
        try:
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
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise SpeakerIdError("The speaker model timed out")
                if not selector.select(min(0.05, remaining)):
                    continue
                chunk = os.read(self.process.stdout.fileno(), 65536)
                if not chunk:
                    raise SpeakerIdError("The speaker model stopped unexpectedly")
                self.buffer.extend(chunk)
                if len(self.buffer) > MAX_RESPONSE_BYTES:
                    raise SpeakerIdError("The speaker model exceeded its output limit")
        line, _, rest = self.buffer.partition(b"\n")
        self.buffer = bytearray(rest)
        try:
            result = json.loads(line)
            if not isinstance(result, dict) or result.get("ok") is not True:
                raise ValueError
        except (ValueError, RecursionError):
            raise SpeakerIdError("The speaker model failed") from None
        return result

    def embed(self, pcm: bytes) -> list[float]:
        """One embedding for mono PCM16 at 16 kHz, at most 30 s."""
        if not isinstance(pcm, bytes) or not pcm or len(pcm) % 2 or len(pcm) > MAX_PCM_BYTES:
            raise SpeakerIdError("Invalid audio")
        try:
            assert self.process.stdin is not None
            request = {"command": "embed", "pcm": base64.b64encode(pcm).decode("ascii")}
            data = memoryview((json.dumps(request) + "\n").encode())
            while data:
                written = self.process.stdin.write(data)
                if not written:
                    raise BrokenPipeError
                data = data[written:]
            vector = self._receive(self.timeout).get("embedding")
        except SpeakerIdError:
            self.close()
            raise
        except (BrokenPipeError, OSError, ValueError):
            self.close()
            raise SpeakerIdError("The speaker model stopped unexpectedly") from None
        if (
            not isinstance(vector, list)
            or not 1 <= len(vector) <= MAX_EMBEDDING_DIM
            or not all(finite_number(item) for item in vector)
        ):
            self.close()
            raise SpeakerIdError("The speaker model failed")
        return [float(item) for item in vector]

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
