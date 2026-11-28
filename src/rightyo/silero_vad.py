"""Local, stateful Silero VAD for private voice exits. No automatic provisioning."""

from __future__ import annotations

import hashlib
import os
import socket
import stat
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

VERSION = "v5.1.2"
MODEL_URL = (
    f"https://raw.githubusercontent.com/snakers4/silero-vad/{VERSION}/"
    "src/silero_vad/data/silero_vad.onnx"
)
MODEL_SHA256 = "2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f"
MODEL_BYTES = 2327524
CHUNK_MS = 32
CHUNK_BYTES = 512 * 2  # 16 kHz mono signed little-endian PCM
THRESHOLD = 0.5
READY_TIMEOUT = 30
PROBE_TIMEOUT = 0.3

# Fixed startup protocol. Never relay paths, native errors, or worker output.
_ERRORS = {
    b"M": "model missing or unreadable",
    b"C": "model checksum mismatch",
    b"W": "worker unavailable",
}


class VadError(ValueError):
    pass


class SpeechVad(Protocol):
    chunk_ms: int

    @property
    def available(self) -> bool:
        """Fresh, bounded evidence the current judge is responsive."""
        ...

    def is_speech(self, pcm: bytes) -> bool: ...


@dataclass(frozen=True)
class VadConfig:
    python: Path
    model: Path

    @classmethod
    def from_dict(cls, value: object) -> VadConfig:
        if not isinstance(value, dict) or set(value) != {"python", "model"}:
            raise ValueError("invalid silero_vad section")
        paths = [value["python"], value["model"]]
        if not all(isinstance(path, str) and Path(path).is_absolute() for path in paths):
            raise ValueError("invalid silero_vad paths")
        return cls(*(Path(path) for path in paths))


def _worker(path: str) -> int:
    # stdin/out share a local socket. Keep the protocol isolated from native output.
    with os.fdopen(os.dup(1), "wb", buffering=0) as reply, open(os.devnull, "wb") as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
        try:
            # Nonblocking open also refuses FIFOs without hanging. Hash exactly the
            # bytes passed to ORT, so replacing a file after verification cannot race load.
            with open(os.open(path, os.O_NONBLOCK | os.O_RDONLY), "rb") as model:
                if not stat.S_ISREG(os.fstat(model.fileno()).st_mode):
                    raise OSError
                data = model.read(MODEL_BYTES + 1)
        except OSError:
            reply.write(b"M")
            return 1
        if hashlib.sha256(data).hexdigest() != MODEL_SHA256:
            reply.write(b"C")
            return 1
        try:
            import numpy as np
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.inter_op_num_threads = options.intra_op_num_threads = 1
            session = ort.InferenceSession(data, options, providers=["CPUExecutionProvider"])
            del data
            # v5's 16 kHz branch takes 64 preceding samples plus 512 new samples,
            # a scalar int64 sample rate, and the previous LSTM state.
            rate = np.array(16000, dtype=np.int64)
            state = np.zeros((2, 1, 128), dtype=np.float32)
            samples = np.zeros((1, 64 + 512), dtype=np.float32)
            pcm = bytearray(CHUNK_BYTES)
            reply.write(b"R")
            try:
                while opcode := sys.stdin.buffer.read(1):
                    if opcode == b"P":
                        reply.write(b"P")
                        continue
                    if opcode != b"S" or sys.stdin.buffer.readinto(pcm) != CHUNK_BYTES:
                        return 1
                    samples[0, 64:] = np.frombuffer(pcm, dtype="<i2")
                    samples[0, 64:] /= 32768.0
                    score, next_state = session.run(
                        ["output", "stateN"], {"input": samples, "sr": rate, "state": state}
                    )
                    probability = float(score.item())
                    if not 0 <= probability <= 1 or not np.isfinite(next_state).all():
                        raise ValueError
                    state[:] = next_state
                    next_state.fill(0)
                    score.fill(0)
                    samples[0, :64] = samples[0, -64:]
                    samples[0, 64:].fill(0)
                    pcm[:] = bytes(CHUNK_BYTES)
                    reply.write(b"1" if probability >= THRESHOLD else b"0")
                return 0
            finally:
                pcm[:] = bytes(CHUNK_BYTES)
                samples.fill(0)
                state.fill(0)
        except Exception:
            reply.write(b"W")
            return 1


class SileroVad:
    """One isolated interpreter and one ordered model stream per audio session."""

    chunk_ms = CHUNK_MS

    def __init__(self, settings: VadConfig, *, cancelled=lambda: False):
        self.cancelled = cancelled
        self._exchange = threading.Lock()
        try:
            self.channel, worker = socket.socketpair()
        except OSError:
            raise VadError(_ERRORS[b"W"]) from None
        try:
            with worker:
                self.process = subprocess.Popen(
                    [str(settings.python), "-I", "-B", __file__, "--worker", str(settings.model)],
                    stdin=worker,
                    stdout=worker,
                    stderr=subprocess.DEVNULL,
                    env={"PATH": os.defpath},
                )
        except (OSError, ValueError):
            self.channel.close()
            raise VadError(_ERRORS[b"W"]) from None
        try:
            ready = self._receive(READY_TIMEOUT)
            if ready != b"R":
                raise VadError(_ERRORS.get(ready, _ERRORS[b"W"]))
        except BaseException:
            self.close()
            raise

    def _receive(self, seconds: float) -> bytes:
        deadline = time.monotonic() + seconds
        self.channel.settimeout(0.1)
        while not self.cancelled() and time.monotonic() < deadline:
            try:
                return self.channel.recv(1)
            except TimeoutError:
                continue
            except OSError:
                break
        raise VadError(_ERRORS[b"W"])

    def is_speech(self, pcm: bytes) -> bool:
        if len(pcm) != CHUNK_BYTES:
            raise VadError("invalid VAD block")
        try:
            with self._exchange:
                self.channel.sendall(b"S")
                self.channel.sendall(pcm)
                response = self._receive(5)
            if response not in (b"0", b"1"):
                raise VadError(_ERRORS[b"W"])
            return response == b"1"
        except (VadError, OSError):
            self.close()
            raise VadError(_ERRORS[b"W"]) from None

    @property
    def available(self) -> bool:
        # Only the worker loop can acknowledge this ordered probe. It consumes
        # no samples and leaves both the native block clock and model state intact.
        if not self._exchange.acquire(timeout=PROBE_TIMEOUT):
            return False
        try:
            try:
                self.channel.settimeout(PROBE_TIMEOUT)
                self.channel.sendall(b"P")
                if self._receive(PROBE_TIMEOUT) == b"P" and self.process.poll() is None:
                    return True
            except (OSError, VadError):
                pass
            self.channel.close()
            return False
        finally:
            self._exchange.release()

    def close(self):
        self.channel.close()
        try:
            self.process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        raise SystemExit(_worker(sys.argv[2]))
    raise SystemExit(2)
