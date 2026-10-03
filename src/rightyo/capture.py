"""Explicit native Mac microphone PCM input with bounded, nonpersistent buffering."""

from __future__ import annotations

import math
import os
import queue
import signal
import subprocess
import tempfile
import threading
from pathlib import Path

PCM_SAMPLE_RATE = 16_000
PCM_SAMPLE_WIDTH = 2
PCM_CHANNELS = 1
PCM_CHUNK_BYTES = 6400


class CaptureError(ValueError):
    """Safe fixed-message failure; native output and paths are never surfaced."""


class MacMicrophoneCapture:
    """Start only on explicit user action; read PCM chunks, stop to discard them.

    The helper itself requests microphone permission. This wrapper does not infer
    readiness from process launch: read() may wait while the user decides. Queue
    pressure fails the session rather than silently dropping samples. Platform
    audio route changes fail closed and require a deliberate restart.
    Short native pipe reads are coalesced into 200 ms PCM chunks. The default
    queue retains at most 32 seconds (1,024,000 bytes), plus a partial chunk.
    """

    def __init__(self, helper: str | Path, *, queue_chunks: int = 160):
        if type(queue_chunks) is not int or not 1 <= queue_chunks <= 256:
            raise CaptureError("Invalid microphone queue bound")
        self.helper = Path(helper)
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=queue_chunks)
        self._process: subprocess.Popen[bytes] | None = None
        self._reader: threading.Thread | None = None
        self._stopped = threading.Event()
        self._lock = threading.Lock()
        self._failure: str | None = None
        self._directory: tempfile.TemporaryDirectory[str] | None = None

    def start(self) -> None:
        """Launch --capture once; only this method may trigger a permission prompt."""
        with self._lock:
            if self._process is not None or self._stopped.is_set():
                raise CaptureError("Microphone capture cannot be restarted")
            if not self.helper.is_file() or not os.access(self.helper, os.X_OK):
                raise CaptureError("An existing executable microphone helper is required")
            directory = tempfile.TemporaryDirectory(prefix="rightyo-microphone-")
            self._directory = directory
            failed = False
            try:
                self._process = subprocess.Popen(
                    [str(self.helper.resolve()), "--capture"],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    cwd=directory.name,
                    env={"PATH": os.defpath, "HOME": directory.name, "TMPDIR": directory.name},
                    start_new_session=True,
                    bufsize=0,
                )
            except OSError:
                directory.cleanup()
                self._directory = None
                failed = True
            if failed:
                raise CaptureError("Microphone helper could not start")
            self._reader = threading.Thread(
                target=self._receive, args=(self._process,), daemon=True
            )
            self._reader.start()

    def _receive(self, process: subprocess.Popen[bytes]) -> None:
        assert process.stdout is not None
        failure = "Microphone capture ended unexpectedly"
        carry = b""
        try:
            while not self._stopped.is_set():
                data = process.stdout.read(PCM_CHUNK_BYTES)
                if self._stopped.is_set():
                    break
                if not data:
                    returncode = process.wait(timeout=2)
                    failure = {
                        12: "Microphone permission was denied",
                        13: "No microphone input is available",
                        14: "Microphone audio conversion failed",
                        15: "Microphone capture exceeded its buffer bound",
                        16: "Microphone input changed; restart capture",
                    }.get(returncode, failure)
                    break
                data = carry + data
                while len(data) >= PCM_CHUNK_BYTES:
                    self._queue.put_nowait(data[:PCM_CHUNK_BYTES])
                    data = data[PCM_CHUNK_BYTES:]
                carry = data
        except queue.Full:
            failure = "Microphone capture exceeded its buffer bound"
        except (OSError, ValueError, subprocess.TimeoutExpired):
            failure = "Microphone input failed"
        # Stop/failure discards the incomplete chunk as well as the public queue.
        carry = b""
        data = b""
        if not self._stopped.is_set():
            self._failure = failure
            self.stop()

    def read(self, timeout: float = 0.25) -> bytes | None:
        """Return 200 ms mono PCM16 chunks; None means no complete chunk yet."""
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (float, int))
            or not math.isfinite(timeout)
            or not 0 <= timeout <= 5
        ):
            raise CaptureError("Invalid microphone read timeout")
        if self._failure is not None:
            raise CaptureError(self._failure)
        if self._process is None or self._stopped.is_set():
            raise CaptureError("Microphone capture is not running")
        try:
            chunk = self._queue.get(timeout=timeout)
        except queue.Empty:
            if self._failure is not None:
                raise CaptureError(self._failure) from None
            if self._stopped.is_set():
                raise CaptureError("Microphone capture is not running") from None
            return None
        if self._failure is not None:
            raise CaptureError(self._failure)
        if self._stopped.is_set():
            raise CaptureError("Microphone capture is not running")
        return chunk

    def stop(self) -> None:
        """Immediately discard pending PCM and reap the complete process group."""
        with self._lock:
            self._stopped.set()
            process = self._process
            self._process = None
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=1)
                if process.stdout is not None:
                    process.stdout.close()
            if self._directory is not None:
                self._directory.cleanup()
                self._directory = None
        reader = self._reader
        # Join outside the lifecycle lock: a failed reader can also own stop().
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=1)
        with self._lock:
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break


class StdinPcmCapture:
    """Raw PCM from a host-supplied binary stream (normally stdin), same format as the helper.

    Format: mono, 16,000 Hz, signed 16-bit little-endian, headerless, no other
    format is accepted or detected. Reads of any length are coalesced into 200 ms
    chunks; an odd byte is carried to the next read, and a final odd byte at EOF is
    discarded and counted. Unlike the microphone, a full queue does not fail the
    session: the chunk is dropped and counted as a gap, so buffering stays bounded
    (at most ``queue_chunks`` chunks plus one partial chunk). The first chunk queued
    after a drop carries the dropped byte count, so the consumer can mark the
    discontinuity instead of splicing the audio on either side. EOF ends the input.
    """

    def __init__(self, stream, *, queue_chunks: int = 160, report=None):
        if type(queue_chunks) is not int or not 1 <= queue_chunks <= 256:
            raise CaptureError("Invalid stdin queue bound")
        self._stream = stream
        self._report = report
        self._queue: queue.Queue[tuple[int, bytes]] = queue.Queue(maxsize=queue_chunks)
        self._reader: threading.Thread | None = None
        self._stopped = threading.Event()
        self._eof = threading.Event()
        self._failure: str | None = None
        self.dropped_bytes = 0
        self.gaps = 0
        self._unmarked_gap = 0
        self.discarded_tail_bytes = 0

    def start(self) -> None:
        if self._reader is not None or self._stopped.is_set():
            raise CaptureError("Stdin capture cannot be restarted")
        self._reader = threading.Thread(target=self._receive, daemon=True)
        self._reader.start()

    def _offer(self, chunk: bytes, dropping: bool) -> bool:
        """Queue without blocking the reader; returns whether a gap is in progress."""
        try:
            self._queue.put_nowait((self._unmarked_gap, chunk))
            self._unmarked_gap = 0
            return False
        except queue.Full:
            self.dropped_bytes += len(chunk)
            self._unmarked_gap += len(chunk)
            if not dropping:
                self.gaps += 1
                if self._report is not None:
                    self._report("stdin audio is arriving faster than it is processed; dropping")
            return True

    def _receive(self) -> None:
        read = getattr(self._stream, "read1", None) or self._stream.read
        carry = b""
        dropping = False
        try:
            while not self._stopped.is_set():
                data = read(PCM_CHUNK_BYTES)
                if self._stopped.is_set():
                    break
                if not data:
                    tail = len(carry) % PCM_SAMPLE_WIDTH
                    self.discarded_tail_bytes = tail
                    if len(carry) > tail:
                        self._offer(carry[: len(carry) - tail], dropping)
                    if tail and self._report is not None:
                        self._report("stdin audio ended mid-sample; discarded the final byte")
                    break
                data = carry + data
                while len(data) >= PCM_CHUNK_BYTES:
                    dropping = self._offer(data[:PCM_CHUNK_BYTES], dropping)
                    data = data[PCM_CHUNK_BYTES:]
                carry = data
        except (OSError, ValueError):
            self._failure = "Stdin audio input failed"
        carry = data = b""
        self._eof.set()

    def read(self, timeout: float = 0.25) -> tuple[int, bytes] | bytes | None:
        """Return ``(gap_bytes_dropped_before, pcm)``; None means none yet; b"" means EOF."""
        if self._stopped.is_set():
            raise CaptureError("Stdin capture is not running")
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            if self._eof.is_set() and self._queue.empty():
                if self._failure is not None:
                    raise CaptureError(self._failure) from None
                return b""
            return None

    def stop(self) -> None:
        """Discard pending PCM; a reader blocked on the stream exits at its next read."""
        self._stopped.set()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
