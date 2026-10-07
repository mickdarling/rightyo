"""Explicit native Mac microphone PCM input with bounded, nonpersistent buffering."""

from __future__ import annotations

import math
import os
import queue
import select
import signal
import subprocess
import tempfile
import threading
from pathlib import Path

PCM_SAMPLE_RATE = 16_000
PCM_SAMPLE_WIDTH = 2
PCM_CHANNELS = 1
PCM_CHUNK_BYTES = 6400
# How often a descriptor-backed stdin reader rechecks stop while no input arrives.
STDIN_POLL_SECONDS = 0.1


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
    discarded and counted. The queue holds at most ``queue_chunks`` chunks plus one
    partial chunk. If it is full, the input overruns: the unqueued audio is dropped
    and counted, reading stops, and the queued audio is followed by end of input with
    ``overrun`` set. No audio after a drop is ever delivered, so nothing is spliced
    across a gap. Otherwise EOF ends the input.

    A stream with a file descriptor (stdin) is read with ``select`` and ``os.read`` on
    that descriptor by a non-daemon thread that rechecks ``stop`` every
    ``STDIN_POLL_SECONDS`` and is joined by ``stop()``. It never enters the stream's
    ``BufferedReader``, so no thread holds the stdin buffer lock when the interpreter
    finalizes (#74, #78: ``could not acquire lock for <stdin>`` aborted with SIGABRT,
    exit 134, when a session ended while the host still held stdin open). Only a
    stream without a usable descriptor (a test double) is read through its own
    ``read1``/``read`` on a daemon thread, as before.
    """

    def __init__(self, stream, *, queue_chunks: int = 160, report=None):
        if type(queue_chunks) is not int or not 1 <= queue_chunks <= 256:
            raise CaptureError("Invalid stdin queue bound")
        self._stream = stream
        self._report = report
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=queue_chunks)
        self._reader: threading.Thread | None = None
        self._stopped = threading.Event()
        self._eof = threading.Event()
        self._failure: str | None = None
        self.overrun = False
        self.dropped_bytes = 0
        self.discarded_tail_bytes = 0

    def start(self) -> None:
        if self._reader is not None or self._stopped.is_set():
            raise CaptureError("Stdin capture cannot be restarted")
        descriptor = _descriptor(self._stream)
        if descriptor is None:
            read = getattr(self._stream, "read1", None) or self._stream.read
            self._reader = threading.Thread(target=self._receive, args=(read,), daemon=True)
        else:
            self._reader = threading.Thread(
                target=self._receive, args=(self._polled_reader(descriptor),), daemon=False
            )
        self._reader.start()

    def _polled_reader(self, descriptor: int):
        """A read that waits in short polls, returning None once stop is requested."""

        def read(limit: int) -> bytes | None:
            while not self._stopped.is_set():
                ready, _, _ = select.select([descriptor], [], [], STDIN_POLL_SECONDS)
                if ready:
                    try:
                        return os.read(descriptor, limit)
                    except BlockingIOError:
                        continue  # A host-set O_NONBLOCK descriptor raced empty; poll again.
            return None

        return read

    def _offer(self, data: bytes) -> bool:
        """Queue one chunk without blocking; on a full queue drop ``data`` and overrun."""
        try:
            self._queue.put_nowait(data[:PCM_CHUNK_BYTES])
            return True
        except queue.Full:
            self.dropped_bytes = len(data)
            self.overrun = True
            if self._report is not None:
                self._report("stdin audio overran the processing queue; ending the session")
            return False

    def _receive(self, read) -> None:
        carry = b""
        try:
            while not self._stopped.is_set():
                data = read(PCM_CHUNK_BYTES)
                if data is None or self._stopped.is_set():
                    break
                if not data:
                    tail = len(carry) % PCM_SAMPLE_WIDTH
                    self.discarded_tail_bytes = tail
                    if len(carry) > tail:
                        self._offer(carry[: len(carry) - tail])
                    if tail and self._report is not None:
                        self._report("stdin audio ended mid-sample; discarded the final byte")
                    break
                data = carry + data
                while len(data) >= PCM_CHUNK_BYTES and self._offer(data):
                    data = data[PCM_CHUNK_BYTES:]
                if self.overrun:
                    break
                carry = data
        except (OSError, ValueError):
            self._failure = "Stdin audio input failed"
        carry = data = b""
        self._eof.set()

    def read(self, timeout: float = 0.25) -> bytes | None:
        """Return PCM chunks; None means none yet; b"" means the input ended (EOF/overrun)."""
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
        """Discard pending PCM; a reader blocked on the stream exits at its next read.

        A descriptor-backed reader notices stop within ``STDIN_POLL_SECONDS`` and is
        joined here, so it has exited before the interpreter can finalize. A test
        double's blocking read cannot be interrupted portably, so the stream belongs to
        this capture alone: whatever the stopped reader still reads is discarded, never
        queued, and the controller refuses to start another session on that stream.
        """
        self._stopped.set()
        reader = self._reader
        if reader is not None and not reader.daemon and reader is not threading.current_thread():
            reader.join(timeout=2)
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break


def _descriptor(stream) -> int | None:
    """The stream's OS file descriptor when it can be polled, otherwise None."""
    try:
        descriptor = stream.fileno()
        select.select([descriptor], [], [], 0)
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    return descriptor if isinstance(descriptor, int) else None
