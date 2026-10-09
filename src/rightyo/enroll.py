"""`rightyo enroll`: local speaker enrollment (#137 step 3, privacy rules from #109).

A voiceprint is the renormalised mean of L2-normalised speaker embeddings over
overlapping windows of at least 20 s of speech. Only that vector and minimal metadata
(identifier, display name, model identity, creation time, seconds used) are stored, one
JSON file per enrolled speaker, in a private directory (mode 700, files 600) on this
machine, never inside a Git checkout. Audio is read or recorded into memory, embedded,
and discarded; it is never written. Output and errors carry identifiers, display names,
durations and scores only: no audio, embeddings, transcript text or local paths.

Live identification (step 4) is not wired yet; `verify` scores a supplied clip against
the enrolled voiceprints so a user can sanity-check enrollment locally.
"""

from __future__ import annotations

import json
import math
import operator
import os
import re
import stat
import sys
import time
import wave
from array import array
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TextIO

from rightyo.speaker_id import (
    DEFAULT_STORE,
    SAMPLE_RATE,
    SpeakerEmbedder,
    SpeakerIdConfig,
    SpeakerIdError,
)

SCHEMA_VERSION = 1
FRAME = 160  # 10 ms at 16 kHz
MAX_INPUT_SECONDS = 300
MIN_ENROLL_SECONDS = 20.0
WARN_ENROLL_SECONDS = 30.0
MAX_ENROLL_SECONDS = 90.0
MAX_VERIFY_SECONDS = 30.0
WINDOW_SECONDS = 3.0
HOP_SECONDS = 1.5
MAX_GAP_FRAMES = 25  # pauses inside speech are shortened to 0.25 s
RELATIVE_FLOOR = 10 ** (-35 / 10)  # a frame is speech within 35 dB of the loudest frame
ABSOLUTE_FLOOR = 1e-6 * 32768**2  # and above -60 dBFS
MAX_ENTRY_BYTES = 65536
IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9_-]{0,31}")
TEMPORARY = re.compile(r"\.[a-z0-9][a-z0-9_-]{0,31}\.json\.tmp-[0-9]+")


class EnrollError(ValueError):
    """A safe, fixed-message failure: never a path, audio, embedding or worker output."""


# ----------------------------------------------------------------------------- audio


def read_wav(path: Path) -> array:
    """Samples of a 16 kHz mono PCM16 WAV of at most five minutes."""
    try:
        with wave.open(str(path), "rb") as source:
            if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (
                1,
                2,
                SAMPLE_RATE,
            ):
                raise EnrollError("Audio must be 16 kHz mono 16-bit PCM WAV")
            if source.getnframes() > MAX_INPUT_SECONDS * SAMPLE_RATE:
                raise EnrollError("Audio must be at most five minutes long")
            data = source.readframes(source.getnframes())
    except (OSError, EOFError, wave.Error):
        raise EnrollError("Audio could not be read as 16 kHz mono 16-bit PCM WAV") from None
    samples = array("h")
    samples.frombytes(data[: len(data) // 2 * 2])
    if sys.byteorder == "big":
        samples.byteswap()
    return samples


def speech(samples: array) -> array:
    """Speech only: leading and trailing silence removed, pauses cut to 0.25 s.

    A 10 ms frame counts as speech when its energy is within 35 dB of the loudest frame
    and above -60 dBFS. This is a level gate, not a voice detector: steady noise or
    music counts as "speech", so enroll from a quiet recording of one person talking.
    """
    count = len(samples) // FRAME
    energies = []
    for index in range(count):
        frame = samples[index * FRAME : (index + 1) * FRAME]
        energies.append(sum(map(operator.mul, frame, frame)) / FRAME)
    floor = max(max(energies, default=0.0) * RELATIVE_FLOOR, ABSOLUTE_FLOOR)
    voiced = [energy > floor for energy in energies]
    kept = array("h")
    pending: list[int] = []
    for index, is_speech in enumerate(voiced):
        if is_speech:
            for held in pending[:MAX_GAP_FRAMES]:
                kept.extend(samples[held * FRAME : (held + 1) * FRAME])
            pending = []
            kept.extend(samples[index * FRAME : (index + 1) * FRAME])
        elif kept:
            pending.append(index)
    return kept


def windows(samples: array, seconds: float = WINDOW_SECONDS, hop: float = HOP_SECONDS):
    """Overlapping windows; the last is aligned to the end so every sample is covered."""
    size, step = int(seconds * SAMPLE_RATE), int(hop * SAMPLE_RATE)
    if len(samples) <= size:
        return [samples]
    starts = list(range(0, len(samples) - size + 1, step))
    if starts[-1] + size < len(samples):
        starts.append(len(samples) - size)
    return [samples[start : start + size] for start in starts]


def _pcm(samples: array) -> bytes:
    if sys.byteorder == "big":
        samples = array("h", samples)
        samples.byteswap()
    return samples.tobytes()


# ----------------------------------------------------------------------------- vectors


def normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if not math.isfinite(norm) or norm < 1e-9:
        raise EnrollError("The speaker model returned an unusable embedding")
    return [value / norm for value in vector]


def voiceprint(embed: Callable[[bytes], list[float]], samples: array) -> tuple[list[float], int]:
    """Mean of L2-normalised window embeddings, renormalised, and the window count."""
    vectors = [normalize(embed(_pcm(window))) for window in windows(samples)]
    if len({len(vector) for vector in vectors}) != 1:
        raise EnrollError("The speaker model returned inconsistent embeddings")
    mean = [sum(column) / len(vectors) for column in zip(*vectors)]
    return normalize(mean), len(vectors)


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


# ----------------------------------------------------------------------------- store


def inside_git_checkout(path: Path) -> bool:
    resolved = path.expanduser().resolve()
    return any((parent / ".git").exists() for parent in (resolved, *resolved.parents))


def _identifier(value: Any) -> str:
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise EnrollError(
            "An identifier is 1 to 32 lowercase letters, digits, '-' or '_', "
            "starting with a letter or digit"
        )
    return value


def _display_name(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 64
        or not value.isprintable()
        or value != value.strip()
    ):
        raise EnrollError("A display name is 1 to 64 printable characters")
    return value


def _entry(document: Any, identifier: str) -> dict[str, Any]:
    """A validated stored entry, or ValueError."""
    keys = {
        "schema_version",
        "id",
        "display_name",
        "model",
        "created_at",
        "speech_seconds",
        "windows",
        "embedding",
    }
    if not isinstance(document, dict) or set(document) != keys:
        raise ValueError
    if document["schema_version"] != SCHEMA_VERSION or document["id"] != identifier:
        raise ValueError
    _display_name(document["display_name"])
    model = document["model"]
    if (
        not isinstance(model, dict)
        or set(model) != {"id", "revision", "sha256"}
        or not all(isinstance(value, str) for value in model.values())
    ):
        raise ValueError
    vector = document["embedding"]
    if (
        not isinstance(vector, list)
        or not 1 <= len(vector) <= 1024
        or not all(type(value) in (int, float) and math.isfinite(value) for value in vector)
        or abs(math.sqrt(sum(value * value for value in vector)) - 1) > 1e-3
    ):
        raise ValueError
    if not isinstance(document["created_at"], str) or type(document["windows"]) is not int:
        raise ValueError
    seconds = document["speech_seconds"]
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError
    return document


class Store:
    """One private directory of `<id>.json` voiceprints; never inside a Git checkout."""

    def __init__(self, path: Path | None = None):
        path = (DEFAULT_STORE if path is None else Path(path)).expanduser()
        if not path.is_absolute():
            raise EnrollError("The enrollment store must be an absolute path")
        if inside_git_checkout(path):
            raise EnrollError("Refusing to keep enrollment data inside a Git checkout")
        self.path = path

    def _check(self) -> bool:
        """Whether the store exists; tightens its permissions and refuses anything odd."""
        try:
            info = os.lstat(self.path)
        except FileNotFoundError:
            return False
        except OSError:
            raise EnrollError("The enrollment store could not be read") from None
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
            raise EnrollError("The enrollment store must be a directory owned by you")
        try:
            os.chmod(self.path, 0o700)
            for child in self.path.iterdir():
                if child.suffix == ".json" and child.is_file() and not child.is_symlink():
                    os.chmod(child, 0o600)
        except OSError:
            raise EnrollError("The enrollment store could not be secured") from None
        return True

    def create(self) -> None:
        if self._check():
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            os.mkdir(self.path, 0o700)
        except FileExistsError:
            pass
        except OSError:
            raise EnrollError("The enrollment store could not be created") from None
        self._check()

    def _file(self, identifier: str) -> Path:
        return self.path / f"{_identifier(identifier)}.json"

    def entries(self) -> tuple[list[dict[str, Any]], int]:
        """Valid entries sorted by identifier, and the number of unreadable files."""
        if not self._check():
            return [], 0
        entries, unreadable = [], 0
        for child in sorted(self.path.glob("*.json")):
            if not IDENTIFIER.fullmatch(child.stem):
                unreadable += 1
                continue
            try:
                if child.is_symlink():
                    raise ValueError
                with child.open("rb") as source:
                    content = source.read(MAX_ENTRY_BYTES + 1)
                if len(content) > MAX_ENTRY_BYTES:
                    raise ValueError
                entries.append(_entry(json.loads(content), child.stem))
            except (OSError, ValueError, OverflowError, RecursionError, EnrollError):
                unreadable += 1
        return entries, unreadable

    def save(self, entry: dict[str, Any], *, replace: bool = False) -> None:
        target = self._file(entry["id"])
        self.create()
        if target.exists() and not replace:
            raise EnrollError("That identifier is already enrolled; pass --replace or delete it")
        temporary = self.path / f".{entry['id']}.json.tmp-{os.getpid()}"
        payload = json.dumps(entry, allow_nan=False, indent=2).encode() + b"\n"
        try:
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            try:
                os.fchmod(fd, 0o600)
                os.write(fd, payload)
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(temporary, target)
        except OSError:
            temporary.unlink(missing_ok=True)
            raise EnrollError("The voiceprint could not be stored") from None

    def _ours(self, child: Path) -> bool:
        """Whether a file is a RightyO voiceprint or one of our temporary files.

        Deletion never trusts the directory alone: a mistaken `--store` (a home or project
        directory) must not lose unrelated files. A voiceprint is a small JSON object
        carrying our schema keys; a dangling link named like one is ours to remove.
        """
        name = child.name
        if TEMPORARY.fullmatch(name):
            return True
        if not (child.suffix == ".json" and IDENTIFIER.fullmatch(child.stem)):
            return False
        if child.is_symlink():
            return not child.exists()
        try:
            with open(
                child, "rb", opener=lambda path, flags: os.open(path, flags | os.O_NOFOLLOW)
            ) as source:
                content = source.read(MAX_ENTRY_BYTES + 1)
            document = json.loads(content) if len(content) <= MAX_ENTRY_BYTES else None
        except (OSError, ValueError, RecursionError):
            return False
        return (
            isinstance(document, dict)
            and document.get("schema_version") == SCHEMA_VERSION
            and {"id", "model", "embedding"} <= set(document)
        )

    def delete(self, identifier: str) -> bool:
        target = self._file(identifier)
        # lexists: a dangling `<id>.json` link is still removed, not reported as absent.
        if not self._check() or not os.path.lexists(target):
            return False
        if not self._ours(target):
            raise EnrollError("That file is not a RightyO voiceprint; nothing was deleted")
        try:
            target.unlink()
        except OSError:
            raise EnrollError("The voiceprint could not be deleted") from None
        return True

    def delete_all(self) -> list[str]:
        """Remove every voiceprint and leftover temporary file, then the directory."""
        if not self._check():
            return []
        removed = []
        try:
            for child in sorted(self.path.iterdir()):
                if (child.is_file() or child.is_symlink()) and self._ours(child):
                    if child.suffix == ".json":
                        removed.append(child.stem)
                    child.unlink()
            if not any(self.path.iterdir()):
                self.path.rmdir()
        except OSError:
            raise EnrollError("The enrollment store could not be fully deleted") from None
        return removed


# ----------------------------------------------------------------------------- commands


def _record(helper: Path, seconds: int, capture_factory, err: TextIO) -> array:
    """Record from the microphone into memory only; nothing is written."""
    from rightyo.capture import CaptureError

    needed = seconds * SAMPLE_RATE * 2
    buffer = bytearray()
    capture = capture_factory(helper)
    print(f"rightyo: recording {seconds} s from the microphone; speak now", file=err)
    try:
        capture.start()
        # Allow time for a first-use permission prompt before audio arrives.
        deadline = time.monotonic() + seconds + 60
        while len(buffer) < needed:
            if time.monotonic() > deadline:
                raise EnrollError("Recording did not finish in time")
            chunk = capture.read(0.5)
            if chunk:
                buffer.extend(chunk)
    except CaptureError as error:
        raise EnrollError(str(error)) from None
    finally:
        capture.stop()
    print("rightyo: recording finished", file=err)
    samples = array("h")
    samples.frombytes(bytes(buffer[:needed]))
    buffer[:] = b""
    if sys.byteorder == "big":
        samples.byteswap()
    return samples


def _audio(args, config, capture_factory, err: TextIO) -> array:
    if args.record is not None:
        if config is None:
            raise EnrollError("--record needs --config for the microphone helper")
        return _record(config.microphone_helper, args.record, capture_factory, err)
    return read_wav(args.audio)


def _speaker_config(config) -> SpeakerIdConfig:
    settings = getattr(config, "speaker_id", None) if config is not None else None
    if settings is None:
        raise EnrollError("The configuration has no enabled speaker_id section")
    return settings


def _load_config(args):
    if args.config is None:
        return None
    from rightyo.prototype import PrototypeConfig, PrototypeError

    try:
        return PrototypeConfig.load(args.config)
    except PrototypeError as error:
        raise EnrollError(str(error)) from None


def _store(args, settings: SpeakerIdConfig | None) -> Store:
    if getattr(args, "store", None) is not None:
        return Store(args.store)
    return Store(None if settings is None else settings.store)


def _default_embedder(settings: SpeakerIdConfig):
    return SpeakerEmbedder(settings.python, settings.model, threads=settings.threads)


def _embedded(settings, embedder_factory, samples) -> tuple[list[float], int, dict[str, str]]:
    embedder = embedder_factory(settings)
    try:
        vector, count = voiceprint(embedder.embed, samples)
        return vector, count, dict(embedder.model)
    finally:
        embedder.close()


def add(args, config, *, embedder_factory, capture_factory, err: TextIO) -> dict[str, Any]:
    settings = _speaker_config(config)
    identifier = _identifier(args.id)
    display = _display_name(args.name if args.name is not None else identifier)
    store = _store(args, settings)
    if not args.replace and store.path.joinpath(f"{identifier}.json").exists():
        raise EnrollError("That identifier is already enrolled; pass --replace or delete it")
    samples = speech(_audio(args, config, capture_factory, err))
    seconds = len(samples) / SAMPLE_RATE
    if seconds < MIN_ENROLL_SECONDS:
        raise EnrollError(
            f"Only {seconds:.1f} s of speech found; enrollment needs at least "
            f"{MIN_ENROLL_SECONDS:.0f} s (30 to 60 s is better)"
        )
    if seconds < WARN_ENROLL_SECONDS:
        print(
            f"rightyo: warning: {seconds:.1f} s of speech; 30 to 60 s gives a steadier voiceprint",
            file=err,
        )
    samples = samples[: int(MAX_ENROLL_SECONDS * SAMPLE_RATE)]
    vector, count, model = _embedded(settings, embedder_factory, samples)
    used = round(len(samples) / SAMPLE_RATE, 2)
    store.save(
        {
            "schema_version": SCHEMA_VERSION,
            "id": identifier,
            "display_name": display,
            "model": model,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "speech_seconds": used,
            "windows": count,
            "embedding": [round(value, 7) for value in vector],
        },
        replace=args.replace,
    )
    return {
        "enrolled": identifier,
        "display_name": display,
        "speech_seconds": used,
        "windows": count,
        "model": model["id"],
    }


def list_entries(args, config) -> dict[str, Any]:
    settings = getattr(config, "speaker_id", None) if config is not None else None
    entries, unreadable = _store(args, settings).entries()
    return {
        "enrolled": [
            {
                "id": entry["id"],
                "display_name": entry["display_name"],
                "created_at": entry["created_at"],
                "speech_seconds": entry["speech_seconds"],
                "model": entry["model"]["id"],
                "model_revision": entry["model"]["revision"],
            }
            for entry in entries
        ],
        "unreadable_entries": unreadable,
    }


def delete(args, config) -> dict[str, Any]:
    settings = getattr(config, "speaker_id", None) if config is not None else None
    store = _store(args, settings)
    if args.all:
        return {"deleted": store.delete_all()}
    identifier = _identifier(args.id)
    if not store.delete(identifier):
        raise EnrollError("No voiceprint is enrolled under that identifier")
    return {"deleted": [identifier]}


def verify(args, config, *, embedder_factory, capture_factory, err: TextIO) -> dict[str, Any]:
    settings = _speaker_config(config)
    entries, unreadable = _store(args, settings).entries()
    if not entries:
        raise EnrollError("No speakers are enrolled")
    samples = speech(_audio(args, config, capture_factory, err))
    seconds = len(samples) / SAMPLE_RATE
    if seconds < settings.min_turn_seconds:
        raise EnrollError(
            f"Only {seconds:.1f} s of speech found; verification needs at least "
            f"{settings.min_turn_seconds:.1f} s"
        )
    samples = samples[: int(MAX_VERIFY_SECONDS * SAMPLE_RATE)]
    vector, _, model = _embedded(settings, embedder_factory, samples)
    scores, other_model = [], 0
    for entry in entries:
        if entry["model"]["sha256"] != model["sha256"] or len(entry["embedding"]) != len(vector):
            other_model += 1
            continue
        score = cosine(vector, entry["embedding"])
        scores.append(
            {
                "id": entry["id"],
                "display_name": entry["display_name"],
                "score": round(score, 3),
                "band": settings.band(score),
            }
        )
    scores.sort(key=lambda item: (-item["score"], item["id"]))
    return {
        "speech_seconds": round(len(samples) / SAMPLE_RATE, 2),
        "scores": scores,
        "thresholds": {
            "bind": settings.bind_threshold,
            "tentative": settings.tentative_threshold,
        },
        "other_model_entries": other_model,
        "unreadable_entries": unreadable,
    }


def run(
    args,
    *,
    output: TextIO | None = None,
    errors: TextIO | None = None,
    embedder_factory=_default_embedder,
    capture_factory=None,
) -> int:
    output = sys.stdout if output is None else output
    errors = sys.stderr if errors is None else errors
    if capture_factory is None:
        from rightyo.capture import MacMicrophoneCapture

        capture_factory = MacMicrophoneCapture
    try:
        config = _load_config(args)
        if args.enroll_command == "add":
            result = add(
                args,
                config,
                embedder_factory=embedder_factory,
                capture_factory=capture_factory,
                err=errors,
            )
        elif args.enroll_command == "verify":
            result = verify(
                args,
                config,
                embedder_factory=embedder_factory,
                capture_factory=capture_factory,
                err=errors,
            )
        elif args.enroll_command == "list":
            result = list_entries(args, config)
        else:
            result = delete(args, config)
    except (EnrollError, SpeakerIdError) as error:
        print(f"rightyo: {error}", file=errors)
        return 2
    print(json.dumps(result, allow_nan=False, sort_keys=True, indent=2), file=output)
    return 0
