"""Local host-rendered self references. Load once on a dedicated background thread."""

import hashlib
import io
import json
import math
import os
import re
import secrets
import stat
import struct
import sys
import wave
from array import array
from contextlib import ExitStack, contextmanager
from pathlib import Path

from rightyo.enroll import SELF_VOICE_CACHE, Store, speech, voiceprint
from rightyo.speaker_id import MAX_EMBEDDING_DIM, SAMPLE_RATE, finite_number

MANIFEST_BYTES = 16 * 1024
WAV_BYTES = 1024 * 1024
# Outside the store's <person-id>.json namespace.
CACHE_NAME = SELF_VOICE_CACHE
CACHE_BYTES = 1024 * 1024


class ProfileError(ValueError):
    """The message is a fixed diagnostic token."""


def _check(info, directory):
    if stat.S_ISLNK(info.st_mode):
        raise ProfileError("symlink")
    if not (stat.S_ISDIR if directory else stat.S_ISREG)(info.st_mode):
        raise ProfileError("file_type")
    if info.st_uid != os.getuid():
        raise ProfileError("owner")
    if info.st_mode & 0o077:
        raise ProfileError("permissions")


@contextmanager
def _directory(path):
    _check(os.lstat(path), True)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_DIRECTORY)
    try:
        _check(os.fstat(fd), True)
        yield fd
    finally:
        os.close(fd)


def _read(directory, name, limit, *, mode=None):
    _check(os.lstat(name, dir_fd=directory), False)
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    with os.fdopen(os.open(name, flags, dir_fd=directory), "rb") as stream:
        info = os.fstat(stream.fileno())
        _check(info, False)
        if mode is not None and stat.S_IMODE(info.st_mode) != mode:
            raise ProfileError("permissions")
        if info.st_size > limit:
            raise ProfileError("size")
        data = stream.read(limit + 1)
        if len(data) > limit:
            raise ProfileError("size")
        return data


def _unique(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ProfileError("duplicate_key")
    return result


def _voices(data):
    try:
        doc = json.loads(data.decode("utf-8"), object_pairs_hook=_unique)
    except ProfileError:
        raise
    except (ValueError, RecursionError):
        raise ProfileError("json") from None
    if not isinstance(doc, dict) or doc.keys() != {"version", "voices"}:
        raise ProfileError("manifest_keys")
    if type(doc["version"]) is not int or doc["version"] != 1:
        raise ProfileError("version")
    voices = doc["voices"]
    if not isinstance(voices, list) or len(voices) > 16:
        raise ProfileError("voices")
    seen = set()
    for voice in voices:
        if not isinstance(voice, dict) or voice.keys() != {
            "id",
            "file",
            "source",
            "rendered_at",
            "sha256",
        }:
            raise ProfileError("voice_keys")
        identifier = voice["id"]
        if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", identifier):
            raise ProfileError("id")
        if identifier in seen:
            raise ProfileError("duplicate_id")
        seen.add(identifier)
        name = voice["file"]
        if (
            not isinstance(name, str)
            or not name.endswith(".wav")
            or name.startswith(".")
            or any(piece in name for piece in ("/", "\\", "..", "\0"))
        ):
            raise ProfileError("file_name")
        if voice["source"] not in ("reply", "ack", "device_cue"):
            raise ProfileError("source")
        if type(voice["rendered_at"]) is not int or voice["rendered_at"] < 0:
            raise ProfileError("rendered_at")
        if not isinstance(voice["sha256"], str) or not re.fullmatch(
            "[0-9a-f]{64}", voice["sha256"]
        ):
            raise ProfileError("sha256")
    return voices


def _samples(data):
    # wave rounds bits per sample up to whole bytes. Verify the PCM format field
    # as well, and reject truncated chunks before passing the bytes to the decoder.
    if (
        data[:4] != b"RIFF"
        or data[8:12] != b"WAVE"
        or int.from_bytes(data[4:8], "little") != len(data) - 8
    ):
        return None
    offset = 12
    seen = set()
    while offset + 8 <= len(data):
        kind, size = struct.unpack_from("<4sI", data, offset)
        offset += 8
        if offset + size > len(data):
            return None
        if kind in (b"fmt ", b"data"):
            if kind in seen:
                return None
            seen.add(kind)
        if kind == b"fmt ":
            if size < 16:
                return None
            tag, channels, rate, byte_rate, align, bits = struct.unpack_from(
                "<HHIIHH", data, offset
            )
            if (channels, rate, byte_rate, align, bits) != (1, SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16):
                return None
            if tag == 0xFFFE:
                # WAVE_FORMAT_EXTENSIBLE with the PCM subtype. Present that PCM tag
                # to wave on Python 3.11, which predates extensible header support.
                if (
                    size < 40
                    or struct.unpack_from("<HH", data, offset + 16) != (22, 16)
                    or data[offset + 24 : offset + 40]
                    != bytes.fromhex("0100000000001000800000aa00389b71")
                ):
                    return None
                data = data[:offset] + b"\x01\x00" + data[offset + 2 :]
            elif tag != 1:
                return None
        if kind == b"data" and size % 2:
            return None
        offset += size + size % 2
    if offset != len(data):
        return None
    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            frames = wav.getnframes()
            if (wav.getcomptype(), wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (
                "NONE",
                1,
                2,
                SAMPLE_RATE,
            ) or not 8 * SAMPLE_RATE <= frames <= 20 * SAMPLE_RATE:
                return None
            pcm = wav.readframes(frames)
            if len(pcm) != 2 * frames:
                return None
    except (EOFError, wave.Error, struct.error, RuntimeError, ValueError, OSError):
        return None
    samples = array("h")
    samples.frombytes(pcm)
    if sys.byteorder == "big":
        samples.byteswap()
    return samples


def _check_stop(stop):
    if stop.is_set():
        raise ProfileError("stopped")


def _clips(path, stop):
    path = Path(path)
    if not path.is_absolute():
        raise ProfileError("absolute_path")
    clips, skipped = [], []
    files = {}
    with _directory(path.parent) as directory:
        voices = _voices(_read(directory, path.name, MANIFEST_BYTES))
        for voice in voices:
            _check_stop(stop)
            name = voice["file"]
            if name not in files:
                data = _read(directory, name, WAV_BYTES)
                files[name] = hashlib.sha256(data).hexdigest(), _samples(data)
            digest, samples = files[name]
            if digest != voice["sha256"]:
                skipped.append("digest")
                continue
            if samples is None:
                skipped.append("wav")
                continue
            clips.append((voice, samples))
    return clips, skipped


def _cache(directory):
    try:
        cached = json.loads(
            _read(directory, CACHE_NAME, CACHE_BYTES, mode=0o600), object_pairs_hook=_unique
        )
        if (
            isinstance(cached, dict)
            and len(cached) <= 16
            and all(
                isinstance(vector, list)
                and 1 <= len(vector) <= MAX_EMBEDDING_DIM
                and all(finite_number(v) for v in vector)
                and abs(math.hypot(*vector) - 1) < 1e-6
                for vector in cached.values()
            )
        ):
            return cached
    except (OSError, ValueError, RecursionError, OverflowError):
        pass
    return None


def _save(directory, vectors):
    name = f".self-voice-tmp-{secrets.token_hex(12)}"
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=directory)
    try:
        with os.fdopen(fd, "w") as temporary:
            os.fchmod(fd, 0o600)
            json.dump(vectors, temporary, allow_nan=False)
        os.replace(name, CACHE_NAME, src_dir_fd=directory, dst_dir_fd=directory)
    finally:
        try:
            os.unlink(name, dir_fd=directory)
        except FileNotFoundError:
            pass


def _vectors(clips, model_sha, cached, open_embedder, stop):
    vectors, missing = {}, []
    for voice, samples in clips:
        _check_stop(stop)
        key = f"{model_sha}:{voice['id']}:{voice['rendered_at']}:{voice['sha256']}"
        if key in cached:
            vectors[key] = cached[key]
        else:
            missing.append((key, samples))
    if missing:
        _check_stop(stop)
        with open_embedder() as embedder:
            if model_sha and embedder.model.get("sha256") != model_sha:
                raise ProfileError("model")

            def embed(pcm):
                _check_stop(stop)
                return embedder.embed(pcm)

            for key, samples in missing:
                _check_stop(stop)
                voiced = speech(samples)
                if not voiced:
                    raise ProfileError("embedding")
                vectors[key], _ = voiceprint(embed, voiced)
    return vectors


def load_profile(path, settings, model_sha, open_embedder, report, stop):
    """Return the complete reference set. A cache hit needs no reference embedder."""
    try:
        clips, skipped = _clips(path, stop)
        with ExitStack() as resources:
            directory = None
            if model_sha:
                try:
                    store = Store(settings.store)
                    fd = resources.enter_context(_directory(store.path))
                    if store.marked(fd):
                        directory = fd
                except (ValueError, OSError):
                    pass  # An unavailable store leaves reference loading compute-only.
            cached = _cache(directory) if directory is not None else None
            vectors = _vectors(clips, model_sha, cached or {}, open_embedder, stop)
            _check_stop(stop)
            if directory is not None and cached != vectors and store.marked(directory):
                try:
                    _save(directory, vectors)
                except OSError:
                    raise ProfileError("cache_write") from None
    except ProfileError as error:
        report(f"self_profile voices=0 unavailable reason={error}")
        return ()
    except OSError:
        report("self_profile voices=0 unavailable reason=read")
        return ()
    except Exception:
        report("self_profile voices=0 unavailable reason=embedding")
        return ()
    for rule in skipped:
        report(f"self_profile skipped reason={rule}")
    reason = "loaded" if vectors else (skipped[-1] if skipped else "empty")
    status = "ready" if vectors else "unavailable"
    report(f"self_profile voices={len(vectors)} {status} reason={reason}")
    return tuple(vectors.values())
