"""Synthetic host references and deterministic speaker vectors."""

import fcntl
import hashlib
import io
import json
import os
import signal
import subprocess
import tempfile
import threading
import time
import unittest
import wave
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, redirect_stderr
from pathlib import Path
from unittest.mock import Mock, patch

from test_live_speaker_id import (
    ENTRIES,
    ScriptedEmbedder,
    TurnEveryTwoSeconds,
    settings,
    unit,
    wait_for,
)
from test_live_speaker_id import turn as labelled

from rightyo.contracts import SpeakerPriority
from rightyo.enroll import STORE_MARKER, Store, speech, voiceprint
from rightyo.live_speaker_id import EnrolledRoles, ShadowSpeakerId
from rightyo.prototype import PrototypeController
from rightyo.self_profile import CACHE_NAME, MANIFEST_BYTES, WAV_BYTES, load_profile
from rightyo.speaker_id import SAMPLE_RATE, SpeakerEmbedder
from rightyo.tool import listen

PCM = b"\x00\x20" * SAMPLE_RATE


def wav(seconds=8, channels=1, width=2, rate=SAMPLE_RATE):
    stream = io.BytesIO()
    with wave.open(stream, "wb") as writer:
        writer.setparams((channels, width, rate, 0, "NONE", ""))
        writer.writeframes(PCM * seconds)
    return stream.getvalue()


class References:
    def __init__(self, root):
        directory = root / "ref"
        directory.mkdir(mode=0o700)
        self.path = directory / "manifest"
        self.config = settings(store=root / "store")
        self.model = ScriptedEmbedder()
        self.open_model = Mock(side_effect=lambda: closing(self.model))
        self.notes = []
        self.voice = self.clip(wav())
        self.write([self.voice])

    def clip(self, data, name="a"):
        path = self.path.parent / (name + ".wav")
        path.write_bytes(data)
        path.chmod(0o600)
        return dict(
            id=name,
            file=path.name,
            sha256=hashlib.sha256(data).hexdigest(),
            source="ack",
            rendered_at=0,
        )

    def write(self, voices):
        self.path.write_text(json.dumps(dict(version=1, voices=voices)))
        self.path.chmod(0o600)

    def load(self):
        self.notes.clear()
        return load_profile(
            self.path,
            self.config,
            self.model.model.get("sha256"),
            self.open_model,
            self.notes.append,
            threading.Event(),
        )

    @property
    def cache(self):
        return self.config.store / CACHE_NAME


def score(shadow, vector, *, inside=True, seconds=1):
    shadow._score(
        ScriptedEmbedder([vector]),
        {"owner": unit(1, 0, 0)},
        shadow.epoch,
        "label",
        seconds * 1000,
        seconds * 1000,
        PCM * seconds,
        inside,
    )


def make_shadow():
    worker = ShadowSpeakerId(
        settings(bind_min_seconds=2), report=Mock(), self_voice_manifest="unused"
    )
    worker._self_prints = (unit(0, 1, 0), unit(0, 0, 1))
    return worker


def listen_args(refs, *, enabled=True, live=True):
    path = refs.path.parent / "config"
    asset = str(refs.path)
    config = dict.fromkeys(
        (
            "whisper_executable",
            "whisper_model",
            "diarization_library",
            "diarization_model",
            "microphone_helper",
        ),
        asset,
    )
    config.update(
        self_voice={"enabled": enabled},
        speaker_id={"python": asset, "model": asset, "store": str(refs.config.store), "live": live},
    )
    path.write_text(json.dumps(config))
    return Namespace(
        config=path,
        mode="stdin",
        provenance="synthetic",
        session_id="synthetic",
        use_jev=False,
        allow_hosted=False,
        self_voice_manifest=refs.path,
    )


def session_bytes(args):
    def worker(config, **options):
        return ShadowSpeakerId(
            config,
            **options,
            entries=lambda _: ENTRIES,
            embedder_factory=lambda _: ScriptedEmbedder(),
        )

    def controller(config, **options):
        ctl = PrototypeController(
            config, **options, processor_factory=TurnEveryTwoSeconds, speaker_id_factory=worker
        )
        ctl._now_ms = lambda: ctl._received_ms
        return ctl

    output, notes = io.StringIO(), io.StringIO()
    with redirect_stderr(notes), patch("rightyo.tool.host_options", return_value={}):
        assert (
            listen(
                args, output=output, controller_factory=controller, audio_input=io.BytesIO(PCM * 2)
            )
            == 0
        )
    return output.getvalue(), notes.getvalue()


class ProfileTests(unittest.TestCase):
    def references(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        return References(Path(directory.name))

    def test_invalid_voice_rejects_manifest(self):
        for field, bad, reason in [
            ("id", "", "id"),
            ("id", "a" * 65, "id"),
            ("id", "é", "id"),
            ("id", [], "id"),
            ("file", "a", "file_name"),
            ("file", "a/b.wav", "file_name"),
            ("file", "a\\b.wav", "file_name"),
            ("file", "a..b.wav", "file_name"),
            ("file", ".a.wav", "file_name"),
            ("file", "a\0.wav", "file_name"),
            ("source", [], "source"),
            ("source", "other", "source"),
            ("rendered_at", True, "rendered_at"),
            ("rendered_at", -1, "rendered_at"),
            ("rendered_at", 1.5, "rendered_at"),
            ("sha256", "A" * 64, "sha256"),
            ("sha256", "a" * 63, "sha256"),
            ("extra", 1, "voice_keys"),
        ]:
            with self.subTest(field=field, bad=bad, reason=reason):
                refs = self.references()
                refs.write([refs.voice, {**refs.voice, "id": "b", field: bad}])
                assert refs.load() == ()
                assert refs.notes == [f"self_profile voices=0 unavailable reason={reason}"]
                assert refs.model.calls == 0

    def test_manifest_shape(self):
        for data, reason in [
            (b"{", "json"),
            (b"\xff", "json"),
            (b"[]", "manifest_keys"),
            (b"{}", "manifest_keys"),
            (b'{"version":1,"voices":[],"extra":0}', "manifest_keys"),
            (b'{"version":1,"version":1,"voices":[]}', "duplicate_key"),
            (b'{"version":true,"voices":[]}', "version"),
            (b'{"version":1.0,"voices":[]}', "version"),
            (b'{"version":2,"voices":[]}', "version"),
            (b'{"version":1,"voices":{}}', "voices"),
            (b'{"version":1,"voices":[{}]}', "voice_keys"),
        ]:
            with self.subTest(data=data, reason=reason):
                refs = self.references()
                refs.path.write_bytes(data)
                assert refs.load() == ()
                assert refs.notes == [f"self_profile voices=0 unavailable reason={reason}"]

    def test_manifest_bounds(self):
        refs = self.references()
        refs.write([refs.voice, refs.voice])
        assert not refs.load()
        assert refs.notes[-1].endswith("reason=duplicate_id")
        maximum = [
            {
                **refs.voice,
                "id": "_A-" + str(i).zfill(61),
                "source": ("ack", "reply", "device_cue")[i % 3],
            }
            for i in range(16)
        ]
        refs.write(maximum)
        refs.path.write_bytes(refs.path.read_bytes().ljust(MANIFEST_BYTES))
        assert len(refs.load()) == 16
        with refs.path.open("ab") as target:
            target.write(b" ")
        assert not refs.load()
        assert refs.notes[-1].endswith("reason=size")
        refs.write(maximum + [refs.voice])
        assert not refs.load()
        assert refs.notes[-1].endswith("reason=voices")

    def test_permissions(self):
        for node in ["directory", "manifest", "wav"]:
            with self.subTest(node=node):
                for mode in [0o070, 0o004, 0o002]:
                    with self.subTest(mode=mode):
                        refs = self.references()
                        b = refs.clip(wav(), name="b")
                        refs.write([refs.voice, b])
                        path = {
                            "directory": refs.path.parent,
                            "manifest": refs.path,
                            "wav": refs.path.parent / b["file"],
                        }[node]
                        path.chmod(path.stat().st_mode | mode)
                        assert refs.load() == ()
                        assert refs.notes == [
                            "self_profile voices=0 unavailable reason=permissions"
                        ]

    def test_symlink_rejects_profile(self):
        for node in ["directory", "manifest", "wav"]:
            with self.subTest(node=node):
                refs = self.references()
                b = refs.clip(wav(), name="b")
                refs.write([refs.voice, b])
                path = {
                    "directory": refs.path.parent,
                    "manifest": refs.path,
                    "wav": refs.path.parent / b["file"],
                }[node]
                other = path.with_name("target")
                path.rename(other)
                path.symlink_to(other)
                assert refs.load() == ()
                assert refs.notes == ["self_profile voices=0 unavailable reason=symlink"]
                assert refs.model.calls == 0

    def test_wrong_uid(self):
        for node in ["directory", "manifest", "wav"]:
            with self.subTest(node=node):
                refs = self.references()
                target = {
                    "directory": refs.path.parent,
                    "manifest": refs.path,
                    "wav": refs.path.with_name("a.wav"),
                }[node]
                real = os.lstat
                ino = target.stat().st_ino

                def wrong(*args, **kw):
                    info = real(*args, **kw)
                    if info.st_ino == ino:
                        return Mock(st_uid=os.getuid() + 1, st_mode=info.st_mode)
                    return info

                with patch("os.lstat", side_effect=wrong):
                    assert refs.load() == ()
                assert refs.notes == ["self_profile voices=0 unavailable reason=owner"]

    def test_relative_and_nonregular(self):
        refs = self.references()
        path = refs.path
        refs.path = Path("manifest")
        assert refs.load() == ()
        assert refs.notes[-1].endswith("reason=absolute_path")
        refs.path = path
        path.unlink()
        os.mkfifo(path, 0o600)
        assert refs.load() == ()
        assert refs.notes[-1].endswith("reason=file_type")

    def test_oversize_wav(self):
        refs = self.references()
        b = refs.clip(b"0" * (WAV_BYTES + 1), name="b")
        refs.write([refs.voice, b])
        assert not refs.load()
        assert refs.notes == ["self_profile voices=0 unavailable reason=size"]

    def test_one_bad_clip(self):
        for kind in [
            "digest",
            "container",
            "channels",
            "width",
            "rate",
            "short",
            "long",
            "truncated",
            "bits",
            "compression",
        ]:
            with self.subTest(kind=kind):
                refs = self.references()
                kwargs = {
                    "channels": {"channels": 2},
                    "width": {"width": 1},
                    "rate": {"rate": 8000},
                    "short": {"seconds": 7},
                    "long": {"seconds": 21},
                }
                data = bytearray(wav(**kwargs.get(kind, {})))
                if kind == "bits":
                    data[34:36] = (12).to_bytes(2, "little")
                if kind == "compression":
                    data[20:22] = (3).to_bytes(2, "little")
                if kind == "truncated":
                    del data[-2:]
                if kind == "container":
                    data[:4] = b"RIFX"
                b = refs.clip(data, name="b")
                if kind == "digest":
                    b["sha256"] = "0" * 64
                refs.write([b, refs.voice])
                assert len(refs.load()) == 1
                assert refs.notes == [
                    f"self_profile skipped reason={'digest' if kind == 'digest' else 'wav'}",
                    "self_profile voices=1 ready reason=loaded",
                ]

    def test_cache_lifecycle(self):
        refs = self.references()
        # Exercise the enrollment default without accessing an installed store.
        from dataclasses import replace

        default = refs.config.store
        self.enterContext(patch("rightyo.enroll.DEFAULT_STORE", default))
        refs.config = replace(refs.config, store=None)
        Store().create()
        path = default / CACHE_NAME
        model = refs.model
        b = refs.clip(wav(seconds=20), name="b")
        refs.write([refs.voice, b])
        with patch("os.replace", wraps=os.replace) as rename:
            assert len(refs.load()) == 2
            assert rename.call_count == 1
            src, dst = rename.call_args.args
            assert src != dst == CACHE_NAME
            assert rename.call_args.kwargs["src_dir_fd"] == rename.call_args.kwargs["dst_dir_fd"]
        assert path.stat().st_mode & 0o777 == 0o600
        assert Store(default).entries() == ([], 0)
        calls = model.calls
        assert len(refs.load()) == 2 and model.calls == calls
        assert refs.open_model.call_count == 1
        refs.write([b])
        assert len(refs.load()) == 1 and model.calls == calls
        assert refs.open_model.call_count == 1
        for field, value in [
            ("id", "B-1"),
            ("rendered_at", 1),
            ("sha256", None),
            ("model", "b" * 64),
        ]:
            if field == "model":
                model.model["sha256"] = value
            elif field == "sha256":
                b[field] = refs.clip(wav(9), "b")[field]
            else:
                b[field] = value
            refs.write([b])
            assert len(refs.load()) == 1 and model.calls > calls
            calls = model.calls
            key = f"{model.model['sha256']}:{b['id']}:{b['rendered_at']}:{b['sha256']}"
            assert json.loads(path.read_bytes()) == {key: unit(1, 0, 0)}
        for bad in (
            b"{",
            b"[]",
            json.dumps({key: [True]}).encode(),
            json.dumps({key: [2]}).encode(),
        ):
            path.write_bytes(bad)
            assert len(refs.load()) == 1 and model.calls > calls
            calls = model.calls
            assert json.loads(path.read_bytes()) == {key: unit(1, 0, 0)}
        refs.write([])
        assert refs.load() == ()
        assert refs.notes == ["self_profile voices=0 unavailable reason=empty"]
        assert json.loads(path.read_bytes()) == {}

    def test_cache_requires_existing_marked_store(self):
        for kind in ("missing", "empty", "unmarked", "symlink_marker"):
            with self.subTest(kind=kind):
                refs = self.references()
                path = refs.config.store
                if kind != "missing":
                    path.mkdir(mode=0o700)
                if kind == "unmarked":
                    refs.cache.write_bytes(b"existing")
                if kind == "symlink_marker":
                    (path / STORE_MARKER).symlink_to(refs.path)
                before = set(path.iterdir()) if path.exists() else None
                assert refs.load()
                calls = refs.model.calls
                assert refs.load() and refs.model.calls > calls
                assert (set(path.iterdir()) if path.exists() else None) == before
                if kind == "unmarked":
                    assert refs.cache.read_bytes() == b"existing"

    def test_delete_all_removes_cached_references(self):
        refs = self.references()
        store = Store(refs.config.store)
        store.create()
        assert refs.load() and refs.cache.is_file()
        (store.path / ".self-voice-tmp-stale").touch()
        assert store.delete_all() == []
        assert not store.path.exists()

    def test_delete_all_waits_for_cache_rename(self):
        refs = self.references()
        store = Store(refs.config.store)
        store.create()
        paused, resume, attempted, blocked = (threading.Event() for _ in range(4))
        rename, flock = os.replace, fcntl.flock

        def hold_rename(*args, **kwargs):
            paused.set()
            assert resume.wait(5)
            return rename(*args, **kwargs)

        def observe_delete_lock(fd, operation):
            if operation == fcntl.LOCK_EX:
                try:
                    flock(fd, operation | fcntl.LOCK_NB)
                except BlockingIOError:
                    blocked.set()
                finally:
                    attempted.set()
            return flock(fd, operation)

        with (
            patch("os.replace", side_effect=hold_rename),
            patch("fcntl.flock", side_effect=observe_delete_lock),
            ThreadPoolExecutor(2) as pool,
        ):
            writer = pool.submit(refs.load)
            try:
                assert paused.wait(3)
                assert len(list(store.path.glob(".self-voice-tmp-*"))) == 1
                deletion = pool.submit(store.delete_all)
                assert attempted.wait(3)
                assert blocked.is_set() and not deletion.done()
            finally:
                resume.set()
            assert writer.result(timeout=3)
            assert deletion.result(timeout=3) == []
        assert not store.path.exists()

    def test_delete_all_before_cache_write(self):
        for keep_directory in (False, True):
            with self.subTest(keep_directory=keep_directory):
                refs = self.references()
                store = Store(refs.config.store)
                store.create()
                if keep_directory:
                    (store.path / "unrelated").touch()
                paused, resume = threading.Event(), threading.Event()

                def hold_factory():
                    paused.set()
                    assert resume.wait(5)
                    return closing(refs.model)

                refs.open_model.side_effect = hold_factory
                with ThreadPoolExecutor(1) as pool:
                    writer = pool.submit(refs.load)
                    try:
                        assert paused.wait(3)
                        assert store.delete_all() == []
                    finally:
                        resume.set()
                    assert writer.result(timeout=3)
                if keep_directory:
                    assert {p.name for p in store.path.iterdir()} == {STORE_MARKER, "unrelated"}
                else:
                    assert not store.path.exists()

    def test_missing_model_sha_computes_without_cache(self):
        refs = self.references()
        Store(refs.config.store).create()
        with patch("rightyo.self_profile._cache") as cache:
            for _ in range(2):
                person, reference = ScriptedEmbedder(), ScriptedEmbedder()
                del person.model["sha256"], reference.model["sha256"]
                worker = ShadowSpeakerId(
                    refs.config,
                    self_voice_manifest=refs.path,
                    entries=lambda _: [{**ENTRIES[0], "model": person.model}],
                    embedder_factory=Mock(side_effect=[person, reference]),
                )
                worker.start()
                try:
                    wait_for(lambda: bool(worker._self_prints))
                    assert reference.calls > 0 and reference.closed
                    worker.audio(PCM)
                    worker.turn(labelled(0, 1000), spans=[(0, 1000)], wait=True)
                    assert worker.scored == 1 and worker.active
                finally:
                    worker.close()
            cache.assert_not_called()
        assert not refs.cache.exists()

    def test_unreadable_cache_is_replaced(self):
        refs = self.references()
        Store(refs.config.store).create()
        separate = refs.path.parent / "separate"
        separate.write_bytes(b"invalid")
        refs.cache.symlink_to(separate)
        assert len(refs.load()) == 1
        assert not refs.cache.is_symlink()
        assert refs.cache.stat().st_mode & 0o777 == 0o600
        assert separate.read_bytes() == b"invalid"

    def test_atomic_cache_replacement_failure(self):
        refs = self.references()
        Store(refs.config.store).create()
        assert refs.load()
        original = refs.cache.read_bytes()
        names = sorted(refs.config.store.iterdir())
        refs.write([{**refs.voice, "rendered_at": 1}])
        with patch("os.replace", side_effect=OSError):
            assert not refs.load()
        assert refs.notes == ["self_profile voices=0 unavailable reason=cache_write"]
        assert refs.cache.read_bytes() == original
        assert sorted(refs.config.store.iterdir()) == names
        refs.cache.chmod(0o400)
        refs.write([refs.voice])
        calls = refs.model.calls
        assert refs.load() and refs.model.calls > calls
        assert refs.cache.stat().st_mode & 0o777 == 0o600

    def test_pcm_extensible_and_multiple_data_chunks(self):
        refs = self.references()
        classic = wav()
        extension = bytes.fromhex("16001000040000000100000000001000800000aa00389b71")
        extended = (
            classic[:4]
            + (len(classic) - 8 + len(extension)).to_bytes(4, "little")
            + classic[8:16]
            + (40).to_bytes(4, "little")
            + b"\xfe\xff"
            + classic[22:36]
            + extension
            + classic[36:]
        )
        refs.write([refs.clip(extended)])
        assert len(refs.load()) == 1
        duplicate = (
            classic[:4]
            + (len(classic) - 8 + len(classic[36:])).to_bytes(4, "little")
            + classic[8:]
            + classic[36:]
        )
        refs.write([refs.clip(duplicate)])
        assert not refs.load() and refs.notes[-1].endswith("reason=wav")

    def test_reference_uses_enrollment_windows(self):
        refs = self.references()
        from array import array

        samples = array("h")
        samples.frombytes(b"\x00\x00" * SAMPLE_RATE + PCM * 8)
        with patch.object(refs.model, "embed", wraps=refs.model.embed) as embed:
            refs.write([refs.clip(wav(9))])
            data = refs.path.with_name("a.wav")
            encoded = data.read_bytes()[: -len(samples) * 2] + samples.tobytes()
            refs.write([refs.clip(encoded)])
            actual = refs.load()
            calls = embed.call_args_list
            embed.reset_mock()
            expected, count = voiceprint(embed, speech(samples))
            assert actual == (expected,)
            assert len(calls) == count and calls == embed.call_args_list

    def test_self_binding_and_renewal(self):
        shadow = make_shadow()
        score(shadow, unit(0, 0, 1))
        assert not shadow.self_bound("label")
        score(shadow, unit(0, 1, 0))
        assert shadow.self_bound("label")
        assert shadow.binding("label") == ("unknown", "owner")
        assert shadow._labels["label"].scores == {"owner": 0}
        assert shadow._labels["label"].self_total == 2
        assert shadow._bound_ids == set()
        roles = EnrolledRoles(SpeakerPriority())
        roles.source = shadow
        assert roles.role_for(labelled(0, 1000, "label")) == "participant"
        assert not roles.enrolled_owner(labelled(0, 1000, "label"))
        note = shadow._report_to.call_args.args[0]
        assert "state=self" in note and "id=" not in note
        shadow.forget()
        assert not shadow.self_bound("label") and shadow.binding("label") is None

    def test_outside_window_adds_no_self_evidence(self):
        shadow = make_shadow()
        score(shadow, unit(0, 1, 0), inside=False, seconds=2)
        assert not shadow.self_bound("label")
        assert shadow._labels["label"].self_seconds == 0
        score(shadow, unit(0, 1, 0))
        assert not shadow.self_bound("label")

    def test_owner_protection(self):
        shadow = make_shadow()
        v = unit(0.7, 0.714, 0)
        shadow._self_prints = (v,)
        score(shadow, v, seconds=2)
        assert shadow.binding("label") == ("bound", "owner")
        assert shadow._labels["label"].self_total > 1.99
        assert not shadow.self_bound("label")

    def test_margin(self):
        shadow = make_shadow()
        # Person below bind, self above bind, with less than 0.10 between them.
        v = unit(0.56, 0.63, (1 - 0.56**2 - 0.63**2) ** 0.5)
        shadow._self_prints = (unit(0, 1, 0),)
        score(shadow, v, seconds=2)
        assert shadow.binding("label")[0] == "tentative"
        assert not shadow.self_bound("label")

    def test_hysteresis(self):
        shadow = make_shadow()
        shadow._self_prints = (unit(0, 1, 0),)
        score(shadow, unit(0, 1, 0), seconds=2)
        assert shadow.self_bound("label")
        score(shadow, unit(0, 0, 1), seconds=2)
        assert shadow.self_bound("label")
        score(shadow, unit(0, 0, 1))
        assert not shadow.self_bound("label")

    def test_listen_progress_during_reference_embedding(self):
        refs = self.references()
        started, release, transcript = (threading.Event() for _ in range(3))
        embed = refs.model.embed
        workers = []

        def delayed(pcm):
            assert threading.current_thread() is not workers[0]._thread
            started.set()
            assert release.wait(10)
            return embed(pcm)

        refs.model.embed = delayed

        def worker(config, **kwargs):
            result = ShadowSpeakerId(
                config,
                **kwargs,
                embedder_factory=Mock(side_effect=[ScriptedEmbedder(), refs.model]),
                entries=lambda _: ENTRIES,
            )
            workers.append(result)
            return result

        def controller(config, **kwargs):
            return PrototypeController(
                config, **kwargs, processor_factory=TurnEveryTwoSeconds, speaker_id_factory=worker
            )

        class Output(io.StringIO):
            def write(self, data):
                if json.loads(data)["type"] == "transcript":
                    transcript.set()
                return super().write(data)

        read, write = os.pipe()
        output = Output()
        with (
            os.fdopen(read, "rb") as stream,
            os.fdopen(write, "wb", buffering=0) as source,
            ThreadPoolExecutor(1) as pool,
        ):
            future = pool.submit(
                listen,
                listen_args(refs),
                output=output,
                controller_factory=controller,
                audio_input=stream,
            )
            try:
                assert started.wait(5)
                source.write(PCM * 2)
                assert transcript.wait(5)
                assert not workers[0].self_bound("Speaker A")
                assert not workers[0]._self_prints
                release.set()
                wait_for(lambda: bool(workers[0]._self_prints))
            finally:
                release.set()
                source.close()
            assert future.result(timeout=10) == 0
        assert [e["type"] for e in map(json.loads, output.getvalue().splitlines())][:2] == [
            "session",
            "transcript",
        ]
        workers[0]._thread.join(5)
        assert len(workers[0]._self_prints) == 1

    def test_blocked_reference_preserves_public_and_privacy_scoring(self):
        def sequence(profile):
            refs = self.references()
            started, release = threading.Event(), threading.Event()

            def hold_reference(pcm):
                started.set()
                if not release.wait(15):
                    raise RuntimeError
                return unit(0, 1, 0)

            refs.model.embed = hold_reference
            person = ScriptedEmbedder()
            worker = ShadowSpeakerId(
                refs.config,
                self_voice_manifest=refs.path if profile else None,
                embedder_factory=Mock(side_effect=[person, refs.model]),
                entries=lambda _: ENTRIES,
            )
            worker.start()
            try:
                if profile:
                    assert started.wait(3)
                    assert not worker.self_bound("Speaker A")
                worker.audio(PCM * 5)
                for i in range(5):
                    span = (1000 * i, 1000 * (i + 1))
                    worker.turn(labelled(*span), spans=[span], in_window=True)
                    wait_for(lambda: worker.scored == i + 1)
                entry = worker.entry_score(0, 1000, [(0, 1000)], ["owner"])
                pcm = bytearray(PCM)
                exit_score = worker.exit_score(pcm, ["owner"])
                assert not any(pcm) and entry.outcome == exit_score.outcome == "scored"
                assert not worker._self_prints and not worker.dropped
                return worker.binding("Speaker A"), worker.scored, person.calls, entry, exit_score
            finally:
                release.set()
                worker.close()

        assert sequence(True) == sequence(False)

    def test_reference_error_preserves_next_public_score(self):
        refs = self.references()
        Store(refs.config.store).create()
        assert refs.load()
        refs.write([refs.voice, {**refs.voice, "id": "b"}])

        class Embedder(ScriptedEmbedder):
            def embed(self, pcm):
                if len(pcm) > len(PCM):
                    self.close()
                    raise ValueError
                assert not self.closed
                return super().embed(pcm)

        person, reference = Embedder(), Embedder()
        worker = ShadowSpeakerId(
            refs.config,
            self_voice_manifest=refs.path,
            entries=lambda _: ENTRIES,
            embedder_factory=Mock(side_effect=[person, reference]),
            report=refs.notes.append,
        )
        worker.start()
        try:
            wait_for(lambda: "self_profile voices=0 unavailable reason=embedding" in refs.notes)
            worker.audio(PCM)
            worker.turn(labelled(0, 1000), spans=[(0, 1000)], wait=True)
            assert worker.scored == 1 and worker.active
            assert reference.closed and not person.closed and not worker._self_prints
        finally:
            worker.close()

    def test_stop_interrupts_reference_loading(self):
        refs = self.references()
        refs.write([refs.voice, {**refs.voice, "id": "b"}])
        started, release = threading.Event(), threading.Event()

        def block(pcm):
            started.set()
            assert release.wait(10)
            return unit(1, 0, 0)

        refs.model.embed = Mock(side_effect=block)
        refs.model.close = release.set
        worker = ShadowSpeakerId(
            refs.config,
            self_voice_manifest=refs.path,
            entries=lambda _: ENTRIES,
            embedder_factory=Mock(side_effect=[ScriptedEmbedder(), refs.model]),
        )
        worker.start()
        try:
            assert started.wait(3)
            before = time.monotonic()
            worker.close()
            assert time.monotonic() - before < 2
            assert not worker._profile_thread.is_alive()
            assert release.is_set() and not worker._self_prints
            refs.model.embed.assert_called_once()
        finally:
            release.set()
            worker.close()

    def test_close_reaps_reference_child_during_handshake(self):
        refs = self.references()
        runtime = refs.path.parent / "worker"
        runtime.write_text('#!/bin/sh\ntrap "" TERM\n: > "$0.ready"\nread -r line\n')
        runtime.chmod(0o700)
        children = []
        popen = subprocess.Popen

        def spawn(*args, **kwargs):
            process = popen(*args, **kwargs)
            children.append(process)
            return process

        def factory(settings):
            if threading.current_thread() is worker._thread:
                return refs.model
            return SpeakerEmbedder(runtime, refs.path, identity=refs.model.model)

        worker = ShadowSpeakerId(
            refs.config,
            self_voice_manifest=refs.path,
            entries=lambda _: ENTRIES,
            embedder_factory=factory,
        )
        with patch("subprocess.Popen", side_effect=spawn):
            worker.start()
            try:
                wait_for((refs.path.parent / "worker.ready").exists)
                (child,) = children
                assert child.poll() is None
                loader = worker._profile_thread
                start = time.monotonic()
                worker.close()
                assert not loader.is_alive()
                assert child.poll() == -signal.SIGKILL
                assert time.monotonic() - start < 5
            finally:
                for child in children:
                    if child.poll() is None:
                        child.kill()
                        child.wait(timeout=2)
                worker.close()

    def test_self_only_privacy_checks_skip_embedding(self):
        refs = self.references()
        person = ScriptedEmbedder()
        person.embed_private = Mock(wraps=person.embed)
        worker = ShadowSpeakerId(
            refs.config,
            self_voice_manifest=refs.path,
            entries=lambda _: [],
            embedder_factory=Mock(side_effect=[person, refs.model]),
        )
        worker.start()
        try:
            wait_for(lambda: bool(worker._self_prints))
            worker.audio(PCM)
            assert worker.entry_score(0, 1000, [(0, 1000)], ["owner"]).outcome == "unavailable"
            pcm = bytearray(PCM)
            assert worker.exit_score(pcm, ["owner"]).outcome == "unavailable"
            assert not any(pcm)
            person.embed_private.assert_not_called()
            assert person.calls == 0
        finally:
            worker.close()

    def test_reference_without_person_enrollment(self):
        refs = self.references()
        worker = ShadowSpeakerId(
            refs.config,
            self_voice_manifest=refs.path,
            embedder_factory=Mock(side_effect=[ScriptedEmbedder(), refs.model]),
            entries=lambda _: [],
            report=refs.notes.append,
        )
        worker.start()
        try:
            wait_for(lambda: bool(worker._self_prints))
            worker.audio(PCM * 3)
            worker.turn(labelled(0, 3000), spans=[(0, 3000)], wait=True)
            assert worker.offered == 0
            snapshot = worker.binding_snapshot("Speaker A")
            worker.turn(labelled(0, 3000), spans=[(0, 3000)], in_window=True, wait=True)
            assert worker.binding_snapshot("Speaker A") == snapshot
            assert worker.self_bound("Speaker A")
            assert worker.binding("Speaker A") is None
            assert worker.owner_evidence(["owner"]) == (False, False, 0)
        finally:
            worker.close(drain=True)
        assert refs.notes.count("self_profile voices=1 ready reason=loaded") == 1
        assert refs.notes[-1].endswith(" self_bound=1")
        final = next(line for line in refs.notes if line.startswith("speaker_id final"))
        assert "state=self" in final and "id=" not in final

    def test_ignored_flag_bytes(self):
        # SHA-256 of the authored stdout and stderr at 627a469 with no manifest flag.
        main = {
            (False, True): (
                "90708a72e98158ac2eb8cabceed23b5bae78c729f04b40c32e9d5890a25321b7",
                "d9b1918d0b90c8d7894bf9a433fd7c52c2706df12f70d97319344d54b12e3539",
            ),
            (True, False): (
                "03ab0621e2ffdf5e883d0417ff9c5d57e1aae48cb22fb108028bf87b759fa87b",
                "c5cace748e1a2b9a8a2d1698a005435df959a026b18abe73b6288290ec138ab9",
            ),
            (True, True): (
                "03ab0621e2ffdf5e883d0417ff9c5d57e1aae48cb22fb108028bf87b759fa87b",
                "568c9f5f50a6dc1d49eeb9849af4852b086deae67c05842906da6938307908cd",
            ),
        }
        for (enabled, live), expected in main.items():
            with self.subTest(enabled=enabled, live=live):
                refs = self.references()
                args = listen_args(refs, enabled=enabled, live=live)

                args.self_voice_manifest = None
                baseline, diagnostics = session_bytes(args)
                assert [e["type"] for e in map(json.loads, baseline.splitlines())] == [
                    "session",
                    "transcript",
                    "session",
                ]
                assert (
                    tuple(
                        hashlib.sha256(value.encode()).hexdigest()
                        for value in (baseline, diagnostics)
                    )
                    == expected
                )
                if enabled and live:
                    continue
                args.self_voice_manifest = refs.path
                assert session_bytes(args) == (
                    baseline,
                    "rightyo: self_profile voices=0 unavailable reason=disabled\n" + diagnostics,
                )
                assert not refs.cache.exists()
