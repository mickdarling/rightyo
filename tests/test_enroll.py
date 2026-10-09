"""Local speaker enrollment, `rightyo enroll` (#137 step 3; privacy rules from #109).

Synthetic data only: every signal is an authored harmonic tone generated in the test and
written to a temporary directory outside the repository, and embeddings come from a
stand-in embedder or a stand-in worker script. No model, microphone, recorded voice or
real embedding is used, and nothing here is committed as audio.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import io
import json
import math
import os
import stat
import tempfile
import unittest
import wave
from array import array
from pathlib import Path

from rightyo import cli
from rightyo.enroll import (
    MIN_ENROLL_SECONDS,
    EnrollError,
    Store,
    read_wav,
    speech,
    voiceprint,
    windows,
)
from rightyo.enroll import run as run_enroll
from rightyo.prototype import PrototypeConfig, PrototypeError
from rightyo.speaker_id import (
    KNOWN_MODELS,
    SpeakerEmbedder,
    SpeakerIdConfig,
    SpeakerIdError,
    model_identity,
)

try:
    import numpy
except ImportError:  # the worker's features need numpy; RightyO itself does not
    numpy = None

RATE = 16000
FAKE_MODEL = {"id": "synthetic/stand-in", "revision": "0" * 40, "sha256": "a" * 64}
DIM = 32


@functools.cache
def _tone(f0: float, seconds: float, amplitude: float) -> bytes:
    """An authored 'voice': three harmonics of f0 with a 4 Hz syllable-like envelope."""
    samples = array("h")
    for n in range(int(seconds * RATE)):
        t = n / RATE
        envelope = 0.6 + 0.4 * math.sin(2 * math.pi * 4 * t)
        value = sum(math.sin(2 * math.pi * f0 * k * t) / k for k in (1, 2, 3))
        samples.append(int(32767 * amplitude * envelope * value / 1.84))
    return samples.tobytes()


def tone(f0: float, seconds: float, *, amplitude: float = 0.3) -> array:
    samples = array("h")
    samples.frombytes(_tone(f0, seconds, amplitude))
    return samples


def silence(seconds: float) -> array:
    return array("h", bytes(int(seconds * RATE) * 2))


def write_wav(path: Path, samples: array, *, rate: int = RATE, channels: int = 1) -> Path:
    with wave.open(str(path), "wb") as target:
        target.setnchannels(channels)
        target.setsampwidth(2)
        target.setframerate(rate)
        target.writeframes(samples.tobytes())
    return path


def fake_vector(pcm: bytes) -> list[float]:
    """A deterministic stand-in embedding: a bump at the clip's zero-crossing rate."""
    samples = array("h")
    samples.frombytes(pcm)
    crossings = sum(1 for a, b in zip(samples, samples[1:]) if (a < 0) != (b < 0))
    rate = crossings / max(1, len(samples)) * RATE / 2  # roughly the fundamental, Hz
    centre = rate / 25.0
    return [math.exp(-((i - centre) ** 2) / 2.0) + 0.01 for i in range(DIM)]


class FakeEmbedder:
    instances: list[FakeEmbedder] = []

    def __init__(self, settings=None):
        self.model = dict(FAKE_MODEL)
        self.calls: list[int] = []
        self.closed = False
        FakeEmbedder.instances.append(self)

    def embed(self, pcm: bytes) -> list[float]:
        self.calls.append(len(pcm))
        return fake_vector(pcm)

    def close(self) -> None:
        self.closed = True


class FakeCapture:
    """Stands in for the microphone helper: yields 200 ms chunks of an authored tone."""

    def __init__(self, helper):
        self.helper = helper
        self.pcm = tone(150, 36).tobytes()
        self.started = self.stopped = False

    def start(self):
        self.started = True

    def read(self, timeout=0.25):
        chunk, self.pcm = self.pcm[:6400], self.pcm[6400:]
        return chunk or None

    def stop(self):
        self.stopped = True


class EnrollTestCase(unittest.TestCase):
    def setUp(self):
        FakeEmbedder.instances = []
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.store = self.root / "private" / "store"
        asset = self.root / "asset"
        asset.touch()
        self.asset = asset
        self.config = self.root / "config.json"
        self.write_config({"python": str(asset), "model": str(asset), "store": str(self.store)})

    def write_config(self, section):
        base = {
            name: str(self.asset)
            for name in (
                "whisper_executable",
                "whisper_model",
                "diarization_library",
                "diarization_model",
                "microphone_helper",
            )
        }
        if section is not None:
            base["speaker_id"] = section
        self.config.write_text(json.dumps(base))

    def wav(self, name: str, *parts: array) -> Path:
        samples = array("h")
        for part in parts:
            samples.extend(part)
        return write_wav(self.root / name, samples)

    def enroll(self, *argv: str, capture_factory=FakeCapture):
        parser = argparse.ArgumentParser()
        cli._add_enroll_parser(parser.add_subparsers(dest="command"))
        args = parser.parse_args(["enroll", *argv])
        output, errors = io.StringIO(), io.StringIO()
        code = run_enroll(
            args,
            output=output,
            errors=errors,
            embedder_factory=FakeEmbedder,
            capture_factory=capture_factory,
        )
        result = json.loads(output.getvalue()) if code == 0 else None
        return code, result, output.getvalue() + errors.getvalue(), errors.getvalue()

    def add(self, identifier, audio, *extra):
        return self.enroll(
            "add", f"--id={identifier}", "--from", str(audio), "--config", str(self.config), *extra
        )


class AddListDeleteTests(EnrollTestCase):
    def test_add_list_and_delete(self):
        low = self.wav("low.wav", tone(120, 35))
        high = self.wav("high.wav", tone(260, 35))
        code, result, _, errors = self.add("alex", low, "--name", "Alex Example")
        self.assertEqual(code, 0, errors)
        self.assertEqual(result["enrolled"], "alex")
        self.assertEqual(result["display_name"], "Alex Example")
        self.assertAlmostEqual(result["speech_seconds"], 35, delta=0.1)
        self.assertEqual(self.add("sam", high)[0], 0)

        code, listing, _, _ = self.enroll("list", "--config", str(self.config))
        self.assertEqual(code, 0)
        self.assertEqual([item["id"] for item in listing["enrolled"]], ["alex", "sam"])
        self.assertEqual(listing["enrolled"][1]["display_name"], "sam")
        self.assertEqual(listing["unreadable_entries"], 0)

        code, result, _, _ = self.enroll("delete", "--id", "alex", "--store", str(self.store))
        self.assertEqual((code, result), (0, {"deleted": ["alex"]}))
        code, _, output, _ = self.enroll("delete", "--id", "alex", "--store", str(self.store))
        self.assertEqual(code, 2)
        self.assertIn("No voiceprint", output)

        code, result, _, _ = self.enroll("delete", "--all", "--store", str(self.store))
        self.assertEqual((code, result), (0, {"deleted": ["sam"]}))
        self.assertFalse(self.store.exists())
        self.assertEqual(
            self.enroll("delete", "--all", "--store", str(self.store))[1], {"deleted": []}
        )

    def test_an_existing_identifier_needs_replace(self):
        audio = self.wav("a.wav", tone(120, 32))
        self.assertEqual(self.add("alex", audio)[0], 0)
        code, _, output, _ = self.add("alex", audio)
        self.assertEqual(code, 2)
        self.assertIn("--replace", output)
        self.assertEqual(self.add("alex", audio, "--replace", "--name", "Alex")[0], 0)
        stored = json.loads((self.store / "alex.json").read_text())
        self.assertEqual(stored["display_name"], "Alex")

    def test_identifiers_and_display_names_are_validated(self):
        audio = self.wav("a.wav", tone(120, 32))
        for identifier in ("Alex", "../alex", "", "-x", "a" * 33, "al ex"):
            self.assertEqual(self.add(identifier, audio)[0], 2, identifier)
        for name in ("", " padded", "x" * 65, "line\nbreak"):
            self.assertEqual(self.add("alex", audio, f"--name={name}")[0], 2, name)
        self.assertFalse(self.store.exists())

    def test_list_and_delete_need_no_configuration(self):
        code, result, _, _ = self.enroll("list", "--store", str(self.store))
        self.assertEqual((code, result["enrolled"]), (0, []))
        self.assertFalse(self.store.exists())

    def test_the_cli_entry_point_routes_enroll(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = cli.main(["enroll", "list", "--store", str(self.store)])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout.getvalue())["enrolled"], [])


class DurationTests(EnrollTestCase):
    def test_less_than_the_minimum_speech_is_refused(self):
        audio = self.wav("short.wav", silence(10), tone(120, 15), silence(10))
        code, _, output, _ = self.add("alex", audio)
        self.assertEqual(code, 2)
        self.assertIn(f"at least {MIN_ENROLL_SECONDS:.0f} s", output)
        self.assertFalse(self.store.exists())
        self.assertEqual(FakeEmbedder.instances, [])

    def test_twenty_to_thirty_seconds_warns(self):
        code, result, _, errors = self.add("alex", self.wav("mid.wav", tone(120, 25)))
        self.assertEqual(code, 0)
        self.assertIn("warning", errors)
        self.assertAlmostEqual(result["speech_seconds"], 25, delta=0.1)

    def test_silence_does_not_count_as_speech(self):
        audio = self.wav(
            "gaps.wav", silence(3), tone(120, 12), silence(4), tone(120, 12), silence(5)
        )
        code, result, _, _ = self.add("alex", audio)
        self.assertEqual(code, 0)
        # 24 s of tone plus one pause shortened to 0.25 s; edges trimmed.
        self.assertAlmostEqual(result["speech_seconds"], 24.25, delta=0.05)

    def test_long_input_uses_at_most_ninety_seconds(self):
        code, result, _, _ = self.add("alex", self.wav("long.wav", tone(120, 100)))
        self.assertEqual(code, 0)
        self.assertEqual(result["speech_seconds"], 90.0)
        self.assertTrue(all(size <= 3 * RATE * 2 for size in FakeEmbedder.instances[0].calls))

    def test_only_sixteen_kilohertz_mono_pcm_is_accepted(self):
        stereo = write_wav(self.root / "stereo.wav", tone(120, 30), channels=2)
        fast = write_wav(self.root / "fast.wav", tone(120, 30), rate=44100)
        not_audio = self.root / "notes.wav"
        not_audio.write_text("not audio")
        for path in (stereo, fast, not_audio, self.root / "missing.wav"):
            code, _, output, _ = self.add("alex", path)
            self.assertEqual(code, 2)
            self.assertIn("16 kHz mono", output)
            self.assertNotIn(str(self.root), output)


class VoiceprintTests(unittest.TestCase):
    def test_windows_overlap_and_cover_the_end(self):
        samples = array("h", bytes(2 * int(10.2 * RATE)))
        parts = windows(samples)
        # Starts at 0, 1.5, 3, 4.5 and 6 s, then one aligned to the end at 7.2 s.
        self.assertEqual(len(parts), 6)
        self.assertTrue(all(len(part) == 3 * RATE for part in parts))
        self.assertEqual(len(windows(array("h", bytes(2 * RATE)))), 1)

    def test_voiceprint_is_the_renormalised_mean_of_normalised_windows(self):
        vectors = iter([[3.0, 0.0], [0.0, 0.5]])
        vector, count = voiceprint(lambda pcm: next(vectors), array("h", bytes(2 * 4 * RATE)))
        self.assertEqual(count, 2)
        self.assertAlmostEqual(vector[0], math.sqrt(0.5))
        self.assertAlmostEqual(vector[1], math.sqrt(0.5))
        with self.assertRaises(EnrollError):
            voiceprint(lambda pcm: [0.0, 0.0], array("h", bytes(2 * RATE)))

    def test_speech_trims_edges_and_long_pauses(self):
        samples = array("h")
        for part in (silence(1), tone(150, 2), silence(2), tone(150, 2), silence(1)):
            samples.extend(part)
        self.assertEqual(len(speech(samples)), int(4.25 * RATE))
        self.assertEqual(len(speech(silence(5))), 0)


class VerifyTests(EnrollTestCase):
    def test_scores_rank_the_same_synthetic_voice_first(self):
        self.assertEqual(self.add("low", self.wav("low.wav", tone(120, 35)))[0], 0)
        self.assertEqual(self.add("high", self.wav("high.wav", tone(300, 35)))[0], 0)
        probe = self.wav("probe.wav", tone(120, 4))
        code, result, _, errors = self.enroll(
            "verify", "--from", str(probe), "--config", str(self.config)
        )
        self.assertEqual(code, 0, errors)
        self.assertEqual([item["id"] for item in result["scores"]], ["low", "high"])
        self.assertEqual(result["scores"][0]["band"], "bind")
        self.assertEqual(result["scores"][1]["band"], "below")
        self.assertEqual(result["thresholds"], {"bind": 0.6, "tentative": 0.45})

    def test_voiceprints_from_another_model_are_not_scored(self):
        self.assertEqual(self.add("low", self.wav("low.wav", tone(120, 35)))[0], 0)
        stored = self.store / "low.json"
        entry = json.loads(stored.read_text())
        entry["model"]["sha256"] = "b" * 64
        stored.write_text(json.dumps(entry))
        probe = self.wav("probe.wav", tone(120, 4))
        code, result, _, _ = self.enroll(
            "verify", "--from", str(probe), "--config", str(self.config)
        )
        self.assertEqual(code, 0)
        self.assertEqual((result["scores"], result["other_model_entries"]), ([], 1))

    def test_too_little_speech_or_no_enrollment_is_refused(self):
        probe = self.wav("probe.wav", tone(120, 4))
        code, _, output, _ = self.enroll(
            "verify", "--from", str(probe), "--config", str(self.config)
        )
        self.assertEqual(code, 2)
        self.assertIn("No speakers", output)
        self.assertEqual(self.add("low", self.wav("low.wav", tone(120, 35)))[0], 0)
        short = self.wav("short.wav", silence(1), tone(120, 0.5), silence(1))
        code, _, output, _ = self.enroll(
            "verify", "--from", str(short), "--config", str(self.config)
        )
        self.assertEqual(code, 2)
        self.assertIn("at least 1.0 s", output)

    def test_recording_stays_in_memory_and_stops_the_microphone(self):
        captures = []

        def factory(helper):
            captures.append(FakeCapture(helper))
            return captures[-1]

        code, result, _, errors = self.enroll(
            "add",
            "--id",
            "alex",
            "--record",
            "35",
            "--config",
            str(self.config),
            capture_factory=factory,
        )
        self.assertEqual(code, 0, errors)
        self.assertTrue(captures[0].started and captures[0].stopped)
        self.assertEqual(captures[0].helper, self.asset)
        self.assertAlmostEqual(result["speech_seconds"], 35, delta=0.1)
        self.assertEqual(
            sorted(p.name for p in self.store.iterdir()), [".rightyo-voice-store", "alex.json"]
        )


class StorePrivacyTests(EnrollTestCase):
    def test_permissions_are_private_and_only_metadata_is_stored(self):
        self.assertEqual(self.add("alex", self.wav("a.wav", tone(120, 35)))[0], 0)
        self.assertEqual(stat.S_IMODE(self.store.stat().st_mode), 0o700)
        entry_path = self.store / "alex.json"
        self.assertEqual(stat.S_IMODE(entry_path.stat().st_mode), 0o600)
        self.assertEqual(
            sorted(p.name for p in self.store.iterdir()), [".rightyo-voice-store", "alex.json"]
        )
        entry = json.loads(entry_path.read_text())
        self.assertEqual(
            set(entry),
            {
                "schema_version",
                "id",
                "display_name",
                "model",
                "created_at",
                "speech_seconds",
                "windows",
                "embedding",
            },
        )
        self.assertEqual(entry["model"], FAKE_MODEL)
        self.assertEqual(len(entry["embedding"]), DIM)
        self.assertLess(entry_path.stat().st_size, 4096)  # a vector, never audio

    def test_loose_permissions_are_tightened(self):
        self.assertEqual(self.add("alex", self.wav("a.wav", tone(120, 35)))[0], 0)
        self.store.chmod(0o755)
        (self.store / "alex.json").chmod(0o644)
        self.assertEqual(self.enroll("list", "--store", str(self.store))[0], 0)
        self.assertEqual(stat.S_IMODE(self.store.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE((self.store / "alex.json").stat().st_mode), 0o600)

    def test_a_store_inside_a_git_checkout_is_refused(self):
        checkout = self.root / "checkout"
        (checkout / ".git").mkdir(parents=True)
        inside = checkout / "nested" / "store"
        self.write_config({"python": str(self.asset), "model": str(self.asset)})
        audio = self.wav("a.wav", tone(120, 35))
        for argv in (
            ("add", "--id", "alex", "--from", str(audio), "--config", str(self.config)),
            ("list",),
            ("delete", "--all"),
        ):
            code, _, output, _ = self.enroll(*argv, "--store", str(inside))
            self.assertEqual(code, 2)
            self.assertIn("inside a Git checkout", output)
            self.assertNotIn(str(checkout), output)
        self.assertFalse((checkout / "nested").exists())
        with self.assertRaises(EnrollError):
            Store(checkout)
        with self.assertRaises(EnrollError):
            Store(Path("relative/store"))

    def test_a_symlinked_or_foreign_store_is_refused(self):
        real = self.root / "real"
        real.mkdir()
        link = self.root / "link"
        link.symlink_to(real)
        code, _, output, _ = self.enroll("list", "--store", str(link))
        self.assertEqual(code, 2)
        self.assertIn("directory owned by you", output)

    def test_corrupt_entries_are_counted_not_shown(self):
        self.assertEqual(self.add("alex", self.wav("a.wav", tone(120, 35)))[0], 0)
        (self.store / "broken.json").write_text("{not json")
        (self.store / "Bad Name.json").write_text("{}")
        code, result, output, _ = self.enroll("list", "--store", str(self.store))
        self.assertEqual(code, 0)
        self.assertEqual([item["id"] for item in result["enrolled"]], ["alex"])
        self.assertEqual(result["unreadable_entries"], 2)
        self.assertNotIn("Bad Name", output)

    def test_a_non_finite_duration_is_counted_not_fatal(self):
        self.assertEqual(self.add("alex", self.wav("a.wav", tone(120, 35)))[0], 0)
        good = json.loads((self.store / "alex.json").read_text())
        for name, value in (("nan", "NaN"), ("inf", "Infinity")):
            seconds = json.dumps(good["speech_seconds"])
            text = json.dumps({**good, "id": name}).replace(
                f'"speech_seconds": {seconds}', f'"speech_seconds": {value}'
            )
            (self.store / f"{name}.json").write_text(text)
        code, result, _, _ = self.enroll("list", "--store", str(self.store))
        self.assertEqual(code, 0)
        self.assertEqual([item["id"] for item in result["enrolled"]], ["alex"])
        self.assertEqual(result["unreadable_entries"], 2)

    def test_delete_all_never_touches_files_that_are_not_voiceprints(self):
        # A mistaken --store (a home or project directory) must lose nothing unrelated.
        self.assertEqual(self.add("alex", self.wav("a.wav", tone(120, 35)))[0], 0)
        bystanders = {
            ".bashrc": "export PATH=x\n",
            ".env": "TOKEN=x\n",
            "package.json": '{"name": "x"}',
            "notes.json": "{not json",
        }
        for name, text in bystanders.items():
            (self.store / name).write_text(text)
        code, result, _, _ = self.enroll("delete", "--all", "--store", str(self.store))
        self.assertEqual(code, 0)
        self.assertEqual(result["deleted"], ["alex"])
        self.assertEqual(
            sorted(p.name for p in self.store.iterdir()),
            sorted([*bystanders, ".rightyo-voice-store"]),
        )

    def test_a_mistaken_store_keeps_its_permissions_and_is_not_taken_over(self):
        home = self.root / "home-like"
        home.mkdir(mode=0o755)
        bystander = home / "package.json"
        bystander.write_text('{"name": "x"}')
        bystander.chmod(0o644)
        for action in (("list",), ("delete", "--all"), ("delete", "--id", "package")):
            self.enroll(*action, "--store", str(home))
        self.assertEqual(stat.S_IMODE(home.stat().st_mode), 0o755)
        self.assertEqual(stat.S_IMODE(bystander.stat().st_mode), 0o644)
        code, _, _, _ = self.add("alex", self.wav("a.wav", tone(120, 35)), "--store", str(home))
        self.assertNotEqual(code, 0)
        self.assertEqual(sorted(p.name for p in home.iterdir()), ["package.json"])

    def test_only_a_strictly_shaped_voiceprint_counts_as_ours(self):
        self.assertEqual(self.add("alex", self.wav("a.wav", tone(120, 35)))[0], 0)
        junk = {"schema_version": True, "id": "junk", "model": 0, "embedding": 0}
        (self.store / "junk.json").write_text(json.dumps(junk))
        (self.store / "other.json").write_text(json.dumps({**junk, "schema_version": 1, "id": "x"}))
        code, result, _, _ = self.enroll("delete", "--all", "--store", str(self.store))
        self.assertEqual(code, 0)
        self.assertEqual(result["deleted"], ["alex"])
        self.assertTrue((self.store / "junk.json").exists())
        self.assertTrue((self.store / "other.json").exists())

    def test_delete_one_refuses_a_file_that_is_not_a_voiceprint(self):
        self.store.mkdir(mode=0o700, parents=True, exist_ok=True)
        (self.store / "package.json").write_text('{"name": "x"}')
        code, _, _, _ = self.enroll("delete", "--id", "package", "--store", str(self.store))
        self.assertNotEqual(code, 0)
        self.assertTrue((self.store / "package.json").exists())

    def test_an_overflowing_embedding_number_is_counted_not_fatal(self):
        self.assertEqual(self.add("alex", self.wav("a.wav", tone(120, 35)))[0], 0)
        good = json.loads((self.store / "alex.json").read_text())
        huge = json.dumps({**good, "id": "huge"}).replace(
            "[" + json.dumps(good["embedding"][0]), "[1" + "0" * 400, 1
        )
        (self.store / "huge.json").write_text(huge)
        code, result, _, _ = self.enroll("list", "--store", str(self.store))
        self.assertEqual(code, 0)
        self.assertEqual(result["unreadable_entries"], 1)

    def test_delete_removes_a_dangling_voiceprint_link(self):
        self.assertEqual(self.add("alex", self.wav("a.wav", tone(120, 35)))[0], 0)
        dangling = self.store / "ghost.json"
        dangling.symlink_to(self.root / "missing-target.json")
        code, _, _, _ = self.enroll("delete", "--id", "ghost", "--store", str(self.store))
        self.assertEqual(code, 0)
        self.assertFalse(os.path.lexists(dangling))

    def test_output_carries_no_audio_embeddings_or_paths(self):
        audio = self.wav("secret-name.wav", tone(120, 35))
        probe = self.wav("probe-name.wav", tone(120, 4))
        runs = [
            self.add("alex", audio, "--name", "Alex"),
            self.enroll("list", "--config", str(self.config)),
            self.enroll("verify", "--from", str(probe), "--config", str(self.config)),
            self.enroll("delete", "--id", "alex", "--config", str(self.config)),
        ]
        stored_values = {f"{value:.4f}" for value in fake_vector(tone(120, 3).tobytes())}
        for code, result, output, _ in runs:
            self.assertEqual(code, 0, output)
            self.assertNotIn(str(self.root), output)
            self.assertNotIn("secret-name", output)
            self.assertNotIn("probe-name", output)
            self.assertNotIn("embedding", output)
            self.assertLess(len(output), 1200)
            numbers = [token for token in output.replace(",", " ").split() if "." in token]
            self.assertLessEqual(len(numbers), 6)
            self.assertFalse(any(value in output for value in stored_values if value != "0.0100"))


class ConfigurationTests(EnrollTestCase):
    def test_the_section_is_off_unless_present_and_enabled(self):
        self.write_config(None)
        self.assertIsNone(PrototypeConfig.load(self.config).speaker_id)
        section = {"python": str(self.asset), "model": str(self.asset)}
        self.write_config(section | {"enabled": False})
        self.assertIsNone(PrototypeConfig.load(self.config).speaker_id)
        self.write_config(section)
        loaded = PrototypeConfig.load(self.config).speaker_id
        self.assertEqual(loaded, SpeakerIdConfig(self.asset, self.asset, None, 0.6, 0.45, 1.0, 4))
        self.write_config(section | {"bind_threshold": 0.7, "tentative_threshold": 0.5})
        self.assertEqual(PrototypeConfig.load(self.config).speaker_id.bind_threshold, 0.7)

    def test_invalid_sections_are_refused(self):
        section = {"python": str(self.asset), "model": str(self.asset)}
        for invalid in (
            section | {"model": "relative.onnx"},
            section | {"model": str(self.root / "missing.onnx")},
            section | {"store": "relative/store"},
            section | {"bind_threshold": 1.5},
            section | {"bind_threshold": 0.4},  # below the default tentative threshold
            section | {"tentative_threshold": 0.6},  # equal to the bind threshold
            section | {"tentative_threshold": 0},
            section | {"bind_threshold": "0.6"},
            section | {"min_turn_seconds": 0.1},
            section | {"threads": 0},
            section | {"enabled": "yes"},
            section | {"unknown": 1},
            {"model": str(self.asset)},
            [],
        ):
            self.write_config(invalid)
            with self.assertRaises(PrototypeError):
                PrototypeConfig.load(self.config)

    def test_bands_follow_the_thresholds(self):
        settings = SpeakerIdConfig(self.asset, self.asset)
        self.assertEqual(
            [settings.band(value) for value in (0.6, 0.59, 0.45, 0.44)],
            ["bind", "tentative", "tentative", "below"],
        )

    def test_add_and_verify_need_an_enabled_section(self):
        self.write_config(None)
        audio = self.wav("a.wav", tone(120, 35))
        code, _, output, _ = self.add("alex", audio, "--store", str(self.store))
        self.assertEqual(code, 2)
        self.assertIn("speaker_id", output)


class ModelAndWorkerTests(unittest.TestCase):
    """The worker client against a stand-in script that speaks the protocol (no model)."""

    def worker(self, body):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        script = Path(directory.name) / "python"
        script.write_text("#!/bin/sh\n" + body)
        script.chmod(script.stat().st_mode | stat.S_IXUSR)
        model = Path(directory.name) / "model.onnx"
        model.write_bytes(b"not a model")
        return script, model

    def test_only_the_documented_model_file_is_accepted(self):
        self.assertEqual(
            KNOWN_MODELS["7bb2f06e9df17cdf1ef14ee8a15ab08ed28e8d0ef5054ee135741560df2ec068"][
                "revision"
            ],
            "f0c48c298fd835726c27956a5d617bad7115627e",
        )
        python, model = self.worker("echo '{\"ok\":true}'\n")
        with self.assertRaises(SpeakerIdError) as error:
            model_identity(model)
        self.assertNotIn(str(model.parent), str(error.exception))
        with self.assertRaises(SpeakerIdError):
            SpeakerEmbedder(python, model)

    def test_embeds_through_the_worker(self):
        python, model = self.worker(
            "echo '{\"ok\":true}'\n"
            'while IFS= read -r line; do echo \'{"ok":true,"embedding":[0.5,-0.25,1]}\'; done\n'
        )
        client = SpeakerEmbedder(python, model, identity=FAKE_MODEL)
        self.addCleanup(client.close)
        self.assertEqual(client.embed(tone(120, 1).tobytes()), [0.5, -0.25, 1.0])
        self.assertEqual(client.model, FAKE_MODEL)
        with self.assertRaises(SpeakerIdError):
            client.embed(b"\x00" * (31 * RATE * 2))

    def test_failures_are_sanitized(self):
        for body in (
            "exit 0\n",
            "echo '{\"ok\":false}'\n",
            "echo '{\"ok\":true}'\nread -r line\nexit 0\n",
            'echo \'{"ok":true}\'\nread -r line\necho \'{"ok":true,"embedding":[]}\'\n',
            'echo \'{"ok":true}\'\nread -r line\necho \'{"ok":true,"embedding":["1"]}\'\n',
            'echo \'{"ok":true}\'\nread -r line\necho \'{"ok":true,"embedding":[NaN]}\'\n',
        ):
            python, model = self.worker(body)
            with self.assertRaises(SpeakerIdError) as error:
                client = SpeakerEmbedder(python, model, identity=FAKE_MODEL)
                self.addCleanup(client.close)
                client.embed(tone(120, 1).tobytes())
            self.assertNotIn(str(python.parent), str(error.exception))

    def test_explicit_existing_files_are_required(self):
        with self.assertRaises(SpeakerIdError):
            SpeakerEmbedder("/nonexistent/python", "/nonexistent/model.onnx", identity=FAKE_MODEL)


class WavTests(unittest.TestCase):
    def test_read_wav_round_trips_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            samples = tone(200, 0.5)
            path = write_wav(Path(directory) / "x.wav", samples)
            self.assertEqual(read_wav(path), samples)


@unittest.skipIf(numpy is None, "numpy is not installed")
class FeatureTests(unittest.TestCase):
    def test_fbank_has_the_reference_shape_and_mean_removal(self):
        from rightyo.speaker_id import _kaldi_fbank

        samples = numpy.random.default_rng(0).normal(0, 0.1, 2 * RATE).astype("float32")
        features = _kaldi_fbank(numpy, samples)
        self.assertEqual(features.shape, (198, 80))
        self.assertLess(float(numpy.abs(features.mean(axis=0)).max()), 1e-4)


if __name__ == "__main__":
    unittest.main()
