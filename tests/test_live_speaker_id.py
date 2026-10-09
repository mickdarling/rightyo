"""Shadow live speaker identification (#137 step 4a).

Synthetic data only: audio is an authored tone generated in the test, embeddings come from
stand-in embedders that return controlled vectors, and voiceprints are injected in memory.
No model, microphone, recorded voice or enrollment store is used.
"""

from __future__ import annotations

import io
import json
import math
import shlex
import tempfile
import threading
import time
import unittest
import wave
from argparse import Namespace
from array import array
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

from test_edge_attribution import SPEECH, Timeline, Units
from test_turn_merge import fragment

from rightyo.contracts import SpeakerPriority, Turn
from rightyo.live_audio import (
    FRAME_BYTES,
    LiveAudioError,
    LiveConfig,
    LiveProcessor,
    _labelled_spans,
)
from rightyo.live_speaker_id import EnrolledRoles, LabelState, PcmRing, ShadowSpeakerId, bind
from rightyo.prototype import PrototypeConfig, PrototypeController
from rightyo.speaker_id import SpeakerIdConfig
from rightyo.tool import listen
from rightyo.tool_events import SpeechEvents
from rightyo.turn_merge import TurnMerger

RATE = 16000
MODEL = {"id": "synthetic/stand-in", "revision": "0" * 40, "sha256": "a" * 64}
OTHER_MODEL = {"id": "synthetic/other", "revision": "1" * 40, "sha256": "b" * 64}
PRIVATE_TEXT = "private words that must never reach a log"


def tone(seconds: float, f0: float = 140.0) -> bytes:
    samples = array("h")
    for n in range(int(seconds * RATE)):
        t = n / RATE
        samples.append(int(9000 * math.sin(2 * math.pi * f0 * t)))
    return samples.tobytes()


def unit(*values: float) -> list[float]:
    norm = math.sqrt(sum(v * v for v in values))
    return [v / norm for v in values]


def turn(
    start_ms, end_ms, speaker="Speaker A", *, provenance="diarization-timeline", overlap=False
):
    return Turn(
        "shadow-test",
        f"live-{start_ms}",
        1,
        start_ms,
        end_ms,
        PRIVATE_TEXT,
        speaker,
        True,
        overlap,
        "fake-local-recognizer",
        "causal-replay",
        provenance,
    )


class ScriptedEmbedder:
    """Returns the next scripted vector per call; can block or fail on demand."""

    def __init__(self, vectors=(), *, fail_on=None, gate=None):
        self.model = dict(MODEL)
        self.vectors = list(vectors)
        self.calls = 0
        self.fail_on = fail_on
        self.gate = gate
        self.closed = False

    def embed(self, pcm: bytes) -> list[float]:
        self.calls += 1
        if self.gate is not None:
            self.gate.wait(5)
        if self.fail_on is not None and self.calls >= self.fail_on:
            raise RuntimeError("synthetic embedder failure /private/path")
        return self.vectors.pop(0) if self.vectors else unit(1, 0, 0)

    def close(self):
        self.closed = True


def settings(**overrides) -> SpeakerIdConfig:
    base = SpeakerIdConfig(Path("/synthetic/python"), Path("/synthetic/model.onnx"), live=True)
    return replace(base, **overrides)


ENTRIES = [
    {"id": "owner", "model": dict(MODEL), "embedding": unit(1, 0, 0)},
    {"id": "guest", "model": dict(MODEL), "embedding": unit(0, 1, 0)},
    # Another model's voiceprint is never compared, even if it would match.
    {"id": "stale", "model": dict(OTHER_MODEL), "embedding": unit(0, 0, 1)},
]


WHOLE = object()  # `Harness.speak`: the turn's whole span is labelled


class Harness:
    def __init__(self, embedder, *, config=None, entries=ENTRIES, queue_size=4, ring_ms=60000):
        self.lines: list[str] = []
        self.embedder = embedder
        self.shadow = ShadowSpeakerId(
            config or settings(),
            report=self.lines.append,
            embedder_factory=lambda _settings: self._make(),
            entries=lambda _settings: [dict(entry) for entry in entries],
            queue_size=queue_size,
            ring_ms=ring_ms,
        )
        self.received_ms = 0

    def _make(self):
        if isinstance(self.embedder, Exception):
            raise self.embedder
        return self.embedder

    def speak(self, seconds, speaker="Speaker A", *, spans=WHOLE, inferred=False, **options):
        """Feed `seconds` of tone and offer it as one finalized turn.

        By default the whole span is the label's own timeline audio (#148).
        """
        start = self.received_ms
        pcm = tone(seconds)
        self.shadow.audio(pcm)
        self.received_ms += len(pcm) // 32
        if spans is WHOLE:
            spans = ((start, self.received_ms),)
        self.shadow.turn(
            turn(start, self.received_ms, speaker, **options), inferred=inferred, spans=spans
        )

    def notes(self):
        return [line for line in self.lines if line.startswith("speaker_id label=")]

    @staticmethod
    def fields(line):
        """Every note is `key=value` pairs after its prefix, parseable with shlex."""
        words = shlex.split(line)
        if words[0] != "speaker_id":
            raise AssertionError("not a speaker_id note")
        return dict(word.split("=", 1) for word in words[1:] if "=" in word)


def wait_for(condition, timeout=3.0):
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.005)
    if not condition():
        raise AssertionError("condition not reached")


class RingTests(unittest.TestCase):
    def test_spans_are_addressed_by_stream_time_and_clipped_when_evicted(self):
        ring = PcmRing(max_ms=1000)
        for second in range(3):
            ring.append(bytes([second + 1]) * 32000)
        self.assertEqual(len(ring._buffer), 32000)  # bounded to one second
        pcm, clipped = ring.span(2000, 2500)
        self.assertEqual((len(pcm), clipped), (16000, False))
        self.assertEqual(set(pcm), {3})
        pcm, clipped = ring.span(1500, 2500)  # the first half was evicted
        self.assertEqual((len(pcm), clipped), (16000, True))
        self.assertEqual(ring.span(0, 1000), (b"", True))
        ring.clear()
        self.assertEqual(len(ring._buffer), 0)


class AccumulationTests(unittest.TestCase):
    def test_duration_weighted_mean_of_unit_vectors_is_renormalised(self):
        state = LabelState()
        state.add(unit(1, 0), 1.0)
        mean = state.add(unit(0, 1), 3.0)
        self.assertEqual(state.seconds, 4.0)
        self.assertEqual(state.turns, 2)
        for got, want in zip(mean, unit(1, 3)):
            self.assertAlmostEqual(got, want)

    def test_binding_needs_the_threshold_and_minimum_seconds(self):
        config = settings(bind_threshold=0.6, tentative_threshold=0.45, bind_min_seconds=3.0)
        state = LabelState(seconds=2.0)
        bind(state, {"owner": 0.9, "guest": 0.1}, config)
        self.assertEqual((state.state, state.identity), ("tentative", "owner"))
        state.seconds = 3.0
        bind(state, {"owner": 0.9, "guest": 0.1}, config)
        self.assertEqual((state.state, state.identity, state.score), ("bound", "owner", 0.9))
        fresh = LabelState(seconds=10.0)
        bind(fresh, {"owner": 0.5, "guest": 0.2}, config)
        self.assertEqual(fresh.state, "tentative")
        bind(fresh, {"owner": 0.3, "guest": 0.44}, config)
        self.assertEqual((fresh.state, fresh.identity), ("unknown", "guest"))

    def test_a_bound_label_stays_bound_until_below_the_tentative_threshold(self):
        config = settings(bind_threshold=0.6, tentative_threshold=0.45)
        state = LabelState(seconds=5.0)
        bind(state, {"owner": 0.7, "guest": 0.1}, config)
        self.assertEqual(state.state, "bound")
        # Between the thresholds, and even when another voice now scores higher.
        bind(state, {"owner": 0.5, "guest": 0.55}, config)
        self.assertEqual((state.state, state.identity, state.score), ("bound", "owner", 0.5))
        bind(state, {"owner": 0.44, "guest": 0.3}, config)
        self.assertEqual((state.state, state.identity), ("unknown", "owner"))
        bind(state, {"owner": 0.5, "guest": 0.3}, config)
        self.assertEqual(state.state, "tentative")  # rebinding needs the bind threshold


class ShadowTests(unittest.TestCase):
    def run_shadow(self, harness, *turns):
        harness.shadow.start()
        for seconds, speaker in turns:
            harness.speak(seconds, speaker)
        harness.shadow.close(drain=True)
        return harness.notes()

    def test_turns_accumulate_per_label_and_bind_after_enough_speech(self):
        embedder = ScriptedEmbedder([unit(1, 0.2, 0), unit(0, 1, 0), unit(1, 0.1, 0)])
        harness = Harness(embedder)
        notes = self.run_shadow(harness, (2.0, "Speaker A"), (2.0, "Speaker B"), (2.0, "Speaker A"))
        fields = [Harness.fields(line) for line in notes]
        self.assertEqual([f["label"] for f in fields], ["Speaker A", "Speaker B", "Speaker A"])
        # Two seconds is below the 3 s minimum: tentative, then bound at 4 s.
        self.assertEqual([f["state"] for f in fields], ["tentative", "tentative", "bound"])
        self.assertEqual([f["id"] for f in fields], ["owner", "guest", "owner"])
        self.assertEqual(fields[0]["turn_ms"], "2000")
        self.assertEqual(fields[2]["acc_s"], "4.0")
        expected = unit(*[a + b for a, b in zip(unit(1, 0.2, 0), unit(1, 0.1, 0))])[0]
        self.assertEqual(fields[2]["acc_score"], f"{expected:.3f}")
        self.assertEqual(fields[2]["turn_score"], f"{unit(1, 0.1, 0)[0]:.3f}")
        self.assertIn("speaker_id start enrolled=2", harness.lines)
        self.assertIn(
            'speaker_id final label="Speaker A" turns=2 acc_score='
            f"{expected:.3f} acc_s=4.0 state=bound id=owner",
            harness.lines,
        )
        self.assertTrue(embedder.closed)

    def test_short_overlapping_and_unlabelled_turns_are_not_scored(self):
        embedder = ScriptedEmbedder()
        harness = Harness(embedder)
        harness.shadow.start()
        harness.speak(0.6)  # below min_turn_seconds
        harness.speak(2.0, overlap=True)
        harness.speak(2.0, "u1 Speaker A", provenance="diarization-utterance")
        harness.speak(2.0, speaker=None, provenance="unknown")
        harness.shadow.close(drain=True)
        self.assertEqual(harness.notes(), [])
        self.assertEqual(embedder.calls, 0)
        self.assertIn(
            "speaker_id summary offered=2 scored=0 short=1 overlap=1 dropped=0 clipped=0"
            " inferred=0 unsegmented=0 labelled_ms=600",
            harness.lines,
        )

    def test_no_matching_voiceprint_switches_shadow_off(self):
        harness = Harness(ScriptedEmbedder(), entries=ENTRIES[2:])
        harness.shadow.start()
        wait_for(lambda: not harness.shadow.active)
        harness.speak(2.0)
        harness.shadow.close()
        self.assertIn(
            "speaker_id live: no voiceprint enrolled with this model; shadow off", harness.lines
        )
        self.assertEqual(harness.notes(), [])

    def test_worker_failure_is_reported_once_and_disables_cleanly(self):
        embedder = ScriptedEmbedder(fail_on=2)
        harness = Harness(embedder)
        harness.shadow.start()
        harness.speak(2.0)
        harness.speak(2.0)
        wait_for(lambda: not harness.shadow.active)
        harness.speak(2.0)  # ignored without error
        harness.shadow.close(drain=True)
        failures = [line for line in harness.lines if "unavailable" in line]
        self.assertEqual(
            failures, ["speaker_id unavailable; shadow identification off for this session"]
        )
        self.assertEqual(len(harness.notes()), 1)
        self.assertEqual(embedder.calls, 2)
        self.assertTrue(embedder.closed)
        self.assertNotIn("/private/path", "\n".join(harness.lines))

    def test_a_model_that_cannot_start_disables_cleanly(self):
        harness = Harness(RuntimeError("synthetic start failure"))
        harness.shadow.start()
        wait_for(lambda: not harness.shadow.active)
        harness.speak(2.0)
        harness.shadow.close()
        self.assertIn(
            "speaker_id unavailable; shadow identification off for this session", harness.lines
        )

    def test_backpressure_drops_and_counts_turns_without_blocking(self):
        gate = threading.Event()
        embedder = ScriptedEmbedder(gate=gate)
        harness = Harness(embedder, queue_size=1)
        harness.shadow.start()
        wait_for(lambda: any("speaker_id start" in line for line in harness.lines))
        started = time.monotonic()
        harness.speak(2.0)
        wait_for(lambda: embedder.calls == 1)  # the worker is now blocked in embed
        for _ in range(4):
            harness.speak(2.0)
        self.assertLess(time.monotonic() - started, 2.0)  # the audio side never waited
        self.assertEqual(harness.shadow.dropped, 3)
        gate.set()
        harness.shadow.close(drain=True)
        self.assertEqual(
            [line for line in harness.lines if "busy" in line],
            ["speaker_id worker busy; dropping turns (counted)"],
        )
        self.assertEqual(len(harness.notes()), 2)
        self.assertIn("dropped=3", harness.lines[-1])

    def test_turns_older_than_the_ring_are_counted_as_clipped(self):
        harness = Harness(ScriptedEmbedder(), ring_ms=800)
        harness.shadow.start()
        harness.speak(2.0)
        harness.shadow.close(drain=True)
        self.assertIn("clipped=1", harness.lines[-1])
        self.assertIn("short=1", harness.lines[-1])  # 0.8 s left: below the minimum

    def test_notes_carry_labels_identifiers_and_numbers_only(self):
        harness = Harness(ScriptedEmbedder([unit(1, 0.3, 0)] * 3))
        self.run_shadow(harness, (2.0, "Speaker A"), (2.0, "Speaker A"))
        log = "\n".join(harness.lines)
        self.assertNotIn(PRIVATE_TEXT, log)
        self.assertNotIn("/", log)
        self.assertNotIn("synthetic", log)  # neither model ids nor paths
        allowed = {
            "turn_ms",
            "labelled_ms",
            "speech_ms",
            "turn_score",
            "acc_score",
            "acc_s",
            "state",
            "id",
        }
        for line in harness.notes():
            self.assertIn(' label="Speaker A" ', line)
            fields = Harness.fields(line)
            self.assertEqual(set(fields) - {"label"}, allowed)
            for name in ("turn_ms", "labelled_ms", "speech_ms", "turn_score", "acc_score", "acc_s"):
                float(fields[name])

    def test_a_failure_mid_update_keeps_the_label_and_the_closing_lines(self):
        # Opposite unit vectors with equal weight leave a zero mean: normalising fails.
        embedder = ScriptedEmbedder([unit(1, 0, 0), unit(-1, 0, 0)])
        harness = Harness(embedder)
        harness.shadow.start()
        harness.speak(2.0)
        harness.speak(2.0)
        wait_for(lambda: not harness.shadow.active)
        harness.shadow.close(drain=True)
        state = harness.shadow._labels["Speaker A"]
        self.assertEqual((state.turns, state.seconds), (1, 2.0))  # the failed turn left no trace
        self.assertTrue(harness.lines[-2].startswith('speaker_id final label="Speaker A" turns=1'))
        self.assertTrue(harness.lines[-1].startswith("speaker_id summary offered=2 scored=1"))

    def test_closing_lines_print_even_for_a_label_without_a_score(self):
        harness = Harness(ScriptedEmbedder())
        harness.shadow.offered = 1
        harness.shadow._labels["Speaker B"] = LabelState()
        harness.shadow.close()
        self.assertEqual(
            harness.lines[-2:],
            [
                'speaker_id final label="Speaker B" turns=0 acc_score=- acc_s=0.0'
                " state=unknown id=-",
                "speaker_id summary offered=1 scored=0 short=0 overlap=0 dropped=0 clipped=0"
                " inferred=0 unsegmented=0 labelled_ms=0",
            ],
        )
        self.assertEqual(Harness.fields(harness.lines[-2])["label"], "Speaker B")

    def test_a_failing_report_channel_changes_nothing(self):
        def broken(_message):
            raise OSError("closed stderr")

        embedder = ScriptedEmbedder()
        shadow = ShadowSpeakerId(
            settings(),
            report=broken,
            embedder_factory=lambda _settings: embedder,
            entries=lambda _settings: ENTRIES,
        )
        shadow.start()
        shadow.audio(tone(2.0))
        shadow.turn(turn(0, 2000), spans=((0, 2000),))
        shadow.close(drain=True)
        self.assertEqual(shadow.scored, 1)
        self.assertEqual(shadow._labels["Speaker A"].turns, 1)


OWNER_F0, GUEST_F0 = 400.0, 140.0


class ToneEmbedder:
    """Keyed on audio content: each 20 ms frame is the owner's tone or the guest's.

    The embedding is the frame-weighted mix of the owner's and guest's voiceprints, so any
    owner audio in a guest label's evidence moves it toward `owner`. Counts every frame it
    was shown, by voice.
    """

    def __init__(self):
        self.model = dict(MODEL)
        self.calls = 0
        self.frames = {"owner": 0, "guest": 0}
        self.closed = False

    def embed(self, pcm: bytes) -> list[float]:
        self.calls += 1
        samples = array("h")
        samples.frombytes(pcm)
        mix = {"owner": 0, "guest": 0}
        for index in range(0, len(samples) - 319, 320):
            frame = samples[index : index + 320]
            crossings = sum((a < 0) != (b < 0) for a, b in zip(frame, frame[1:]))
            # 140 Hz crosses zero about 6 times in 20 ms; 400 Hz about 16 times.
            mix["owner" if crossings > 10 else "guest"] += 1
        for voice, count in mix.items():
            self.frames[voice] += count
        return unit(mix["owner"], mix["guest"], 0)

    def close(self):
        self.closed = True


class SegmentScoringTests(unittest.TestCase):
    """Only the label's own diarization pieces of a turn are embedded (#148)."""

    def shadow(self, embedder, **overrides):
        lines: list[str] = []
        shadow = ShadowSpeakerId(
            settings(**overrides),
            report=lines.append,
            embedder_factory=lambda _settings: embedder,
            entries=lambda _settings: [dict(entry) for entry in ENTRIES],
        )
        shadow.start()
        return shadow, lines

    def test_a_guest_turn_with_an_owner_tail_embeds_only_guest_pieces(self):
        embedder = ToneEmbedder()
        shadow, lines = self.shadow(embedder, bind_min_seconds=1.0)
        received = 0
        for index in range(4):
            # 2 s of the guest, then the owner's 1 s tail joined under the guest's label.
            shadow.audio(tone(2.0, GUEST_F0) + tone(1.0, OWNER_F0))
            current = turn(received, received + 3000, "Speaker B")
            shadow.turn(current, inferred=True, spans=((received, received + 2000),))
            received += 3000
        shadow.close(drain=True)
        self.assertEqual(embedder.frames["owner"], 0)
        self.assertGreater(embedder.frames["guest"], 0)
        self.assertEqual((shadow.offered, shadow.inferred, shadow.scored), (4, 4, 4))
        self.assertEqual(shadow.labelled_ms, 8000)
        state = shadow._labels["Speaker B"]
        self.assertEqual((state.state, state.identity), ("bound", "guest"))
        self.assertLess(state.scores["owner"], 0.01)
        roles = EnrolledRoles(SpeakerPriority(owners=("owner",)))
        roles.source = shadow
        self.assertEqual(roles.role_for(turn(0, 1, "Speaker B")), "participant")
        notes = [Harness.fields(line) for line in lines if line.startswith("speaker_id label=")]
        self.assertEqual({note["turn_ms"] for note in notes}, {"3000"})
        self.assertEqual({note["labelled_ms"] for note in notes}, {"2000"})
        self.assertTrue(lines[-1].endswith(" inferred=4 unsegmented=0 labelled_ms=8000"))

    def test_the_same_turn_scored_whole_would_have_moved_toward_owner(self):
        """Control: the tone embedder does see owner audio when it is in the pieces."""
        embedder = ToneEmbedder()
        shadow, _ = self.shadow(embedder)
        shadow.audio(tone(2.0, GUEST_F0) + tone(1.0, OWNER_F0))
        shadow.turn(turn(0, 3000, "Speaker B"), spans=((0, 3000),))
        shadow.close(drain=True)
        self.assertGreater(embedder.frames["owner"], 0)
        self.assertGreater(shadow._labels["Speaker B"].scores["owner"], 0.3)

    def test_an_owner_turn_with_an_inferred_guest_tail_still_binds_the_owner(self):
        embedder = ToneEmbedder()
        shadow, _ = self.shadow(embedder)  # default bind_min_seconds: 3 s
        received = 0
        for _ in range(2):
            shadow.audio(tone(2.0, OWNER_F0) + tone(1.0, GUEST_F0))
            current = turn(received, received + 3000, "Speaker A")
            shadow.turn(current, inferred=True, spans=((received, received + 2000),))
            received += 3000
        shadow.close(drain=True)
        self.assertEqual(embedder.frames["guest"], 0)
        state = shadow._labels["Speaker A"]
        self.assertEqual((state.turns, state.seconds), (2, 4.0))
        self.assertEqual(shadow.binding("Speaker A"), ("bound", "owner"))

    def test_pieces_are_joined_and_another_voice_between_them_is_left_out(self):
        embedder = ToneEmbedder()
        shadow, lines = self.shadow(embedder)
        # Owner 0-1.5 s, a guest backchannel 1.5-2 s the recognizer dropped, owner 2-3.5 s.
        shadow.audio(tone(1.5, OWNER_F0) + tone(0.5, GUEST_F0) + tone(1.5, OWNER_F0))
        shadow.turn(turn(0, 3500), spans=((0, 1500), (2000, 3500)))
        shadow.close(drain=True)
        self.assertEqual(embedder.frames["guest"], 0)
        self.assertEqual(shadow.scored, 1)
        self.assertEqual(shadow._labels["Speaker A"].seconds, 3.0)
        note = Harness.fields([line for line in lines if "label=" in line][0])
        self.assertEqual((note["turn_ms"], note["labelled_ms"]), ("3500", "3000"))

    def test_min_turn_seconds_applies_to_the_summed_labelled_speech(self):
        embedder = ToneEmbedder()
        shadow, _ = self.shadow(embedder, min_turn_seconds=1.0)
        shadow.audio(tone(4.0, OWNER_F0))
        # A 4 s turn with only 0.8 s of it labelled: too short to score.
        shadow.turn(turn(0, 4000), spans=((0, 400), (3600, 4000)))
        # Two 0.6 s pieces sum to 1.2 s: scored.
        shadow.audio(tone(4.0, OWNER_F0))
        shadow.turn(turn(4000, 8000), spans=((4000, 4600), (7400, 8000)))
        shadow.close(drain=True)
        self.assertEqual((shadow.short, shadow.scored), (1, 1))
        self.assertEqual(shadow.labelled_ms, 2000)

    def test_a_turn_without_pieces_is_skipped_and_counted(self):
        embedder = ToneEmbedder()
        shadow, lines = self.shadow(embedder)
        shadow.audio(tone(8.0, OWNER_F0))
        shadow.turn(turn(0, 2000))  # no pieces were reported
        shadow.turn(turn(2000, 4000), spans=())  # the timeline gave the label none
        shadow.turn(turn(4000, 6000), spans=((0, 1000), (6500, 7000)))  # none inside
        shadow.turn(turn(6000, 8000), inferred=True, spans=[("bad", 1), (7000, 7000)])
        shadow.close(drain=True)
        self.assertEqual(embedder.calls, 0)
        self.assertEqual((shadow.offered, shadow.unsegmented, shadow.scored), (4, 4, 0))
        self.assertIn(
            "speaker_id summary offered=4 scored=0 short=0 overlap=0 dropped=0 clipped=0"
            " inferred=1 unsegmented=4 labelled_ms=0",
            lines,
        )

    def test_pieces_outside_the_turn_are_clipped_to_it(self):
        embedder = ToneEmbedder()
        shadow, _ = self.shadow(embedder)
        shadow.audio(tone(1.0, GUEST_F0) + tone(2.0, OWNER_F0) + tone(1.0, GUEST_F0))
        # Overlapping, unordered pieces reaching past both ends of the 1-3 s turn.
        shadow.turn(turn(1000, 3000), spans=((2500, 4000), (0, 2000), (1500, 2600)))
        shadow.close(drain=True)
        self.assertEqual(embedder.frames["guest"], 0)
        self.assertEqual(shadow.labelled_ms, 2000)


class LabelledSpanTests(unittest.TestCase):
    """The live window reports each turn's own-label, unoverlapped timeline pieces."""

    def test_pieces_exclude_other_speakers_overlap_and_unlabelled_audio(self):
        timeline = [
            {"start_ms": 0, "end_ms": 1000, "speaker": 1},
            {"start_ms": 400, "end_ms": 600, "speaker": 2},  # overlaps A
            {"start_ms": 1200, "end_ms": 1500, "speaker": 1},  # after an unlabelled gap
            {"start_ms": 1400, "end_ms": 1800, "speaker": 2},
            {"start_ms": 1500, "end_ms": 1700, "speaker": 1},  # inside B: overlap only
        ]
        self.assertEqual(
            _labelled_spans("Speaker A", 100, 1900, timeline),
            [(100, 400), (600, 1000), (1200, 1400)],
        )
        self.assertEqual(_labelled_spans("Speaker B", 0, 2000, timeline), [(1700, 1800)])
        self.assertEqual(_labelled_spans("Speaker C", 0, 2000, timeline), [])
        self.assertEqual(_labelled_spans(None, 0, 2000, timeline), [])
        self.assertEqual(_labelled_spans("Speaker A", 500, 500, timeline), [])
        # Touching segments of the same label merge into one piece.
        touching = [
            {"start_ms": 0, "end_ms": 500, "speaker": 1},
            {"start_ms": 500, "end_ms": 900, "speaker": 1},
        ]
        self.assertEqual(_labelled_spans("Speaker A", 0, 1000, touching), [(0, 900)])

    def run_live(self, units, timeline, **options):
        turns, spans, inferred = [], {}, []
        config = LiveConfig(
            "span-test",
            provenance="causal-replay",
            transcriber=units,
            diarizer=timeline,
            on_inferred=inferred.append,
            on_labelled_spans=lambda utterance, pieces: spans.__setitem__(utterance, pieces),
            **options,
        )
        processor = LiveProcessor(config, turns.append)
        for offset in range(0, len(SPEECH), FRAME_BYTES * 10):
            processor.push_pcm16(SPEECH[offset : offset + FRAME_BYTES * 10])
        processor.finish()
        return turns, spans, inferred

    def test_edge_attributed_words_are_outside_the_pieces(self):
        units = Units((" Turn the volume", 0, 600), (" down", 600, 800), (" a bit.", 850, 1000))
        turns, spans, inferred = self.run_live(
            units, Timeline((0, 700, 1)), edge_attribution_ms=300
        )
        self.assertEqual(
            [(t.speaker_id, t.start_ms, t.end_ms) for t in turns], [("Speaker A", 0, 1000)]
        )
        self.assertEqual(inferred, [turns[0].utterance_id])
        self.assertEqual(spans, {turns[0].utterance_id: ((0, 700),)})

    def test_an_untranscribed_voice_inside_the_turn_is_left_out(self):
        units = Units((" Turn the volume", 0, 400), (" down.", 600, 1000))
        turns, spans, _ = self.run_live(units, Timeline((0, 1000, 1), (420, 580, 2)))
        self.assertEqual([t.speaker_id for t in turns], ["Speaker A"])
        self.assertEqual(spans, {turns[0].utterance_id: ((0, 420), (580, 1000))})

    def test_each_turn_gets_its_own_pieces_and_unlabelled_turns_none(self):
        units = Units((" Hello", 0, 300), (" there.", 400, 700), (" Hm.", 800, 950))
        turns, spans, _ = self.run_live(units, Timeline((0, 350, 1), (380, 720, 2)))
        self.assertEqual([t.speaker_id for t in turns], ["Speaker A", "Speaker B", None])
        self.assertEqual([spans[t.utterance_id] for t in turns], [((0, 300),), ((400, 700),), ()])

    def test_without_an_observer_nothing_is_computed_or_reported(self):
        units = Units(
            (" Turn the volume", 0, 600),
        )
        turns = []
        config = LiveConfig(
            "span-test",
            provenance="causal-replay",
            transcriber=units,
            diarizer=Timeline((0, 700, 1)),
        )
        processor = LiveProcessor(config, turns.append)
        with patch("rightyo.live_audio._labelled_spans") as computed:
            for offset in range(0, len(SPEECH), FRAME_BYTES * 10):
                processor.push_pcm16(SPEECH[offset : offset + FRAME_BYTES * 10])
            processor.finish()
        computed.assert_not_called()
        self.assertEqual(len(turns), 1)

    def test_utterance_local_labels_get_no_pieces(self):
        units = Units(
            (" Turn the volume", 0, 600),
        )
        turns, spans, _ = self.run_live(
            units, Timeline((0, 700, 1), provenance="diarization-utterance")
        )
        self.assertEqual(len(turns), 1)
        self.assertEqual(spans, {})

    def test_an_invalid_observer_is_refused(self):
        with self.assertRaises(LiveAudioError):
            LiveConfig("x", transcriber=Units(), diarizer=Timeline(), on_labelled_spans="no")

    def test_joined_fragments_carry_the_union_of_their_pieces(self):
        emitted = []
        merger = TurnMerger(2000, 24000, emitted.append, tail_join_ms=400)
        held = fragment("Turn the volume", 0, 600)
        held["labelled_spans"] = [(0, 600)]
        tail = fragment("down a bit.", 900, 1100, speaker=None)
        tail["labelled_spans"] = []
        merger.offer(held)
        merger.offer(tail)
        more = fragment("Thanks.", 1500, 1800)
        more["labelled_spans"] = [(1500, 1750)]
        merger.offer(more)
        merger.flush()
        self.assertEqual(len(emitted), 1)
        self.assertTrue(emitted[0]["inferred"])
        self.assertEqual(emitted[0]["labelled_spans"], [(0, 600), (1500, 1750)])
        plain = []
        merger = TurnMerger(2000, 24000, plain.append)
        merger.offer(fragment("Turn the volume", 0, 600))
        merger.offer(fragment("down.", 1000, 1200))
        merger.flush()
        self.assertNotIn("labelled_spans", plain[0])


class TurnEveryTwoSeconds:
    """A stand-in processor: one finalized turn per two seconds of audio."""

    def __init__(self, config, on_turn):
        self.config, self.on_turn = config, on_turn
        self.received_ms = self.emitted_ms = 0
        self.count = 0

    def push_pcm16(self, pcm):
        self.received_ms += len(pcm) // 32
        while self.received_ms - self.emitted_ms >= 2000:
            self.count += 1
            start, self.emitted_ms = self.emitted_ms, self.emitted_ms + 2000
            if self.config.on_labelled_spans is not None:
                # The whole turn is its label's own timeline audio (#148).
                self.config.on_labelled_spans(f"live-{self.count}", ((start, self.emitted_ms),))
            self.on_turn(
                Turn(
                    self.config.session_id,
                    f"live-{self.count}",
                    1,
                    start,
                    self.emitted_ms,
                    PRIVATE_TEXT,
                    "Speaker A" if self.count % 2 else "Speaker B",
                    True,
                    False,
                    "fake-local-recognizer",
                    self.config.provenance,
                    "diarization-timeline",
                )
            )

    def finish(self):
        pass

    def close(self):
        pass


class ControllerTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        asset = Path(directory.name) / "asset"
        asset.touch()
        demo = Path(directory.name) / "generated-tone.wav"
        with wave.open(str(demo), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(RATE)
            audio.writeframes(tone(8.0))
        self.base = PrototypeConfig(asset, asset, asset, asset, asset, demo)
        self.asset = asset

    def session(self, config, factory):
        lines: list[str] = []
        controller = PrototypeController(
            config,
            processor_factory=TurnEveryTwoSeconds,
            capture_factory=MagicMock(),
            provider_factory=MagicMock(),
            event_publisher=SpeechEvents(),
            report=lines.append,
            speaker_id_factory=factory,
        )
        self.addCleanup(controller.close)
        controller.start({"mode": "demo", "session_id": "shadow-session"})
        wait_for(lambda: controller.snapshot(heartbeat=False)["phase"] == "complete", 5)
        wait_for(lambda: not controller._audio_thread.is_alive(), 5)
        events = json.dumps(controller.drain_events(), sort_keys=True)
        return events, lines

    def test_live_false_starts_no_worker_and_events_are_byte_identical(self):
        baseline, baseline_lines = self.session(self.base, MagicMock())
        factory = MagicMock()
        configured = replace(self.base, speaker_id=SpeakerIdConfig(self.asset, self.asset))
        events, lines = self.session(configured, factory)
        factory.assert_not_called()
        self.assertEqual(events, baseline)
        self.assertEqual(lines, baseline_lines)
        self.assertIn('"transcript"', events)

    def test_live_true_scores_in_the_shadow_and_leaves_events_unchanged(self):
        baseline, _ = self.session(self.base, MagicMock())
        embedder = ScriptedEmbedder()
        created = []

        def factory(settings, *, report):
            shadow = ShadowSpeakerId(
                settings,
                report=report,
                embedder_factory=lambda _settings: embedder,
                entries=lambda _settings: ENTRIES,
            )
            created.append(shadow)
            return shadow

        live = replace(self.base, speaker_id=SpeakerIdConfig(self.asset, self.asset, live=True))
        events, lines = self.session(live, factory)
        self.assertEqual(events, baseline)
        self.assertEqual(len(created), 1)
        notes = [line for line in lines if line.startswith("speaker_id label=")]
        self.assertEqual(len(notes), 4)
        self.assertEqual(embedder.calls, 4)
        self.assertTrue(embedder.closed)
        self.assertNotIn(PRIVATE_TEXT, "\n".join(lines))
        self.assertEqual(created[0]._ring._buffer, bytearray())

    def test_labelled_pieces_are_requested_only_with_an_identifier(self):
        configs = []

        def processor(config, on_turn):
            configs.append(config)
            return TurnEveryTwoSeconds(config, on_turn)

        live = replace(self.base, speaker_id=SpeakerIdConfig(self.asset, self.asset, live=True))
        for config, factory in (
            (self.base, MagicMock()),
            (live, MagicMock(side_effect=RuntimeError("synthetic"))),
        ):
            controller = PrototypeController(
                config,
                processor_factory=processor,
                capture_factory=MagicMock(),
                provider_factory=MagicMock(),
                report=lambda _line: None,
                speaker_id_factory=factory,
            )
            self.addCleanup(controller.close)
            controller.start({"mode": "demo"})
            wait_for(lambda c=controller: c.snapshot(heartbeat=False)["phase"] == "complete", 5)
        self.assertEqual([config.on_labelled_spans for config in configs], [None, None])

    def test_without_a_diagnostic_channel_no_worker_starts(self):
        factory = MagicMock()
        live = replace(self.base, speaker_id=SpeakerIdConfig(self.asset, self.asset, live=True))
        controller = PrototypeController(
            live,
            processor_factory=TurnEveryTwoSeconds,
            capture_factory=MagicMock(),
            provider_factory=MagicMock(),
            speaker_id_factory=factory,
        )
        self.addCleanup(controller.close)
        controller.start({"mode": "demo"})
        wait_for(lambda: controller.snapshot(heartbeat=False)["phase"] == "complete", 5)
        factory.assert_not_called()

    def test_a_factory_failure_never_fails_the_session(self):
        baseline, _ = self.session(self.base, MagicMock())

        def factory(settings, *, report):
            raise RuntimeError("synthetic")

        live = replace(self.base, speaker_id=SpeakerIdConfig(self.asset, self.asset, live=True))
        events, lines = self.session(live, factory)
        self.assertEqual(events, baseline)
        self.assertIn("speaker_id unavailable; shadow identification off for this session", lines)


class ListenTests(unittest.TestCase):
    """`rightyo listen` wires stderr notes in every mode, so shadow ID runs in demo too."""

    def test_demo_listen_writes_shadow_notes_to_stderr(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset = root / "asset"
            asset.touch()
            demo = root / "generated-tone.wav"
            with wave.open(str(demo), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(RATE)
                audio.writeframes(tone(4.0))
            config = root / "config.json"
            local = (
                "whisper_executable",
                "whisper_model",
                "diarization_library",
                "diarization_model",
                "microphone_helper",
            )
            config.write_text(
                json.dumps(
                    {
                        **{key: str(asset) for key in local},
                        "demo_audio": str(demo),
                        "speaker_id": {"python": str(asset), "model": str(asset), "live": True},
                    }
                )
            )
            embedder = ScriptedEmbedder()

            def shadow(settings, *, report):
                return ShadowSpeakerId(
                    settings,
                    report=report,
                    embedder_factory=lambda _settings: embedder,
                    entries=lambda _settings: ENTRIES,
                )

            def factory(loaded, *, event_publisher, report=None):
                return PrototypeController(
                    loaded,
                    event_publisher=event_publisher,
                    report=report,
                    processor_factory=TurnEveryTwoSeconds,
                    capture_factory=MagicMock(side_effect=AssertionError("no microphone")),
                    provider_factory=MagicMock(side_effect=AssertionError("no hosted")),
                    speaker_id_factory=shadow,
                )

            args = Namespace(
                config=config,
                mode="demo",
                session_id="listen-shadow",
                use_jev=False,
                allow_hosted=False,
            )
            output = io.StringIO()
            with patch("sys.stderr", io.StringIO()) as errors:
                self.assertEqual(listen(args, output=output, controller_factory=factory), 0)
                # `listen` returns once the session completes; the worker closes after.
                wait_for(lambda: "speaker_id summary" in errors.getvalue(), 5)
            log = errors.getvalue()
            self.assertIn("rightyo: speaker_id start enrolled=2", log)
            notes = [line for line in log.splitlines() if "speaker_id label=" in line]
            self.assertGreaterEqual(len(notes), 1)
            self.assertNotIn(PRIVATE_TEXT, log)
            self.assertIn(PRIVATE_TEXT, output.getvalue())  # the transcript still went out


if __name__ == "__main__":
    unittest.main()
