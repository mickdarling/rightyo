"""Opt-in #350 owner recordings. File replay with local Silero, Whisper and WeSpeaker."""

import json
import os
import queue
import re
import unittest
import wave
from array import array
from pathlib import Path
from unittest.mock import patch

from test_duck_mode import OURS, SESSION
from test_live_audio import FakeDiarizer
from test_live_speaker_id import wait_for
from test_silero_vad import RealSilero, tones

from rightyo.contracts import SpeakerPriority
from rightyo.enroll import MAX_ENTRY_BYTES, _entry, inside_git_checkout
from rightyo.live_audio import BYTES_PER_MS, LiveConfig, LiveProcessor
from rightyo.live_speaker_id import EnrolledRoles, ShadowSpeakerId
from rightyo.memory import TranscriptMemory
from rightyo.prototype import PrototypeConfig, PrototypeController
from rightyo.silero_vad import CHUNK_MS
from rightyo.speaker_id import SpeakerIdConfig
from rightyo.tool_events import SpeechEvents, privacy_phrase

LEAD_MS = 400
TRAIL_MS = 2304  # enough observed silence for ASR close and confirmation from the last word
PODCAST = "jarvis start listening to the podcast"


def quiet(ms):
    return bytes(ms * BYTES_PER_MS)


def words(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def with_machine(pcm, hz):
    # Align the synthesized background with sample zero of the entire replay.
    tone = array("h", tones((hz, 1000)))
    return array(
        "h",
        (max(-32768, min(32767, s + tone[i % len(tone)])) for i, s in enumerate(array("h", pcm))),
    ).tobytes()


class OwnerClipTests(RealSilero, unittest.TestCase):
    def setUp(self):
        self.root = Path(
            os.environ.get("RIGHTYO_350_FIXTURES", "~/.local/share/rightyo-fixtures/350")
        ).expanduser()
        if not self.root.is_dir():
            self.skipTest("owner clip directory absent; see docs/prototype.md")
        super().setUp()

    def asset(self, key):
        value = os.environ.get(key)
        if not value or not Path(value).is_file():
            self.skipTest(f"local asset absent; set {key} (docs/prototype.md)")
        return Path(value)

    def clip(self, name):
        path = self.root / f"{name}.wav"
        if not path.is_file():
            self.skipTest(f"owner clip {name}.wav absent; see docs/prototype.md")
        self.assertFalse(inside_git_checkout(path), "owner clips belong outside Git")
        with wave.open(str(path), "rb") as wav:
            self.assertEqual(
                (wav.getnchannels(), wav.getframerate(), wav.getsampwidth(), wav.getcomptype()),
                (1, 16000, 2, "NONE"),
            )
            frames = wav.getnframes()
            self.assertTrue(0 < frames <= 10 * 16000, "expected one trimmed owner phrase")
            pcm = wav.readframes(frames)
            self.assertEqual(len(pcm), frames * 2)
            return pcm

    def start_replay(self, *expected_turns, private=True):
        cli = self.asset("RIGHTYO_350_WHISPER_CLI")
        model = self.asset("RIGHTYO_350_WHISPER_MODEL")
        speaker_model = self.asset("RIGHTYO_350_SPEAKER_MODEL")
        enrollment = self.asset("RIGHTYO_350_OWNER_PRINT")
        self.assertFalse(inside_git_checkout(enrollment), "owner enrollment belongs outside Git")
        # Read just the designated, separately enrolled print. Store.entries() may chmod a store.
        with enrollment.open("rb") as source:
            content = source.read(MAX_ENTRY_BYTES + 1)
            self.assertLessEqual(len(content), MAX_ENTRY_BYTES)
            entry = _entry(json.loads(content), enrollment.stem)
        priority = SpeakerPriority(owners=(entry["id"],))
        self.shadow = ShadowSpeakerId(
            SpeakerIdConfig(self.settings.python, speaker_model, live=True, roles=True),
            entries=lambda _: [entry],
        )
        self.shadow.start()
        self.addCleanup(self.shadow.close)
        wait_for(
            lambda: self.shadow.owner_evidence(priority.owners)[0] or not self.shadow.active,
            timeout=40,
        )
        self.assertTrue(self.shadow.owner_evidence(priority.owners)[0], "owner print unavailable")

        self.matches = []
        score = self.shadow.exit_score

        def count_check(pcm, owners):
            self.matches.append(False)
            result = score(pcm, owners)
            self.matches[-1] = (
                result.outcome == "scored"
                and result.labelled >= self.shadow.settings.entry_threshold
            )
            return result

        self.shadow.exit_score = count_check
        roles = EnrolledRoles(priority)
        roles.source = self.shadow
        self.events = SpeechEvents()
        self.events.start(SESSION, addressing=OURS, priority=roles, duck=True)
        self.events.drain()
        c = self.controller = PrototypeController(
            PrototypeConfig(cli, model, None, None, None, addressing=OURS, speakers=priority),
            event_publisher=self.events,
        )
        self.addCleanup(c.close)
        c._enrolled_roles = roles
        c._shadow = self.shadow
        roles.exit_audio = c._private_sentence

        self.vad = self.real_worker()
        config = LiveConfig(
            SESSION,
            whisper_executable=cli,
            whisper_model=model,
            diarizer=FakeDiarizer,
            provenance="causal-replay",
            silero_vad=self.vad,
            private=c._heard_private_span,
            pre_roll_ms=LEAD_MS,
            turn_merge_gap_ms=4000,
            turn_break=c._turn_break(),
            on_private_turn_audio=lambda turn, pcm: c._private_turn_audio(c._generation, turn, pcm),
        )
        self.memory = TranscriptMemory()
        work = queue.Queue()
        self.expected_count = len(expected_turns)
        pending = iter(expected_turns)
        self.recognized = []

        def publish(turn):
            # Compare in memory; assertion failures must never print recognizer output.
            self.recognized.append(words(turn.text) == next(pending, None))
            c._turn(c._generation, work, self.memory, self.shadow, turn)

        self.processor = LiveProcessor(config, publish)
        self.addCleanup(self.processor.close)
        c._processor = self.processor
        if private:
            self.enter_privacy()

    def enter_privacy(self):
        self.controller.duck(True)
        self.events.drain()
        self.public_ids = self.memory.retained_ids

    def push(self, pcm):
        for at in range(0, len(pcm), 640):
            self.assertTrue(
                self.controller._feed(
                    self.controller._generation,
                    self.processor,
                    pcm[at : at + 640],
                    "demo",
                    self.shadow,
                )
            )
        self.assertIs(self.processor._vad, self.vad, "Silero worker failed")

    def private_result(self, checks, exits):
        self.assertTrue(self.shadow.active, "owner verifier unavailable during replay")
        self.assertTrue(self.events.ducked)
        events = self.events.drain()
        self.assertEqual(len(events), exits, "unexpected events in privacy")
        for event in events:
            self.assertEqual(
                set(event),
                {"schema_version", "sequence", "emitted_at_ms", "session_id", "type", "action"},
            )
            self.assertTrue((event["type"], event["action"]) == ("privacy", "exit"))
        self.assertEqual(self.matches, [True] * checks, "owner check count or outcome")
        self.assertTrue(self.memory.retained_ids == self.public_ids)
        self.assertEqual(len(self.controller._private_sentence.pcm), 0, "private audio retained")

    def finish_private(self, checks, exits):
        self.private_result(checks, exits)  # Natural close, before end-of-input can flush a turn.
        self.assertEqual(self.recognized, [True] * self.expected_count, "unexpected recognition")
        self.processor.finish()
        self.private_result(checks, 0)
        self.assertEqual(self.recognized, [True] * self.expected_count)

    def single_exit(self, hz=None):
        command = self.clip("privacy-off")
        self.start_replay("jarvis privacy off")
        lead = quiet(24000 if hz in (100, 200) else LEAD_MS)
        stream = lead + command + quiet(TRAIL_MS)
        if hz:
            stream = with_machine(stream, hz)
        self.push(stream[: len(lead)])
        background_windows = self.processor._utterances + bool(self.processor._utterance)
        self.push(stream[len(lead) :])
        if hz in (100, 200):
            with self.subTest(phase="machinery alone"):
                self.assertEqual(background_windows, 0, "background alone opened ASR")
        with self.subTest(phase="single spoken exit"):
            self.finish_private(1, 1)

    def test_single_exit_clean(self):
        self.single_exit()

    def test_single_exit_over_50_hz(self):
        self.single_exit(50)

    def test_single_exit_over_100_hz(self):
        self.single_exit(100)

    def test_single_exit_over_200_hz(self):
        self.single_exit(200)

    def split_exit(self, gap, checks, exits):
        name, rest = self.clip("jarvis"), self.clip("start-listening")
        self.start_replay("jarvis", "start listening")
        self.push(quiet(LEAD_MS) + name + quiet(gap))
        self.assertEqual(self.recognized, [True], "name must finalize on its own")
        self.private_result(1, 0)
        self.push(rest + quiet(TRAIL_MS))
        self.finish_private(checks, exits)

    def test_split_exit_at_two_seconds(self):
        self.split_exit(2000, 2, 1)

    def test_split_exit_expires_at_four_point_two_seconds(self):
        self.split_exit(4200, 1, 0)

    def test_remainder_alone_stays_private(self):
        rest = self.clip("start-listening")
        self.start_replay("start listening")
        self.push(quiet(LEAD_MS) + rest + quiet(TRAIL_MS))
        self.finish_private(0, 0)

    def entry(self, name, expected):
        pcm = quiet(LEAD_MS) + self.clip(name) + quiet(TRAIL_MS)
        self.start_replay(expected, expected, private=False)
        self.push(pcm)
        events = self.events.drain()
        self.assertEqual([e["type"] for e in events], ["transcript"])
        self.assertTrue(words(events[0]["turn"]["text"]) == expected, "wrong public entry")
        self.enter_privacy()
        self.push(pcm)
        self.finish_private(0, 0)

    def test_stop_recording_in_public_and_privacy(self):
        self.entry("stop-recording", "jarvis stop recording")

    def test_go_private_in_public_and_privacy(self):
        self.entry("go-private", "go private")

    def continuing_sentence(self, *, omit_tail):
        sentence = self.clip("podcast")
        expected = ("jarvis start listening",) if omit_tail else (PODCAST, "to the podcast")
        self.start_replay(*expected)
        labels = bytearray()
        classify = self.vad.is_speech

        def observe(block):
            label = classify(block)
            labels.append(label)
            return label

        recognizer = self.processor._transcriber
        decode = recognizer.transcribe
        tail_starts = []
        end_ms = LEAD_MS + len(sentence) // BYTES_PER_MS

        def recognize(pcm, register=None):
            self.assertGreaterEqual(
                self.processor._utterance_start + len(pcm) // BYTES_PER_MS,
                end_ms,
                "ASR closed during continuing speech",
            )
            units = decode(pcm, register)
            text = " ".join(u["text"] for u in units)
            self.assertTrue(words(text) == PODCAST, "expected the full recorded sentence")
            self.assertNotIn("exit", privacy_phrase(text, OURS))
            cut = next((i for i, u in enumerate(units) if words(u["text"]) == "to"), None)
            self.assertIsNotNone(cut, "expected word timing for the recorded tail")
            tail_starts.append(self.processor._utterance_start + units[cut]["start_ms"])
            if omit_tail:
                # Drop only Whisper's suffix and its timing. Silero still hears the entire WAV.
                self.assertIn(
                    "exit", privacy_phrase(" ".join(u["text"] for u in units[:cut]), OURS)
                )
                window = recognizer._window
                self.assertFalse(
                    window.timing_incomplete, "expected reliable original recognition timings"
                )
                del units[cut:]
                if window.spoken_intervals is not None:
                    end = units[-1]["end_ms"]
                    window.spoken_intervals[:] = [i for i in window.spoken_intervals if i[1] <= end]
            return units

        with (
            patch.object(self.vad, "is_speech", observe),
            patch.object(recognizer, "transcribe", recognize),
        ):
            self.push(quiet(LEAD_MS) + sentence)
            self.assertEqual(self.processor._utterances, 0, "ASR closed during continuing speech")
            self.assertTrue(bool(self.processor._utterance), "no speech window for the sentence")
            self.push(quiet(TRAIL_MS))
        self.assertEqual(len(tail_starts), 1, "expected one natural ASR close")
        tail_ms = tail_starts[0]
        self.assertTrue(LEAD_MS < tail_ms < end_ms, "tail timing outside the recording")
        self.assertTrue(
            any(labels[(tail_ms + CHUNK_MS - 1) // CHUNK_MS : end_ms // CHUNK_MS]),
            "Silero missed continuing speech after the exit-shaped prefix",
        )
        self.assertNotIn(
            bytes(7),  # Seven whole 32 ms blocks could supply a 200 ms absence.
            labels[(tail_ms - 400) // CHUNK_MS : tail_ms // CHUNK_MS],
            "Silero certified an onset gap in the uninterrupted sentence",
        )
        self.assertEqual(self.recognized, [True], "expected one matching private sentence")
        self.private_result(0, 0)
        return sentence[(tail_ms - LEAD_MS) * BYTES_PER_MS :]

    def test_complete_sentence_and_side_talk_stay_private(self):
        tail = self.continuing_sentence(omit_tail=False)
        self.push(quiet(LEAD_MS) + tail + quiet(TRAIL_MS))
        self.finish_private(0, 0)

    def test_omitted_continuing_speech_stays_private(self):
        self.continuing_sentence(omit_tail=True)
        self.finish_private(0, 0)
