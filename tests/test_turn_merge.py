"""Joining a speaker's turn split by a short pause before the decision (#73).

Synthetic PCM, a scripted transcriber and a scripted diarizer only: no capture, native
inference, hosted calls or recorded speech.
"""

from __future__ import annotations

import array
import contextlib
import io
import json
import tempfile
import unittest
import wave
from argparse import Namespace
from dataclasses import replace
from pathlib import Path

from rightyo.contracts import SpeakerPriority
from rightyo.live_audio import FRAME_BYTES, LiveAudioError, LiveConfig, LiveProcessor
from rightyo.prototype import PrototypeConfig, PrototypeController, PrototypeError
from rightyo.providers import MockProvider
from rightyo.tool import listen
from rightyo.turn_merge import DEFAULT_TURN_MERGE_GAP_MS, TurnMerger

SILENCE = bytes(FRAME_BYTES)
VOICE = array.array("h", [5000] * (FRAME_BYTES // 2)).tobytes()


def fragment(text, start, end, speaker="Speaker A", overlap=False):
    return {
        "text": text,
        "start_ms": start,
        "end_ms": end,
        "speaker": speaker,
        "overlap": overlap,
        "speaker_provenance": "diarization-timeline",
    }


class TurnMergerTests(unittest.TestCase):
    def setUp(self):
        self.emitted = []
        self.merger = TurnMerger(1000, 24000, self.emitted.append)

    def texts(self):
        return [item["text"] for item in self.emitted]

    def test_split_question_is_joined_with_first_start_and_last_end(self):
        self.merger.offer(fragment("Haili, what time is", 0, 900))
        self.merger.offer(fragment("it?", 1600, 1800))
        self.assertEqual(self.emitted, [])
        self.merger.due(2799, None)
        self.assertEqual(self.emitted, [])
        self.merger.due(2800, None)
        self.assertEqual(self.texts(), ["Haili, what time is it?"])
        self.assertEqual((self.emitted[0]["start_ms"], self.emitted[0]["end_ms"]), (0, 1800))
        self.assertEqual(self.emitted[0]["speaker"], "Speaker A")

    def test_three_fragments_chain_into_one(self):
        for text, start, end in (
            ("Haili,", 0, 400),
            ("what time", 1200, 1700),
            ("is it?", 2400, 2900),
        ):
            self.merger.offer(fragment(text, start, end))
        self.merger.flush()
        self.assertEqual(self.texts(), ["Haili, what time is it?"])

    def test_different_speakers_are_not_joined(self):
        self.merger.offer(fragment("Haili, what time is", 0, 900))
        self.merger.offer(fragment("it?", 1200, 1400, speaker="Speaker B"))
        self.assertEqual(self.texts(), ["Haili, what time is"])
        self.merger.flush()
        self.assertEqual(self.texts(), ["Haili, what time is", "it?"])
        self.assertEqual([item["speaker"] for item in self.emitted], ["Speaker A", "Speaker B"])

    def test_gap_above_threshold_is_not_joined(self):
        self.merger.offer(fragment("Haili, what time is", 0, 900))
        self.merger.offer(fragment("it?", 1901, 2100))
        self.merger.flush()
        self.assertEqual(self.texts(), ["Haili, what time is", "it?"])
        # Exactly at the threshold still joins.
        self.emitted.clear()
        self.merger.offer(fragment("Haili, what time is", 0, 900))
        self.merger.offer(fragment("it?", 1900, 2100))
        self.merger.flush()
        self.assertEqual(self.texts(), ["Haili, what time is it?"])

    def test_unattributed_or_overlapping_fragments_are_never_joined(self):
        # Joining would attribute words to a speaker the diarizer did not name.
        self.merger.offer(fragment("Haili, what time is", 0, 900))
        self.merger.offer(fragment("it?", 1000, 1200, speaker=None))
        self.merger.offer(fragment("now", 1300, 1400, speaker=None, overlap=True))
        self.assertEqual(self.texts(), ["Haili, what time is", "it?", "now"])
        self.assertFalse(self.merger.holding)

    def test_different_speaker_provenance_is_not_joined(self):
        self.merger.offer(fragment("one", 0, 100))
        other = fragment("two", 200, 300)
        other["speaker_provenance"] = "unknown"
        self.merger.offer(other)
        self.merger.flush()
        self.assertEqual(self.texts(), ["one", "two"])

    def test_an_open_utterance_inside_the_gap_keeps_the_fragment_held(self):
        self.merger.offer(fragment("Haili, what time is", 0, 900))
        # Audio of a possible continuation began at 1500 ms, inside the 1,900 ms deadline.
        self.merger.due(5000, 1500)
        self.assertEqual(self.emitted, [])
        # An utterance that opened after the deadline cannot continue the fragment.
        self.merger.due(5000, 1901)
        self.assertEqual(self.texts(), ["Haili, what time is"])

    def test_span_and_text_bounds_stop_joining(self):
        merger = TurnMerger(1000, 2000, self.emitted.append)
        merger.offer(fragment("one", 0, 900))
        merger.offer(fragment("two", 1500, 2100))
        merger.flush()
        self.assertEqual(self.texts(), ["one", "two"])
        self.emitted.clear()
        merger = TurnMerger(1000, 24000, self.emitted.append)
        merger.offer(fragment("x" * 3000, 0, 900))
        merger.offer(fragment("y" * 1000, 1000, 1100))
        merger.flush()
        self.assertEqual([len(text) for text in self.texts()], [3000, 1000])

    def test_zero_gap_emits_immediately_and_discard_drops(self):
        merger = TurnMerger(0, 24000, self.emitted.append)
        merger.offer(fragment("one", 0, 100))
        merger.offer(fragment("two", 100, 200))
        self.assertEqual(self.texts(), ["one", "two"])
        self.emitted.clear()
        self.merger.offer(fragment("held", 0, 100))
        self.merger.discard()
        self.merger.flush()
        self.assertEqual(self.emitted, [])

    def test_a_stop_phrase_is_never_joined_or_held(self):
        merger = TurnMerger(2000, 24000, self.emitted.append, SpeakerPriority().is_stop_phrase)
        merger.offer(fragment("Haili, order a pizza", 0, 900))
        merger.offer(fragment("never mind", 2400, 2900))
        # The request is released first and the stop phrase follows as a turn of its own.
        self.assertEqual(self.texts(), ["Haili, order a pizza", "never mind"])
        self.assertFalse(merger.holding)
        self.emitted.clear()
        merger.offer(fragment("Stop.", 0, 300))
        merger.offer(fragment("stop", 1000, 1300))
        self.assertEqual(self.texts(), ["Stop.", "stop"])
        self.assertFalse(merger.holding)
        # A stop phrase inside a longer turn is not a stop phrase and joins as usual.
        self.emitted.clear()
        merger.offer(fragment("please do not", 0, 300))
        merger.offer(fragment("stop the music", 1000, 1300))
        merger.flush()
        self.assertEqual(self.texts(), ["please do not stop the music"])
        with self.assertRaises(ValueError):
            TurnMerger(2000, 24000, self.emitted.append, "stop")

    def test_utterance_local_labels_never_join(self):
        first = fragment("Haili, what time is", 0, 900, speaker="u1 Speaker A")
        second = fragment("it?", 1200, 1400, speaker="u2 Speaker A")
        for item in (first, second):
            item["speaker_provenance"] = "diarization-utterance"
            self.merger.offer(item)
        self.merger.flush()
        self.assertEqual(self.texts(), ["Haili, what time is", "it?"])

    def test_gap_is_validated(self):
        for invalid in (-1, 5001, 1.5, True, "700", None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                TurnMerger(invalid, 24000, self.emitted.append)


class ScriptedTranscriber:
    """One scripted text per utterance, timed to the utterance's voiced frames."""

    recognizer_id = "scripted-fixture"

    def __init__(self, texts):
        self.texts = list(texts)

    def transcribe(self, pcm, register=None):
        voiced = [
            index
            for index in range(len(pcm) // FRAME_BYTES)
            if pcm[index * FRAME_BYTES : (index + 1) * FRAME_BYTES] == VOICE
        ]
        if not voiced or not self.texts:
            return []
        return [
            {
                "text": " " + self.texts.pop(0),
                "start_ms": voiced[0] * 20,
                "end_ms": (voiced[-1] + 1) * 20,
            }
        ]


class ScriptedDiarizer:
    """Speaker 1 throughout, or speaker 2 from `switch_ms` on (a contract-valid timeline)."""

    speaker_provenance = "diarization-timeline"

    def __init__(self, switch_ms=None):
        self.switch_ms, self.pushed_ms = switch_ms, 0

    def push(self, pcm):
        self.pushed_ms += len(pcm) // 32

    def segments(self):
        end = self.pushed_ms + 1000
        if self.switch_ms is None:
            return [{"start_ms": 0, "end_ms": end, "speaker": 1}]
        return [
            {"start_ms": 0, "end_ms": self.switch_ms, "speaker": 1},
            {"start_ms": self.switch_ms, "end_ms": end, "speaker": 2},
        ]

    finish = segments

    def close(self):
        pass


def audio(*parts):
    """Synthetic PCM from (frame, milliseconds) pairs."""
    return b"".join(frame * (ms // 20) for frame, ms in parts)


# "Haili, what time is" (0-200 ms), a 1,500 ms pause, then "it?" (1,700-1,900 ms). The
# pause exceeds the 1,440 ms hangover, so the live window finalizes two utterances.
SPLIT_QUESTION = audio((VOICE, 200), (SILENCE, 1500), (VOICE, 200), (SILENCE, 3000))
TEXTS = ("Hailey, what time is", "it?")


class LiveProcessorMergeTests(unittest.TestCase):
    def run_processor(self, pcm, *, gap, switch_ms=None, finish=True, diarizer=None):
        turns = []
        config = LiveConfig(
            "merge-test",
            provenance="causal-replay",
            transcriber=ScriptedTranscriber(TEXTS),
            diarizer=diarizer or ScriptedDiarizer(switch_ms),
            turn_merge_gap_ms=gap,
        )
        processor = LiveProcessor(config, turns.append)
        self.addCleanup(processor.close)
        for offset in range(0, len(pcm), FRAME_BYTES * 10):
            processor.push_pcm16(pcm[offset : offset + FRAME_BYTES * 10])
        if finish:
            processor.finish()
        return processor, turns

    def test_split_question_becomes_one_turn_with_coherent_ids_and_times(self):
        _, turns = self.run_processor(SPLIT_QUESTION, gap=DEFAULT_TURN_MERGE_GAP_MS, finish=False)
        # Released by stream time alone, before end of input: 1,900 + 2,000 ms.
        self.assertEqual(len(turns), 1)
        turn = turns[0]
        self.assertEqual(turn.text, "Hailey, what time is it?")
        self.assertEqual((turn.start_ms, turn.end_ms), (0, 1900))
        self.assertEqual((turn.utterance_id, turn.revision), ("live-1", 1))
        self.assertEqual(turn.speaker_id, "Speaker A")
        self.assertEqual(turn.speaker_provenance, "diarization-timeline")
        self.assertEqual(turn.recognizer_id, "scripted-fixture")

    def test_library_default_keeps_every_finalized_turn_separate(self):
        self.assertEqual(
            LiveConfig(
                "x", transcriber=ScriptedTranscriber(()), diarizer=ScriptedDiarizer()
            ).turn_merge_gap_ms,
            0,
        )
        _, turns = self.run_processor(SPLIT_QUESTION, gap=0)
        self.assertEqual([t.text for t in turns], list(TEXTS))
        self.assertEqual([t.utterance_id for t in turns], ["live-1", "live-2"])

    def test_gap_above_threshold_is_not_merged(self):
        _, turns = self.run_processor(SPLIT_QUESTION, gap=1000)
        self.assertEqual([t.text for t in turns], list(TEXTS))
        self.assertEqual([(t.start_ms, t.end_ms) for t in turns], [(0, 200), (1700, 1900)])

    def test_different_speakers_are_not_merged(self):
        _, turns = self.run_processor(SPLIT_QUESTION, gap=DEFAULT_TURN_MERGE_GAP_MS, switch_ms=1000)
        self.assertEqual([t.text for t in turns], list(TEXTS))
        self.assertEqual([t.speaker_id for t in turns], ["Speaker A", "Speaker B"])

    def test_utterance_local_diarizer_turns_are_never_joined(self):
        diarizer = ScriptedDiarizer()
        diarizer.speaker_provenance = "diarization-utterance"
        _, turns = self.run_processor(
            SPLIT_QUESTION, gap=DEFAULT_TURN_MERGE_GAP_MS, diarizer=diarizer
        )
        self.assertEqual([t.text for t in turns], list(TEXTS))
        self.assertEqual([t.speaker_id for t in turns], ["u1 Speaker A", "u2 Speaker A"])

    def test_held_turn_waits_only_for_its_gap(self):
        pcm = audio((VOICE, 200), (SILENCE, 1440))
        processor, turns = self.run_processor(pcm, gap=DEFAULT_TURN_MERGE_GAP_MS, finish=False)
        # Finalized at 1,640 ms and held until 200 + 2,000 ms of stream time.
        self.assertEqual(turns, [])
        for _ in range(27):
            processor.push_pcm16(SILENCE)
        self.assertEqual(turns, [])
        processor.push_pcm16(SILENCE)
        self.assertEqual([t.text for t in turns], ["Hailey, what time is"])

    def test_release_finish_and_close(self):
        pcm = audio((VOICE, 200), (SILENCE, 1440))
        processor, turns = self.run_processor(pcm, gap=DEFAULT_TURN_MERGE_GAP_MS, finish=False)
        processor.release_pending()
        self.assertEqual(len(turns), 1)
        processor, turns = self.run_processor(pcm, gap=DEFAULT_TURN_MERGE_GAP_MS)
        self.assertEqual(len(turns), 1)
        processor, turns = self.run_processor(pcm, gap=DEFAULT_TURN_MERGE_GAP_MS, finish=False)
        processor.close()
        processor.release_pending()
        self.assertEqual(turns, [])

    def test_gap_is_validated_by_live_config(self):
        for invalid in (-20, 5001, 700.0, None):
            with self.subTest(invalid=invalid), self.assertRaises(LiveAudioError):
                LiveConfig(
                    "x",
                    transcriber=ScriptedTranscriber(()),
                    diarizer=ScriptedDiarizer(),
                    turn_merge_gap_ms=invalid,
                )


class CountingProvider:
    """The fixture rule, counting decisions; stands in for opted-in Jev."""

    def __init__(self, built, **options):
        self.requests = 0
        self.states = []
        built.append(self)

    def decide(self, state):
        self.requests += 1
        self.states.append(state)
        return MockProvider().decide(state)


class DecisionPathTests(unittest.TestCase):
    """`listen` over a demo replay: the merged turn reaches the decision exactly once."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        asset = self.root / "asset"
        asset.touch()
        self.demo = self.root / "authored.wav"
        with wave.open(str(self.demo), "wb") as file:
            file.setnchannels(1)
            file.setsampwidth(2)
            file.setframerate(16000)
            file.writeframes(SPLIT_QUESTION)
        self.base = {
            key: str(asset)
            for key in (
                "whisper_executable",
                "whisper_model",
                "diarization_library",
                "diarization_model",
                "microphone_helper",
            )
        }
        self.base["demo_audio"] = str(self.demo)
        self.base["addressing"] = {"names": ["Haili"], "variants": {"Haili": ["Hailey"]}}
        self.config = self.root / "config.json"

    def run_listen(self, texts=TEXTS, pcm=None, switch_ms=None, **extra):
        if pcm is not None:
            with wave.open(str(self.demo), "wb") as file:
                file.setnchannels(1)
                file.setsampwidth(2)
                file.setframerate(16000)
                file.writeframes(pcm)
        self.config.write_text(json.dumps({**self.base, **extra}))
        built = []

        def processor(config, callback):
            backends = {
                "transcriber": ScriptedTranscriber(texts),
                "diarizer": ScriptedDiarizer(switch_ms),
            }
            return LiveProcessor(replace(config, **backends), callback)

        def factory(config, *, event_publisher):
            return PrototypeController(
                config,
                event_publisher=event_publisher,
                processor_factory=processor,
                provider_factory=lambda **options: CountingProvider(built, **options),
            )

        args = Namespace(
            config=self.config,
            mode="demo",
            session_id="merge-decision",
            use_jev=True,
            allow_hosted=True,
            names=None,
        )
        output = io.StringIO()
        with contextlib.redirect_stderr(io.StringIO()):
            code = listen(args, output=output, controller_factory=factory)
        events = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(code, 0)
        return events, built[0]

    @staticmethod
    def of(events, kind):
        return [event for event in events if event["type"] == kind]

    def test_split_question_is_decided_once_on_the_merged_turn(self):
        events, provider = self.run_listen()
        self.assertEqual(provider.requests, 1)
        self.assertEqual(provider.states[0]["current_turn"]["text"], "Hailey, what time is it?")
        transcripts = self.of(events, "transcript")
        self.assertEqual([e["turn"]["text"] for e in transcripts], ["Hailey, what time is it?"])
        requests = self.of(events, "request")
        self.assertEqual(len(requests), 1)
        self.assertEqual(
            requests[0]["turn"]["utterance_id"], transcripts[0]["turn"]["utterance_id"]
        )
        self.assertEqual(
            events[0]["addressing"], {"names": ["Haili"], "variants": {"Haili": ["Hailey"]}}
        )

    def test_without_merging_the_same_speech_takes_two_decisions(self):
        events, provider = self.run_listen(turns={"merge_gap_ms": 0})
        self.assertEqual(provider.requests, 2)
        self.assertEqual([e["turn"]["text"] for e in self.of(events, "transcript")], list(TEXTS))

    def test_owner_stop_after_a_pause_still_supersedes_the_request(self):
        # A participant (Speaker A) asks; the owner (Speaker B) says "Wait.", pauses for
        # 1.5 s, then "never mind". Joined, "Wait. never mind" would not be a stop phrase.
        pcm = audio(
            (VOICE, 200),
            (SILENCE, 1500),
            (VOICE, 200),
            (SILENCE, 1500),
            (VOICE, 200),
            (SILENCE, 3000),
        )
        events, provider = self.run_listen(
            texts=("Haili, order a pizza", "Wait.", "never mind"),
            pcm=pcm,
            switch_ms=1000,
            speakers={"owner": ["Speaker B"], "trusted": [], "owner_only": False},
        )
        transcripts = [e["turn"] for e in self.of(events, "transcript")]
        self.assertEqual(
            [(t["text"], t["speaker_id"]) for t in transcripts],
            [
                ("Haili, order a pizza", "Speaker A"),
                ("Wait.", "Speaker B"),
                ("never mind", "Speaker B"),
            ],
        )
        self.assertEqual(provider.requests, 3)
        (request,) = self.of(events, "request")
        self.assertEqual(request["turn"]["utterance_id"], transcripts[0]["utterance_id"])
        (override,) = self.of(events, "override")
        self.assertEqual(override["superseded_request_id"], request["request_id"])
        self.assertEqual(override["by_utterance_id"], transcripts[2]["utterance_id"])

    def test_without_the_stop_break_the_pause_would_hide_the_stop(self):
        # Control for the test above: an ordinary second fragment does join.
        pcm = audio((VOICE, 200), (SILENCE, 1500), (VOICE, 200), (SILENCE, 3000))
        events, _ = self.run_listen(texts=("Wait.", "never mind me"), pcm=pcm)
        texts = [e["turn"]["text"] for e in self.of(events, "transcript")]
        self.assertEqual(texts, ["Wait. never mind me"])

    def test_config_section_is_validated(self):
        self.config.write_text(json.dumps(self.base))
        self.assertEqual(PrototypeConfig.load(self.config).turn_merge_gap_ms, 2000)
        self.config.write_text(json.dumps({**self.base, "turns": {"merge_gap_ms": 800}}))
        self.assertEqual(PrototypeConfig.load(self.config).turn_merge_gap_ms, 800)
        for invalid in (
            {"turns": {"merge_gap_ms": -1}},
            {"turns": {"merge_gap_ms": 5001}},
            {"turns": {"merge_gap_ms": "800"}},
            {"turns": {"merge_gap_ms": True}},
            {"turns": {"gap": 800}},
            {"turns": None},
            {"turns": [800]},
        ):
            self.config.write_text(json.dumps({**self.base, **invalid}))
            with self.subTest(invalid=invalid), self.assertRaises(PrototypeError):
                PrototypeConfig.load(self.config)


if __name__ == "__main__":
    unittest.main()
