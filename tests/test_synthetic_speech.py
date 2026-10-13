"""Authored PCM/backend qualification only: no speech, assets, devices or services."""

import array
import io
import json
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from rightyo.live_audio import BYTES_PER_MS, LiveAudioError, LiveConfig, LiveProcessor
from rightyo.prototype import PrototypeConfig, PrototypeController, PrototypeError
from rightyo.providers import Diarizer, Transcriber
from rightyo.speech_backends import (
    MAX_SYNTHETIC_AUDIO_MS,
    SYNTHETIC_KIND,
    SyntheticAuthoredDiarizer,
    SyntheticAuthoredTranscriber,
    describe,
    diarizer_factory,
    diarizer_spec,
    speech_summary,
    transcriber_factory,
    transcriber_spec,
)
from rightyo.tool_events import SpeechEvents

SESSION = "8cbbf275-cabb-4f99-8ae4-f4d610450a91"
SCRIPT = ["Authored first synthetic turn.", "Authored second synthetic turn."]
# A generated constant-amplitude signal plus sufficient real silence to endpoint it.
# This is neither a recording nor a voice, and is generated only in memory.
TONE = array.array("h", [5000] * 16000).tobytes()
UTTERANCE = TONE + bytes(2 * 16000 * 2)


def config_dict(script=None):
    return {
        "transcriber": {"kind": SYNTHETIC_KIND, "utterances": SCRIPT if script is None else script},
        "diarizer": {"kind": SYNTHETIC_KIND},
        "turns": {"merge_gap_ms": 0, "reply_wait_ms": 0},
    }


def live_config(**options):
    options.setdefault("transcriber", transcriber_factory(config_dict()["transcriber"]))
    options.setdefault("diarizer", diarizer_factory(config_dict()["diarizer"]))
    options.setdefault("provenance", "synthetic")
    return LiveConfig(session_id=SESSION, **options)


class SyntheticBackendTests(unittest.TestCase):
    def test_strict_finite_script_validation_and_encoded_byte_budget(self):
        for script in (None, [], [" "], ["x\n"], ["\ud800"], ["\u2028"], ["x" * 1025], ["x"] * 33):
            with self.subTest(script_type=type(script).__name__, count=len(script or [])):
                with self.assertRaises(LiveAudioError):
                    transcriber_spec({"kind": SYNTHETIC_KIND, "utterances": script})
        with self.assertRaises(LiveAudioError):
            transcriber_spec({"kind": SYNTHETIC_KIND, "utterances": ["\U0001f642" * 1024] * 2})
        for spec in (
            {"kind": SYNTHETIC_KIND, "utterances": ["Text."], "loop": True},
            {"kind": SYNTHETIC_KIND, "utterances": ["Text."], "endpoint": "https://example.test"},
        ):
            with self.assertRaises(LiveAudioError):
                transcriber_spec(spec)
        with self.assertRaises(LiveAudioError):
            diarizer_spec({"kind": SYNTHETIC_KIND, "speaker": 1})
        value = {"kind": SYNTHETIC_KIND, "utterances": ["Unicode authored \U0001f642."]}
        validated = transcriber_spec(value)
        value["utterances"].clear()
        self.assertEqual(validated["utterances"], ["Unicode authored \U0001f642."])

    def test_protocols_explicit_mock_ids_and_factory_provenance_guard(self):
        transcriber = transcriber_factory(config_dict()["transcriber"])
        diarizer = diarizer_factory(config_dict()["diarizer"])
        self.assertIsInstance(transcriber(live_config()), Transcriber)
        self.assertIsInstance(diarizer(live_config()), Diarizer)
        for provenance in ("live-microphone", "recorded-file", "causal-replay"):
            with self.subTest(provenance=provenance):
                config = live_config(provenance=provenance)
                with self.assertRaises(LiveAudioError):
                    transcriber(config)
                with self.assertRaises(LiveAudioError):
                    diarizer(config)
        summary = speech_summary(config_dict()["transcriber"], config_dict()["diarizer"])
        self.assertEqual(summary["transcriber"]["id"], "synthetic-authored-v1")
        self.assertEqual(summary["diarizer"]["id"], "synthetic-authored-v1")
        self.assertFalse(describe(config_dict()["diarizer"])["hosted"])

    def test_one_text_per_pcm_utterance_exhaustion_and_cancellation(self):
        cancelled = threading.Event()
        transcriber = SyntheticAuthoredTranscriber(live_config(cancelled=cancelled.is_set), SCRIPT)
        self.assertEqual(transcriber.transcribe(TONE), [
            {"text": SCRIPT[0], "start_ms": 0, "end_ms": 1000}
        ])
        cancelled.set()
        with self.assertRaises(LiveAudioError):
            transcriber.transcribe(TONE)
        cancelled.clear()
        self.assertEqual(transcriber.transcribe(TONE)[0]["text"], SCRIPT[1])
        with self.assertRaisesRegex(LiveAudioError, "exhausted"):
            transcriber.transcribe(TONE)
        for invalid in (b"", b"x", bytearray(TONE), bytes(16001 * BYTES_PER_MS)):
            with self.assertRaises(LiveAudioError):
                SyntheticAuthoredTranscriber(live_config(), SCRIPT).transcribe(invalid)

    def test_diarizer_constant_space_received_timeline_and_finite_ceiling(self):
        diarizer = SyntheticAuthoredDiarizer(live_config())
        self.assertEqual(diarizer.segments(), [])
        pcm = bytes(32000)
        for _ in range(MAX_SYNTHETIC_AUDIO_MS // 1000):
            diarizer.push(pcm)
        self.assertEqual(diarizer.segments(), [
            {"start_ms": 0, "end_ms": MAX_SYNTHETIC_AUDIO_MS, "speaker": 1}
        ])
        self.assertTrue(all(
            not isinstance(value, (bytes, bytearray, list)) for value in vars(diarizer).values()
        ))
        with self.assertRaisesRegex(LiveAudioError, "budget"):
            diarizer.push(bytes(640))
        self.assertEqual(diarizer.finish(), diarizer.segments())
        diarizer.close()
        self.assertEqual(diarizer.segments(), [])
        self.assertEqual(diarizer._received_bytes, 0)
        with self.assertRaises(LiveAudioError):
            diarizer.push(pcm)

    def test_real_processor_uses_observed_pcm_endpoints_not_script_timer(self):
        turns = []
        config = live_config(
            transcriber=transcriber_factory(config_dict()["transcriber"]),
            diarizer=diarizer_factory(config_dict()["diarizer"]),
            turn_merge_gap_ms=0,
            reply_wait_ms=0,
        )
        processor = LiveProcessor(config, turns.append)
        self.addCleanup(processor.close)
        processor.push_pcm16(bytes(32000))
        self.assertEqual(turns, [])  # Silence consumes no authored text.
        for pcm in (UTTERANCE, UTTERANCE):
            for start in range(0, len(pcm), 6400):
                processor.push_pcm16(pcm[start:start + 6400])
        processor.finish()
        self.assertEqual([turn.text for turn in turns], SCRIPT)
        self.assertEqual([turn.utterance_id for turn in turns], ["live-1", "live-2"])
        self.assertTrue(all(turn.provenance == "synthetic" for turn in turns))
        self.assertTrue(all(turn.speaker_provenance == "authored-fixture" for turn in turns))
        self.assertTrue(all(turn.speaker_id == "Speaker A" for turn in turns))
        self.assertGreaterEqual(turns[0].start_ms, 700)
        self.assertGreater(turns[1].start_ms, turns[0].end_ms)
        self.assertLessEqual(turns[-1].end_ms, processor.received_ms)


class SyntheticPrototypeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "synthetic.json"

    def load(self, value=None):
        self.path.write_text(json.dumps(config_dict() if value is None else value))
        return PrototypeConfig.load(self.path)

    def test_config_needs_no_assets_and_rejects_mixed_or_native_extras(self):
        config = self.load()
        self.assertTrue(config.synthetic_speech)
        self.assertIsNone(config.microphone_helper)
        self.assertIsNone(config.whisper_model)
        self.assertIsNone(config.diarization_model)
        for key, value in (
            ("diarizer", {"kind": "nemotron.cpp"}),
            ("transcriber", {"kind": "whisper.cpp"}),
            ("end_of_turn", {"enabled": False}),
            ("speaker_id", {"enabled": False}),
            ("decision", {"provider": "jev", "allow_hosted": True}),
        ):
            raw = config_dict()
            raw[key] = value
            with self.subTest(key=key), self.assertRaises(PrototypeError):
                self.load(raw)

    def test_rejects_live_sources_before_any_factory_runs(self):
        for mode, provenance, hosted in (
            ("microphone", "synthetic", False),
            ("demo", "synthetic", False),
            ("stdin", "live-microphone", False),
            ("stdin", "recorded-file", False),
            ("stdin", "causal-replay", False),
            ("stdin", "synthetic", True),
        ):
            factories = [
                Mock(side_effect=AssertionError("A backend must not start")) for _ in range(5)
            ]
            controller = PrototypeController(
                self.load(), audio_input=io.BytesIO(UTTERANCE), audio_provenance=provenance,
                processor_factory=factories[0], capture_factory=factories[1],
                provider_factory=factories[2], end_of_turn_factory=factories[3],
                speaker_id_factory=factories[4],
            )
            try:
                with self.subTest(mode=mode, provenance=provenance, hosted=hosted):
                    with self.assertRaises(PrototypeError):
                        controller.start({"mode": mode, "use_jev": hosted})
                    self.assertTrue(all(factory.call_count == 0 for factory in factories))
                    self.assertIsNone(controller._audio_thread)
            finally:
                controller.close()

    def test_direct_config_native_optional_components_cannot_bypass_guard(self):
        for key in ("end_of_turn", "speaker_id"):
            config = replace(self.load(), **{key: object()})
            controller = PrototypeController(
                config, audio_input=io.BytesIO(), audio_provenance="synthetic"
            )
            try:
                with self.assertRaises(PrototypeError):
                    controller.start({"mode": "stdin"})
            finally:
                controller.close()

    def run_controller(self, config, pcm):
        events = SpeechEvents()
        controller = PrototypeController(
            config, event_publisher=events, audio_input=io.BytesIO(pcm),
            audio_provenance="synthetic",
        )
        try:
            controller.start({"mode": "stdin", "session_id": SESSION})
            controller._audio_thread.join(timeout=5)
            self.assertFalse(controller._audio_thread.is_alive())
            state = controller.snapshot()
            return state, events.drain()
        finally:
            controller.close()

    def test_controller_runs_real_pipeline_without_native_or_service_factories(self):
        with (
            patch("rightyo.live_audio.WhisperCppTranscriber", side_effect=AssertionError),
            patch("rightyo.live_audio._Diarizer", side_effect=AssertionError),
            patch(
                "rightyo.speech_backends.OpenAICompatibleTranscriber", side_effect=AssertionError
            ),
            patch("rightyo.speech_backends.DeepgramDiarizer", side_effect=AssertionError),
            patch("rightyo.prototype.MacMicrophoneCapture", side_effect=AssertionError),
        ):
            state, events = self.run_controller(self.load(), UTTERANCE * 2)
        self.assertEqual(state["phase"], "complete")
        turns = [event["turn"] for event in events if event["type"] == "transcript"]
        self.assertEqual([turn["text"] for turn in turns], SCRIPT)
        self.assertTrue(all(turn["provenance"] == "synthetic" for turn in turns))
        self.assertFalse(any(event["type"] == "request" for event in events))

    def test_script_exhaustion_is_fatal_without_extra_fabricated_turn(self):
        state, events = self.run_controller(self.load(config_dict([SCRIPT[0]])), UTTERANCE * 2)
        self.assertEqual(state["phase"], "error")
        self.assertEqual(
            [event["phase"] for event in events if event["type"] == "session"], ["error"]
        )
        self.assertFalse(any(event["type"] == "transcript" for event in events))


if __name__ == "__main__":
    unittest.main()
