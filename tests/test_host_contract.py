"""Host key contracts: authored config and inert asset placeholders; no speech/runtime."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from rightyo.contracts import Dismissal
from rightyo.prototype import PrototypeConfig
from rightyo.tool_events import SpeechEvents, encode_json
from scripts.emit_started_event import started_event

# Vendored wire key sets from hailing-station origin/main at
# bd9897a22f5a24590491fa7c7826b6a92f874778 (snake case on the wire):
# Sources/HailDaemonKit/Input/RightyoInputEvent.swift,
#   RightyoInputEvent.Dismissal.keys / init(from:), and supportsRequestOutcomes(_:).
# Sources/HailDaemonKit/Input/RightyoInputFormedRequest.swift,
#   RightyoInputEvent.RequestForming.init(from:).
# Update producer, these sets AND the host decoder together when changing a closed
# object; do not derive these sets from RightyO serializers (that hides regressions).
# Conversation/addressing/acknowledgement are raw JSON; Capabilities and Decision use
# synthesized Codable decoding, which ignores unknown keys. They are not closed sets.
HOST_CLOSED_KEYS = {
    "dismissal": frozenset({"version", "window_ms", "cooldown_ms", "cooldown_min_confidence"}),
    "request_forming": frozenset({"kind"}),
    "request_outcomes": frozenset({"version"}),
}


def maximal_config(asset: Path) -> dict:
    """Every optional config key, enabled; all paths refer only to authored placeholders.

    Never run this config as a listener: the empty asset is not a model or executable.
    Hosted decision opt-in is parsed only to exercise the attention advertisement.
    """
    path = str(asset)
    return {
        **dict.fromkeys(
            (
                "whisper_executable",
                "whisper_model",
                "diarization_library",
                "diarization_model",
                "microphone_helper",
                "demo_audio",
            ),
            path,
        ),
        "transcriber": {"kind": "whisper.cpp"},
        "diarizer": {"kind": "nemotron.cpp"},
        "addressing": {"names": ["Helper"], "variants": {"Helper": ["Helpah"]}},
        "other_agents": {"names": ["Other"], "variants": {"Other": ["Otha"]}},
        "speakers": {
            "owner": ["authored-owner"],
            "trusted": ["authored-guest"],
            "owner_only": True,
            "stop_phrases": ["stop"],
            "source": "configured",
        },
        "session_budget_seconds": 10,
        "request_former": {"kind": "template"},
        "decision": {
            "provider": "jev",
            "allow_hosted": True,
            "max_requests": 1,
            "scene": "Authored contract fixture only.",
            "attend_min_probability": 0.9,
            "other_answered_blocks": True,
            "state_log": str(asset.parent / "unused.jsonl"),
            "request_context": True,
        },
        "turns": {
            "merge_gap_ms": 1000,
            "reply_wait_ms": 1000,
            "edge_attribution_ms": 1000,
            "tail_join_ms": 1000,
            "hold_unfinished": True,
            "word_label_coverage": 0.9,
            "complete_hold_ms": 1000,
            "collapse_restarts": True,
            "bridge_backchannels": True,
            "wake_join": True,
            "owner_pause_ms": 1000,
        },
        "dismissal": {
            "window_ms": 12345,
            "cooldown_ms": 45678,
            "cooldown_min_confidence": 0.91,
            "min_attend": 0.6,
        },
        "end_of_turn": {
            "enabled": True,
            "python": path,
            "model": path,
            "threshold": 0.6,
            "silence_ms": 200,
            "threads": 1,
        },
        "conversation": {
            "window_ms": 12000,
            "follow_up_min_probability": 0.5,
            "closing_phrases": ["all done"],
            "appendix_ms": 1000,
            "require_cue": True,
            "cue_min_attend": 0.8,
            "continuation_cue": True,
            "cue_when_others_ms": 30000,
        },
        "acknowledgement": {
            "min_confidence": 0.8,
            "dedup_window_ms": 8000,
            "follow_up_min_confidence": 0.9,
        },
        "meeting": {"ask_anywhere": True},
        "speaker_id": {
            "enabled": True,
            "python": path,
            "model": path,
            "store": str(asset.parent / "unused-store"),
            "bind_threshold": 0.6,
            "tentative_threshold": 0.45,
            "min_turn_seconds": 1.0,
            "threads": 1,
            "live": True,
            "bind_min_seconds": 3.0,
            "roles": True,
            "enrolled_follow_up_min_probability": 0.4,
            "emit_names": True,
            "entry_threshold": 0.35,
            "entry_min_seconds": 0.5,
            "entry_remainder_threshold": 0.35,
            "entry_gap_max_seconds": 0.6,
            "entry_window_seconds": 0.25,
        },
        "audio_retention": {"speaker_ring_seconds": 60, "timeline_window_seconds": 60},
    }


class HostContractTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        asset = self.root / "inert-asset.txt"
        asset.touch()
        self.config_path = self.root / "maximal.json"
        self.config_path.write_text(json.dumps(maximal_config(asset)), encoding="utf-8")
        self.config = PrototypeConfig.load(self.config_path)

    def test_maximal_started_event_matches_host_closed_objects(self):
        # A started-only fixture must never reach runtime, capture, enrollment or network.
        with (
            patch("rightyo.prototype.PrototypeController.start", side_effect=AssertionError),
            patch("rightyo.speaker_id.SpeakerEmbedder", side_effect=AssertionError),
            patch("socket.socket", side_effect=AssertionError),
            patch("subprocess.Popen", side_effect=AssertionError),
        ):
            event = started_event(self.config)
        self.assertEqual(event["type"], "session")
        self.assertEqual(event["phase"], "started")
        # Prevent the maximal fixture silently losing a feature and weakening coverage.
        self.assertEqual(
            set(event),
            {
                "schema_version",
                "type",
                "session_id",
                "sequence",
                "emitted_at_ms",
                "phase",
                "capabilities",
                "addressing",
                "speaker_names",
                "owner_voice",
                "request_forming",
                "speech",
                "audio_input",
                "dismissal",
                "conversation",
                "acknowledgement",
                "duck",
                "meeting",
                "request_outcomes",
            },
        )
        self.assertEqual(self.config.dismissal.min_attend, 0.6)
        self.assertEqual(event["conversation"]["cue_when_others_ms"], 30000)
        self.assertEqual(event["duck"], {"version": 3, "private": True})
        for name, keys in HOST_CLOSED_KEYS.items():
            with self.subTest(object=name):
                self.assertEqual(set(event[name]), keys)

    def test_dismissal_wire_bytes_match_pre_284_for_every_bar(self):
        for window, cooldown, confidence in ((1, 0, 0), (10000, 30000, 0.9), (60000, 600000, 1)):
            # Pre-#284 field order and representation, independent of today's serializer.
            expected = (
                '{"version": 1, "window_ms": '
                + str(window)
                + ', "cooldown_ms": '
                + str(cooldown)
                + ', "cooldown_min_confidence": '
                + str(confidence)
                + "}"
            ).encode("ascii")
            for bar in (None, 0, 0.6, 1):
                with self.subTest(window=window, cooldown=cooldown, confidence=confidence, bar=bar):
                    events = SpeechEvents()
                    events.start("authored", dismissal=Dismissal(window, cooldown, confidence, bar))
                    self.assertEqual(
                        encode_json(events.drain()[0]["dismissal"]).encode("ascii"), expected
                    )
        events = SpeechEvents()
        events.start("absent")
        self.assertNotIn("dismissal", events.drain()[0])

    def test_emit_cli_writes_the_real_maximal_event(self):
        output = self.root / "started.json"
        repo = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                str(repo / "scripts/emit_started_event.py"),
                "--config",
                str(self.config_path),
                "--out",
                str(output),
            ],
            cwd=repo,
            env={**os.environ, "PYTHONPATH": str(repo / "src")},
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        self.assertEqual(result.stdout, "")
        self.assertEqual(output.read_text(), encode_json(started_event(self.config)) + "\n")
        self.assertFalse((self.root / "unused.jsonl").exists())
        self.assertFalse((self.root / "unused-store").exists())

    def test_min_attend_stays_in_local_configuration_serialization(self):
        for bar in (None, 0, 0.6, 1):
            with self.subTest(bar=bar):
                dismissal = replace(self.config.dismissal, min_attend=bar)
                raw = dismissal.to_dict()
                raw.pop("version")
                self.assertEqual(Dismissal.from_dict(raw), dismissal)
