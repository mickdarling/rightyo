"""Tag-only self exclusion with authored turns and inert workers."""

import hashlib
import io
import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr
from dataclasses import replace
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, patch

from test_conversation import SESSION, decided, turn
from test_duck_mode import HOSTED_CONSUMERS, enrolled, heard
from test_live_speaker_id import ENTRIES, ScriptedEmbedder, settings, unit, wait_for

from rightyo.contracts import SpeakerPriority
from rightyo.live_audio import LiveConfig, LiveProcessor
from rightyo.live_speaker_id import EnrolledRoles, ShadowSpeakerId
from rightyo.prototype import PCM_BYTES_PER_MS, PrototypeConfig, PrototypeController, PrototypeError
from rightyo.self_voice import HISTORY_MS, SelfVoice, SelfVoiceConfig
from rightyo.tool import _read_control
from rightyo.tool_events import (
    PRIVACY_ENTRY,
    PRIVACY_EXIT,
    PRIVACY_STATUS,
    SpeechEvents,
    encode_json,
)
from scripts.emit_started_event import started_event

WINDOW = dict(type="reply_window", reply="r-1_", from_sample=0, to_sample=None, tail_ms=400)
ECHO = dict(type="self_echo", reply="r-1_", from_sample=0, to_sample=1600, confidence=None)
ECHO["class"] = "echo_only"


def read_lines(controller, *records):
    with tempfile.TemporaryFile() as fd, redirect_stderr(io.StringIO()) as notes:
        for record in records:
            line = record if isinstance(record, bytes) else json.dumps(record).encode("ascii")
            fd.write(line + b"\n")
        fd.seek(0)
        _read_control(os.dup(fd.fileno()), controller)
        return notes.getvalue().splitlines()


class ControlTests(unittest.TestCase):
    def test_valid_controls(self):
        ctl = Mock()
        records = [
            WINDOW,
            {**WINDOW, "to_sample": 0, "tail_ms": 2000},
            ECHO,
            {**ECHO, "class": "mixed", "confidence": 1},
        ]
        self.assertEqual(read_lines(ctl, *records), [])
        self.assertEqual(
            ctl.reply_window.call_args_list,
            [
                (("r-1_", 0, None, 400),),
                (("r-1_", 0, 0, 2000),),
            ],
        )
        self.assertEqual(
            ctl.self_echo.call_args_list,
            [
                (("r-1_", 0, 1600, "echo_only", None),),
                (("r-1_", 0, 1600, "mixed", 1),),
            ],
        )

    def test_invalid_lines_skip_and_resume(self):
        records = []
        for record in (WINDOW, ECHO):
            records.extend({k: v for k, v in record.items() if k != key} for key in record)
            for key, values in {
                "extra": [0],
                "reply": ["", ".", "a" * 65, "é", 1],
                "from_sample": [-1, True, 0.5],
                "to_sample": [-1, False, 0.5],
                "type": ["unknown", "self_echo" if record is WINDOW else "reply_window"],
            }.items():
                records.extend({**record, key: value} for value in values)
        for key, values in {
            "to_sample": [None, 0],
            "class": ["other", []],
            "confidence": [-0.1, 1.1, False, "0", float("nan")],
        }.items():
            records.extend({**ECHO, key: value} for value in values)
        records.extend({**WINDOW, "tail_ms": v} for v in (-1, 2001, False, 0.5))
        records.extend(
            (
                {**WINDOW, "from_sample": 2, "to_sample": 1},
                b" " * 2000 + json.dumps(WINDOW).encode(),
            )
        )
        ctl = Mock()
        notes = read_lines(ctl, *records, {"duck": True})
        self.assertEqual(
            notes,
            ["rightyo: invalid control line; skipped"] * (len(records) - 1)
            + ["rightyo: control line too long; skipped"],
        )
        ctl.duck.assert_called_once_with(True)
        ctl.reply_window.assert_not_called()
        ctl.self_echo.assert_not_called()


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.store = SelfVoice(SelfVoiceConfig(enabled=True))
        self.window = self.store.reply_window
        self.tag = self.store.self_echo
        self.classify = self.store.classify

    def test_close_and_tail(self):
        for opened in (False, True):
            with self.subTest(opened=opened):
                self.setUp()
                if opened:
                    self.window("r", 100, None, 100, 100)
                self.window("r", 100, 200, 25, 201)
                self.window("r", 100, 300, 50, 202)
                self.tag("r", 200, 225, "echo_only", 225)
                self.assertEqual(self.classify(200, 225, 225), ("echo_only", 1))
                self.assertEqual(self.classify(225, 226, 225)[0], "outside")

    def test_refresh_and_expiry(self):
        self.window("r", 100, None, 0, 100)
        self.assertEqual(self.classify(100, 201, 200)[0], "outside")
        self.window("r", 100, None, 0, 60_000)
        self.assertEqual(self.classify(150_000, 180_000, 180_000)[0], "partial")
        self.window("r", 100, None, 0, 181_000)
        self.assertEqual(self.classify(180_000, 180_001, 181_000)[0], "outside")
        self.window("s", 182_000, None, 50, 182_000)
        self.assertEqual(self.classify(302_000, 302_050, 310_000)[0], "partial")
        self.assertEqual(self.classify(302_050, 302_051, 310_000)[0], "outside")

    def test_delayed_open_and_refresh(self):
        self.window("r", 0, None, 0, 60_000)
        self.assertEqual(self.store.windows["r"].expires, 180_000)
        self.window("r", 0, None, 0, 120_000)
        self.tag("r", 120_000, 121_000, "echo_only", 121_000)
        self.assertEqual(self.classify(120_000, 121_000, 121_000)[0], "echo_only")
        self.assertEqual(self.classify(240_000, 240_001, 241_000)[0], "outside")
        self.window("r", 0, None, 0, 241_000)
        self.tag("r", 240_000, 240_001, "echo_only", 241_000)
        self.assertEqual(self.store.coverage(240_000, 240_001), (False, 0))
        self.assertEqual(self.classify(240_000, 240_001, 241_000)[0], "outside")
        self.window("r", 0, 241_000, 0, 241_000)
        self.window("r", 0, None, 0, 242_000)
        self.assertEqual(self.classify(241_000, 241_001, 242_000)[0], "outside")

    def test_window_cap_and_eviction(self):
        for idx in range(9):
            self.window(str(idx), 0, 100, 0, 100)
            self.tag(str(idx), idx, idx + 1, "echo_only", 100)
        self.assertEqual(list(self.store.windows), list(map(str, range(1, 9))))
        self.assertEqual([s.reply for s in self.store.spans], list(self.store.windows))

    def test_lateness(self):
        self.window("r", 0, 200, 0, 200)
        self.tag("r", 0, 99, "echo_only", 5100)
        self.assertFalse(self.store.spans)
        self.tag("r", 0, 100, "echo_only", 5100)
        self.assertEqual(self.classify(0, 100, 5100), ("echo_only", 1))
        self.tag("r", 0, 99, "mixed", 5100)
        self.assertEqual(self.classify(0, 100, 5100)[0], "mixed")

    def test_outside_window(self):
        self.window("r", 10, 110, 0, 110)
        self.tag("s", 10, 110, "echo_only", 110)
        self.assertEqual(self.store.spans, [])
        self.tag("r", 0, 10, "mixed", 110)
        self.tag("r", 10, 110, "echo_only", 110)
        self.assertEqual(self.store.coverage(0, 10), (True, 0))
        self.assertEqual(self.classify(0, 110, 110)[0], "outside")

    def test_merge_and_union(self):
        for reply in ("r", "s"):
            self.window(reply, 0, 100, 0, 100)
        for start, end in ((0, 20), (40, 70), (20, 45)):
            self.tag("r", start, end, "echo_only", 100)
        self.tag("s", 10, 65, "echo_only", 100)
        self.assertEqual(len(self.store.spans), 2)
        self.assertEqual(self.classify(0, 100, 100), ("partial", 0.7))
        self.tag("s", 65, 85, "echo_only", 100)
        self.assertEqual(self.classify(0, 100, 100), ("echo_only", 0.85))
        self.tag("s", 80, 81, "mixed", 100)
        self.assertEqual(self.classify(0, 100, 100), ("mixed", 0.85))

    def test_mixed_straddles_window_edges(self):
        for start, end, outside in ((90, 110, 90), (190, 210, 200)):
            with self.subTest(start=start):
                self.setUp()
                self.window("r", 100, 200, 0, 200)
                self.window("s", 90, 210, 0, 210)
                self.tag("s", 90, 210, "echo_only", 210)
                self.tag("r", start, end, "mixed", 210)
                self.assertEqual(self.classify(start, end, 210)[0], "mixed")
                self.assertEqual(
                    self.classify(outside, outside + 10, 210)[0],
                    "mixed",
                )

    def test_retracted_tail_keeps_mixed(self):
        self.window("r", 0, None, 100, 100)
        self.window("s", 0, 150, 0, 100)
        self.tag("r", 0, 150, "echo_only", 100)
        self.tag("r", 99, 120, "mixed", 100)
        self.window("r", 0, 100, 5, 100)
        self.assertEqual(self.classify(105, 120, 120), ("mixed", 0))
        self.assertEqual(self.classify(104, 105, 120), ("mixed", 1))

    def test_removing_window_keeps_existing_mixed(self):
        for removal in ("window_eviction", "empty_close"):
            with self.subTest(removal=removal):
                self.setUp()
                self.window("r", 0, None, 0, 100)
                self.window("s", 0, 100, 0, 100)
                self.tag("r", 0, 100, "echo_only", 100)
                self.tag("r", 0, 50, "mixed", 100)
                self.tag("s", 0, 80, "echo_only", 100)
                if removal == "empty_close":
                    self.window("r", 0, 0, 0, 100)
                else:
                    for idx in range(7):
                        self.window(str(idx), 0, 100, 0, 100)
                self.assertEqual(self.classify(0, 100, 100), ("mixed", 0.8))

    def test_span_eviction_removes_overlapping_echo(self):
        self.window("r", 0, 2000, 0, 2000)
        self.tag("r", 0, 10, "mixed", 2000)
        self.tag("r", 1, 9, "echo_only", 2000)
        for idx in range(511):
            self.tag("r", 20 + 2 * idx, 21 + 2 * idx, "echo_only", 2000)
        self.assertEqual(self.classify(1, 9, 2000), ("partial", 0))

    def test_pruned_mixed_prefix_removes_merged_echo(self):
        self.window("r", 0, 1000, 0, 1000)
        self.tag("r", 0, 20, "mixed", 1000)
        self.tag("r", 0, 50, "echo_only", 1000)
        self.tag("r", 50, 1000, "echo_only", 1000)
        self.store.advance(HISTORY_MS + 21)
        self.assertEqual(self.classify(0, 1000, HISTORY_MS + 21), ("partial", 0))

    def test_lost_mixed_prefix_with_adjacent_echo(self):
        for removal in ("span_cap", "age_pruning"):
            with self.subTest(removal=removal):
                self.setUp()
                received = 2000
                self.window("r", 0, 500, 0, received)
                self.window("s", 0, 2000, 0, received)
                self.tag("r", 0, 100, "mixed", received)
                self.tag("s", 100, 500, "echo_only", received)
                self.assertEqual(self.classify(0, 500, received), ("mixed", 0.8))
                if removal == "span_cap":
                    for idx in range(511):
                        self.tag("s", 501 + 2 * idx, 502 + 2 * idx, "echo_only", received)
                else:
                    received = HISTORY_MS + 101
                    self.store.advance(received)
                self.assertEqual(self.classify(0, 500, received), ("partial", 0.8))
                self.assertEqual(self.classify(100, 500, received), ("echo_only", 1))

    def test_future_close_and_echo_are_dropped(self):
        self.window("r", 0, None, 0, 0)
        self.window("r", 0, 35_001, 0, 0)
        self.assertIsNone(self.store.windows["r"].end)
        self.tag("r", 0, 35_001, "echo_only", 0)
        self.assertEqual(self.classify(0, 100, 100), ("partial", 0))
        self.tag("r", 0, 35_000, "echo_only", 0)
        self.tag("r", 0, 35_001, "mixed", 0)
        self.assertEqual(self.classify(0, 100, 100), ("mixed", 1))
        self.assertEqual(self.store.windows_dropped, 1)

    def test_span_cap_and_pruning(self):
        self.window("r", 0, 2000, 0, 0)
        for idx in range(513):
            self.tag("r", idx * 2, idx * 2 + 1, "echo_only", 1100)
        self.assertEqual(len(self.store.spans), 512)
        self.assertEqual(self.store.spans[0].start, 2)
        self.store.advance(HISTORY_MS + 2000)
        self.assertEqual(self.store.spans, [])


class SessionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "config.json"
        asset = self.path.with_suffix(".asset")
        asset.touch()
        self.base = dict.fromkeys(
            (
                "microphone_helper",
                "whisper_executable",
                "whisper_model",
                "diarization_library",
                "diarization_model",
            ),
            str(asset),
        )
        self.base.update(
            addressing={"names": ["Unit"]},
            speakers={"owner": ["a"]},
            conversation={"cue_when_others_ms": 5000},
            dismissal={},
        )

    def config(self, **sections):
        self.path.write_text(json.dumps({**self.base, **sections}))
        return PrototypeConfig.load(self.path)

    def session(self, *, allow_hosted_speech=False, **sections):
        config = self.config(**sections)
        events = SpeechEvents(request_control=True)
        self.notes = []
        with patch("threading.Thread.start"):
            self.ctl = PrototypeController(
                config,
                event_publisher=events,
                allow_hosted_speech=allow_hosted_speech,
                report=self.notes.append,
                processor_factory=Mock(),
                provider_factory=Mock(),
                audio_input=io.BytesIO(),
                audio_provenance="synthetic",
            )
            self.ctl.start({"mode": "stdin", "session_id": SESSION, "use_jev": True})
        self.ctl._audio_thread = None
        self.addCleanup(self.ctl.close)
        self.events = events
        self.started = events.drain()[0]
        self.shadow = SimpleNamespace(turn=Mock())
        return self.ctl

    def accept(self, current, *, decide=True):
        ctl = self.ctl
        ctl._received_ms = current.end_ms
        ctl._received_bytes = current.end_ms * PCM_BYTES_PER_MS
        ctl._turn(ctl._generation, ctl._decision_queue, ctl._memory, self.shadow, current)
        if decide and not ctl._decision_queue.empty():
            self.assertEqual(ctl._decision_queue.get_nowait(), current)
            self.events.decision(decided(current, "attend"), current.end_ms)
        return self.events.drain()

    def tag(self, start, end, kind="echo_only"):
        self.ctl.self_echo("r", start * 16, end * 16, kind)

    def test_config(self):
        for raw in (
            {},
            {"enabled": False},
            {"transcripts": "suppress", "echo_only_min_fraction": 0.8},
        ):
            self.assertEqual(self.config(self_voice=raw).self_voice.to_config(), {})
        for raw in (
            {"enabled": True, "echo_only_min_fraction": 0.5},
            {"transcripts": "publish"},
            {"echo_only_min_fraction": 1},
        ):
            section = self.config(self_voice=raw).self_voice
            self.assertEqual(section.to_config(), {"self_voice": raw})
            self.assertEqual(self.config(**section.to_config()).self_voice, section)
        for bad in (
            None,
            [],
            {"unknown": True},
            {"enabled": 1},
            {"transcripts": "other"},
            {"echo_only_min_fraction": False},
            {"echo_only_min_fraction": 0.49},
            {"echo_only_min_fraction": 1.01},
        ):
            with self.subTest(bad=bad), self.assertRaises(PrototypeError):
                self.config(self_voice=bad)

    def test_off_matches_main_bytes(self):
        for sections in ({}, {"self_voice": {"enabled": False}}):
            self.session(**sections)
            self.assertEqual(
                read_lines(self.ctl, WINDOW, ECHO), ["rightyo: invalid control line; skipped"] * 2
            )
            # Check the exact addition to the historical started event before hashing the rest.
            self.assertEqual(
                encode_json(self.started["capabilities"].pop("duck_after_sample")), "1"
            )
            payloads = [self.started]
            for index, text in enumerate(("Unit", SpeakerPriority().stop_phrases[0])):
                payloads += self.accept(
                    turn(str(index), 1000 * index, 1000 * index + 500, text, "a")
                )
            self.events.end("stopped", 2000)
            payloads += self.events.drain()
            # Verify the additive decision fields before checking the historical wire bytes.
            for payload in payloads:
                if "decision" in payload:
                    decision = payload["decision"] = payload["decision"].copy()
                    self.assertEqual(decision.pop("speaker_label"), "a")
                    self.assertEqual(decision.pop("speaker_provenance"), "authored-fixture")
                    self.assertIs(decision.pop("speaker_bound_to_owner"), False)
                    self.assertIsNone(decision.pop("speaker_binding_state"))
                    self.assertEqual(decision.pop("speaker_binding_source"), "configured_label")
            wire = "".join(encode_json(p) + "\n" for p in payloads).encode("ascii")
            # SHA-256 of the identical authored script at origin/main 2ae361c.
            self.assertEqual(
                hashlib.sha256(wire).hexdigest(),
                "d0d228b511d5d4a71b28fb19f9a9af39b17c8798e000e3e6f8c3395d4c7736fc",
            )

    def test_submillisecond_rounding(self):
        ctl = self.session(self_voice={"enabled": True})
        ctl._received_ms = 1001
        store = self.events.self_voice
        ctl.reply_window("r", 1, 16_015, 0)
        ctl.self_echo("r", 1, 16_015, "echo_only")
        self.assertEqual(store.classify(0, 1, 1001)[0], "outside")
        self.assertEqual(store.classify(1000, 1001, 1001)[0], "outside")
        ctl.reply_window("s", 0, 1001 * 16, 0)
        self.assertEqual(store.classify(0, 1, 1001), ("partial", 0))
        self.assertEqual(store.classify(1000, 1001, 1001), ("partial", 0))
        ctl.self_echo("s", 1, 15, "echo_only")
        ctl.reply_window("empty", 1, 15, 0)
        ctl.reply_window("empty", 1, None, 0)
        self.assertNotIn("empty", store.windows)
        self.assertEqual(len(store.spans), 1)
        self.assertEqual(store.classify(1, 1000, 1001)[0], "echo_only")
        ctl.self_echo("r", 8000, 8001, "mixed")
        self.assertEqual(store.classify(500, 501, 1001), ("mixed", 1))

    def test_mixed_at_sample_8001_window_edge(self):
        ctl = self.session(self_voice={"enabled": True})
        ctl.reply_window("r", 0, 8001, 0)
        self.assert_mixed_barge_in(ctl)

    def test_mixed_after_empty_close(self):
        ctl = self.session(self_voice={"enabled": True})
        ctl.reply_window("r", 8000, None, 0)
        ctl.reply_window("r", 8000, 8001, 0)
        self.assert_mixed_barge_in(ctl)

    def test_mixed_for_unseen_reply(self):
        ctl = self.session(self_voice={"enabled": True})
        self.assert_mixed_barge_in(ctl)

    def assert_mixed_barge_in(self, ctl):
        ctl.reply_window("s", 8000, 16_000, 0)
        ctl.self_echo("r", 8000, 8001, "mixed")
        ctl.self_echo("s", 8000, 16_000, "echo_only")
        current = turn("x", 500, 1000, ctl.config.addressing.names[0], "a")
        self.assertIn("request", [e["type"] for e in self.accept(current)])
        self.assertIn("self_voice outcome=person reason=mixed fraction=1.00", self.notes)

    def test_revival_preserves_expired_gap(self):
        for mixed in (True, False):
            with self.subTest(mixed=mixed):
                ctl = self.session(self_voice={"enabled": True})
                ctl.reply_window("r", 0, None, 400)
                ctl._received_ms = 120_500
                if mixed:
                    self.tag(120_400, 120_500, "mixed")
                    self.assertEqual(self.events.self_voice.coverage(120_400, 120_500), (True, 0))
                ctl.reply_window("r", 0, None, 400)
                self.tag(120_500, 120_900)
                current = turn("x", 120_400, 120_900, ctl.config.addressing.names[0], "a")
                self.assertIn("request", [e["type"] for e in self.accept(current)])
                self.assertIn("self_voice outcome=person reason=outside fraction=0.80", self.notes)
                self.assertEqual(
                    self.accept(replace(current, utterance_id="y", start_ms=120_500)), []
                )
                self.assertIn("self_voice outcome=self reason=echo_only fraction=1.00", self.notes)

    def test_capability_and_written_clock(self):
        ctl = self.session(self_voice={"enabled": True})
        for payload in (self.started, started_event(ctl.config)):
            self.assertEqual(payload["capabilities"]["self_echo"], 1)
        # A length-only chunk exercises receipt accounting without constructing audio.
        ctl._feed(ctl._generation, Mock(), range(6400), "stdin")
        self.assertEqual((ctl._received_bytes, ctl._received_ms), (6400, 200))
        with patch.object(ctl, "_now_ms", return_value=10000):
            self.assertEqual(
                read_lines(
                    ctl,
                    {**WINDOW, "from_sample": 1600, "tail_ms": 0},
                    {**ECHO, "from_sample": 1600, "to_sample": 3200},
                ),
                [],
            )
            accepted = []
            processor = LiveProcessor(
                LiveConfig(
                    SESSION,
                    provenance="synthetic",
                    transcriber=Mock(recognizer_id="fixture"),
                    diarizer=Mock(),
                ),
                accepted.append,
            )
            processor._publish_units(
                [{"start_ms": 20, "end_ms": 120, "text": "x"}],
                [],
                80,
                b"",
                "unknown",
                number=1,
                timeline_present=False,
                complete=True,
                packet=SimpleNamespace(discarded=False, exit_voice_span=None),
            )
            (current,) = accepted
            self.assertEqual((current.start_ms, current.end_ms), (100, 200))
            ctl._turn(ctl._generation, ctl._decision_queue, ctl._memory, self.shadow, current)
            self.assertEqual(self.events.drain(), [])
        self.assertIn("self_voice outcome=self reason=echo_only fraction=1.00", self.notes)
        self.shadow.turn.assert_not_called()

    def test_self_effects(self):
        for policy in ("suppress", "publish"):
            ctl = self.session(self_voice={"enabled": True, "transcripts": policy})
            ctl.reply_window("r", 0, 64_000, 0)
            self.tag(150, 3000)
            current = turn("x", 0, 1500, "Unit x x", "b")
            output = self.accept(current)
            self.assertEqual(
                [e["type"] for e in output], ["transcript"] if policy == "publish" else []
            )
            self.assertFalse(self.events._others_present(current))
            self.assertEqual(ctl._memory.retained_ids, set())
            self.assertEqual(self.events._memory.retained_ids, set())
            self.assertTrue(self.events.self_voice.excludes_answer(0))
            self.shadow.turn.assert_not_called()
            stop = turn("y", 2000, 3000, SpeakerPriority().stop_phrases[0], "a")
            self.assertFalse(any(e["type"] == "dismiss" for e in self.accept(stop)))
            self.assertEqual(self.events.self_voice.self_count, 2)
            self.assertIsNone(self.events._engaged)
            owner = turn("z", 3100, 4000, "x", "a")
            self.assertIn("request", [e["type"] for e in self.accept(owner)])
            self.assertFalse(self.events._others_present(owner))

    def test_self_precedes_renewed_label_scoring(self):
        ctl = self.session(self_voice={"enabled": True, "echo_only_min_fraction": 0.7})
        ctl.reply_window("r", 0, 32_000, 0)
        self.tag(300, 1000)
        roles = ctl._enrolled_roles = Mock()
        self.shadow = Mock(epoch=1)
        ctl._observe_spans(ctl._generation, "x", ((0, 1000),))
        self.assertEqual(self.accept(heard("x", 0, 1000, "x")), [])
        self.shadow.binding.assert_not_called()
        self.shadow.turn.assert_not_called()
        roles.note_spans.assert_not_called()
        self.assertEqual(ctl._turn_spans, {})

    def test_self_preserves_active_conversation(self):
        ctl = self.session(self_voice={"enabled": True})
        self.accept(turn("x", 0, 500, "Unit", "a"))
        engaged = dict(self.events._engaged)
        ctl.reply_window("r", 8000, 32_000, 0)
        self.tag(500, 2000)
        for idx, text in enumerate((SpeakerPriority().stop_phrases[0], "Unit", "x")):
            start = 500 + idx * 500
            self.assertEqual(self.accept(turn(str(idx), start, start + 500, text, "a")), [])
        self.assertEqual(self.events._engaged, engaged)
        self.assertEqual(len(self.events._pending), 0)
        self.assertEqual(self.shadow.turn.call_count, 1)

    def test_published_self_has_no_person_identity(self):
        entry = " ".join(sorted(PRIVACY_ENTRY)[0])
        for policy in ("publish", "suppress"):
            with self.subTest(policy=policy):
                ctl = self.session(self_voice={"enabled": True, "transcripts": policy})
                roles = enrolled()
                self.events._priority, self.events._per_turn = roles, True
                self.events._role_sources = True
                ctl.reply_window("r", 0, 16_000, 0)
                self.tag(0, 1000)
                current = heard("x", 0, 1000, entry if policy == "suppress" else "x")
                self.assertEqual(roles.role_for(current), "owner")
                with patch.object(roles, "role_for", wraps=roles.role_for) as lookup:
                    (event,) = self.accept(current)
                self.assertEqual(event["type"], "transcript")
                self.assertEqual(event["turn"], {**current.to_dict(), "role": "unknown"})
                lookup.assert_not_called()
                self.assertNotIn("speaker_name", event)
                self.assertNotIn("role_source", event)
                self.assertEqual(self.events._roles, {})

    def test_self_publications_leave_person_budget(self):
        ctl = self.session(self_voice={"enabled": True, "transcripts": "publish"})
        ctl.reply_window("r", 0, 1100 * 16, 0)
        self.tag(0, 1100)
        with patch.object(self.events._memory, "snapshot") as snapshot:
            for idx in range(1100):
                current = heard(str(idx), idx, idx + 1, "x")
                (event,) = self.accept(current)
                self.assertEqual(event["type"], "transcript")
            self.assertEqual(self.accept(current), [])
        snapshot.assert_not_called()
        self.assertLessEqual(len(self.events.self_voice.published), 512)
        self.assertNotIn("0", self.events.self_voice.published)
        person = turn("person", 1100, 1500, "Unit", "a")
        self.assertIn("request", [e["type"] for e in self.accept(person)])
        self.assertEqual(set(self.events._seen), {"person"})
        self.assertFalse(ctl._stop.is_set())

    def test_lagged_mixed_preserves_stop(self):
        ctl = self.session(self_voice={"enabled": True})
        self.accept(turn("a", 0, 500, "Unit", "a"))
        ctl.reply_window("r", 500 * 16, None, 0)
        ctl._received_ms = 1000
        self.tag(500, 1000)
        self.tag(950, 1600, "mixed")
        stop = turn("b", 500, 1000, SpeakerPriority().stop_phrases[0], "a")
        self.assertIn("dismiss", [e["type"] for e in self.accept(stop)])
        self.assertIn("self_voice outcome=person reason=mixed fraction=1.00", self.notes)

    def test_mixed_wholly_ahead_preserves_stop(self):
        ctl = self.session(self_voice={"enabled": True})
        self.accept(turn("a", 0, 500, "Unit", "a"))
        ctl.reply_window("r", 500 * 16, None, 0)
        ctl._received_ms = 1300
        self.tag(500, 1300)
        self.tag(1400, 1900, "mixed")
        stop = turn("b", 500, 1500, SpeakerPriority().stop_phrases[0], "a")
        self.assertIn("dismiss", [e["type"] for e in self.accept(stop)])
        self.assertIn("self_voice outcome=person reason=mixed fraction=0.80", self.notes)

    def test_mixed_barge_in(self):
        def sequence(mixed):
            self.session(**({"self_voice": {"enabled": True}} if mixed else {}))
            if mixed:
                self.ctl.reply_window("r", 0, 32_000, 0)
                self.tag(0, 2000)
                self.tag(499, 501, "mixed")
                self.tag(1499, 1501, "mixed")
            first = self.accept(turn("x", 0, 1000, "Unit", "a"))
            stop = self.accept(turn("y", 1000, 2000, SpeakerPriority().stop_phrases[0], "a"))
            self.assertIn("request", [e["type"] for e in first])
            self.assertIn("dismiss", [e["type"] for e in stop])
            return first + stop

        self.assertEqual(sequence(False), sequence(True))

    def profile_session(self):
        ctl = self.session(self_voice={"enabled": True})
        ctl.self_voice_manifest = self.path
        self.shadow.self_bound = lambda label: label == "a"
        ctl.reply_window("r", 0, 2000 * 16, 0)
        return ctl

    def test_profile_turns_and_watermark(self):
        ctl = self.profile_session()
        self.assertEqual(self.accept(heard("x", 0, 500, "x", "a")), [])
        self.assertIn("self_voice outcome=self reason=self_profile fraction=0.00", self.notes)
        ctl._events.self_voice.mixed_lost_until = 501
        self.assertIn(
            "request", [e["type"] for e in self.accept(heard("y", 500, 1000, "Unit", "a"))]
        )
        self.shadow.turn.assert_called_once()
        self.assertFalse(self.shadow.turn.call_args.kwargs["in_window"])

    def test_profile_mixed_barge_in(self):
        self.profile_session()
        self.tag(499, 501, "mixed")
        self.tag(1499, 1501, "mixed")
        first = self.accept(heard("x", 0, 1000, "Unit", "a"))
        stop = self.accept(heard("y", 1000, 2000, SpeakerPriority().stop_phrases[0], "a"))
        self.assertIn("request", [e["type"] for e in first])
        self.assertIn("dismiss", [e["type"] for e in stop])
        self.assertEqual(self.shadow.turn.call_count, 2)

    def test_profile_outside_role_and_presence(self):
        def sequence(profile):
            ctl = self.profile_session()
            if not profile:
                ctl.self_voice_manifest = None
            roles = enrolled()
            self.shadow.self_bound = lambda _: True
            self.shadow.binding = lambda _: ("unknown", "owner")
            roles.source = self.shadow
            self.events._priority, self.events._per_turn = roles, True
            output = self.accept(heard("x", 2000, 4000, "Unit x x", "other"))
            self.assertEqual(output[0]["turn"]["role"], "participant")
            self.assertIn("request", [e["type"] for e in output])
            self.assertEqual(list(self.events._cue_voices), [(4000, "other", None)])
            return [encode_json(event) for event in output]

        self.assertEqual(sequence(False), sequence(True))

    def scoring_session(self, profile):
        ctl = self.session(self_voice={"enabled": True})
        ctl.self_voice_manifest = self.path if profile else None
        ctl._now_ms = lambda: ctl._received_ms
        embedder = ScriptedEmbedder([unit(0, 1, 0), unit(1, 0, 0)])
        self.shadow = ShadowSpeakerId(
            settings(bind_min_seconds=2),
            entries=lambda _: ENTRIES[:1],
            embedder_factory=lambda _: embedder,
        )
        if profile:
            self.shadow._self_prints = (unit(0, 1, 0),)
        self.shadow.start()
        self.addCleanup(self.shadow.close)
        self.shadow.audio(b"\x00\x20" * 16_000 * 4)
        ctl._observe_spans(ctl._generation, "x", ((0, 2000),))
        ctl._observe_spans(ctl._generation, "y", ((2000, 4000),))
        return ctl

    def test_renewed_owner_stop_outside_window(self):
        def sequence(profile):
            ctl = self.scoring_session(profile)
            roles = EnrolledRoles(SpeakerPriority.from_dict({"owner": ["owner"]}))
            roles.source = self.shadow
            ctl._enrolled_roles = self.events._priority = roles
            self.events._per_turn = True
            self.shadow.forget()
            ctl.reply_window("r", 0, 2000 * 16, 0)
            self.accept(heard("x", 0, 2000, "Unit", "a"))
            self.assertEqual(self.shadow.self_bound("a"), profile)
            stop = heard("y", 2000, 4000, SpeakerPriority().stop_phrases[0], "a")
            output = self.accept(stop)
            self.assertEqual(self.shadow.binding("a"), ("bound", "owner"))
            self.assertEqual(output[0]["turn"]["role"], "owner")
            self.assertIn("dismiss", [e["type"] for e in output])
            return [encode_json(event) for event in output]

        self.assertEqual(sequence(False), sequence(True))

    def test_mixed_then_untagged_stop(self):
        def sequence(profile):
            ctl = self.scoring_session(profile)
            ctl.reply_window("r", 0, 4000 * 16, 0)
            self.tag(0, 2000, "mixed")
            first = self.accept(heard("x", 0, 2000, "Unit", "a"))
            wait_for(lambda: self.shadow.scored == 1)
            stop = self.accept(heard("y", 2000, 4000, SpeakerPriority().stop_phrases[0], "a"))
            self.assertIn("dismiss", [e["type"] for e in stop])
            wait_for(lambda: self.shadow.scored == 2)
            # Only the untagged second turn was eligible for reference evidence.
            self.assertEqual(self.shadow._labels["a"].self_seconds, 2 if profile else 0)
            return [encode_json(event) for event in first + stop]

        self.assertEqual(sequence(False), sequence(True))

    def test_profile_private_turns_never_scored(self):
        ctl = self.profile_session()
        ctl.duck(True)
        self.events.drain()
        for label in ("a", "b"):
            self.assertEqual(self.accept(heard(label, 0, 1000, "x", label)), [])
        self.shadow.turn.assert_not_called()
        self.assertEqual(self.events.self_voice.self_count, 0)

    def test_acceptance_precedes_late_tag(self):
        ctl = self.session(self_voice={"enabled": True})
        ctl.reply_window("r", 0, 16_000, 0)
        current = turn("x", 0, 1000, "Unit", "a")
        self.accept(current, decide=False)
        self.tag(0, 1000)
        self.events.decision(decided(ctl._decision_queue.get_nowait(), "attend"), 1100)
        self.assertIn("request", [p["type"] for p in self.events.drain()])
        self.shadow.turn.assert_called_once()

    def test_answer_start_excluded_before_decision(self):
        ctl = self.session(self_voice={"enabled": True})
        ctl.reply_window("r", 16_000, 32_000, 0)
        self.tag(1150, 2000)
        current = turn("x", 0, 900, "x", "a")
        gap = {
            "observed": True,
            "following": "different_speaker",
            "silence_ms": 100,
            "window_ms": 1200,
        }
        ctl._observe_gap(ctl._generation, "x", gap)
        self.accept(current, decide=False)
        self.accept(turn("s", 1000, 2000, "x", "b"), decide=False)
        ctl._phase = "finishing"
        runner = ctl._runner
        runner.provider.decide.return_value = decided(current, "attend").decision
        ctl._decide(ctl._generation, ctl._stop, ctl._decision_queue, runner, True)
        (state,) = runner.provider.decide.call_args.args
        self.assertEqual(state["post_turn_gap"], {"observed": False})
        self.assertIn("request", [p["type"] for p in self.events.drain()])

    def test_unclassified_answer_point_keeps_gap(self):
        for kind in ("echo_only", "mixed"):
            ctl = self.session(self_voice={"enabled": True})
            ctl.reply_window("r", 16_000, 32_000, 0)
            self.tag(1200, 1300)
            self.tag(1250, 1251, kind)
            current = turn("x", 0, 900, "x", "a")
            self.accept(current, decide=False)
            event = decided(current, "attend")
            self.events.decision(
                replace(event, decision=replace(event.decision, answered_at_ms=1250)), 2000
            )
            self.assertNotIn("request", [e["type"] for e in self.events.drain()])

    def test_echo_then_mixed_answer_keeps_base_gap(self):
        gap = dict(observed=True, following="different_speaker", silence_ms=100, window_ms=1200)
        for enabled in (False, True):
            ctl = self.session(
                self_voice={"enabled": enabled},
                decision={"provider": "jev", "allow_hosted": True, "other_answered_blocks": True},
            )
            if enabled:
                ctl.reply_window("r", 16_000, 32_000, 0)
                self.tag(1000, 2000)
                self.tag(1500, 1501, "mixed")
            current = turn("x", 0, 900, "x", "a")
            ctl._observe_gap(ctl._generation, "x", gap)
            self.accept(current, decide=False)
            self.accept(turn("answer", 1000, 2000, "x", "b"), decide=False)
            ctl._phase = "finishing"
            runner = ctl._runner
            runner.provider.requests = 0
            runner.provider.decide.return_value = decided(current, "attend").decision
            ctl._decide(ctl._generation, ctl._stop, ctl._decision_queue, runner, True)
            (state,) = runner.provider.decide.call_args_list[0].args
            self.assertEqual(state["post_turn_gap"], gap)
            requests = [
                e["turn"]["utterance_id"] for e in self.events.drain() if e["type"] == "request"
            ]
            self.assertNotIn("x", requests)

    def test_self_entry_waits_for_host_duck(self):
        entry = " ".join(sorted(PRIVACY_ENTRY)[0])
        current = heard("entry", 0, 1000, entry)
        self.session()
        (expected,) = self.accept(current, decide=False)
        self.assertEqual(expected["type"], "transcript")
        expected["turn"]["role"] = "unknown"
        for policy in ("suppress", "publish"):
            with self.subTest(policy=policy):
                ctl = self.session(self_voice={"enabled": True, "transcripts": policy})
                ctl.reply_window("r", 0, 32_000, 0)
                self.tag(0, 2000)
                ctl._processor = Mock()
                output = self.accept(current)
                self.assertFalse(self.events.ducked)
                self.assertEqual(output, [expected])
                self.shadow.turn.assert_not_called()
                self.assertEqual(ctl._memory.retained_ids, set())
                self.assertEqual(self.events._memory.retained_ids, set())

                self.assertEqual(read_lines(ctl, {"duck": True}), [])
                self.assertTrue(self.events.ducked)
                ctl._processor.begin_privacy.assert_called_once_with(1000 * PCM_BYTES_PER_MS)
                (ack,) = self.events.drain()
                self.assertEqual(
                    (ack["type"], ack["action"], ack["at_ms"]), ("privacy", "on", 1000)
                )
                self.assertEqual(self.accept(heard("private", 1000, 2000, entry)), [])

    def test_hosted_self_entry_keeps_session_running(self):
        ctl = self.session(
            self_voice={"enabled": True},
            allow_hosted_speech=True,
            **HOSTED_CONSUMERS["transcriber"],
        )
        ctl.reply_window("r", 0, 16_000, 0)
        self.tag(0, 1000)
        entry = " ".join(sorted(PRIVACY_ENTRY)[0])
        (event,) = self.accept(heard("entry", 0, 1000, entry))
        self.assertEqual(event["type"], "transcript")
        self.assertFalse(ctl._stop.is_set())
        self.assertFalse(self.events.ducked)

    def test_self_entry_matcher_error_publishes_once(self):
        ctl = self.session(self_voice={"enabled": True})
        ctl.reply_window("r", 0, 16_000, 0)
        self.tag(0, 1000)
        with patch("rightyo.tool_events.privacy_phrase", side_effect=ValueError):
            current = heard("x", 0, 1000, "x")
            (event,) = self.accept(current)
            self.assertEqual(self.accept(current), [])
        self.assertEqual((event["type"], event["turn"]["role"]), ("transcript", "unknown"))
        self.assertFalse(self.events.ducked)

    def test_private_exit_from_self_is_ignored(self):
        ctl = self.session(self_voice={"enabled": True})
        ctl.reply_window("r", 0, 48_000, 0)
        self.tag(0, 3000)
        ctl.duck(True)
        self.events.drain()
        roles = enrolled()
        self.events._priority, self.events._per_turn = roles, True
        phrase = " ".join(("Unit", *PRIVACY_EXIT[0]))
        self.assertEqual(self.accept(heard("exit", 1000, 2000, phrase)), [])
        self.assertTrue(self.events.ducked)
        # The same authenticated candidate outside the region reaches the exit path.
        actions = self.accept(heard("control", 3000, 4000, phrase))
        self.assertEqual([e["action"] for e in actions], ["exit"])

    def test_private_self_status_and_exit_leave_no_record(self):
        ctl = self.session(self_voice={"enabled": True})
        ctl.reply_window("r", 0, 64_000, 0)
        self.tag(0, 4000)
        self.accept(heard("public", 0, 1000, "x"))
        store = self.events.self_voice
        notes = list(self.notes)
        ctl.duck(True)
        self.events.drain()
        roles = enrolled()
        self.events._priority, self.events._per_turn = roles, True
        for idx, phrase in enumerate((PRIVACY_STATUS[0], ("Unit", *PRIVACY_EXIT[0]))):
            start = (idx + 1) * 1000
            self.assertEqual(
                self.accept(heard(str(idx), start, start + 1000, " ".join(phrase))), []
            )
        self.assertTrue(self.events.ducked)
        ctl.duck(False)
        self.assertEqual(self.accept(heard("held", 1000, 2000, " ".join(PRIVACY_STATUS[0]))), [])
        self.assertEqual((list(store.self_starts), store.self_count), ([0], 1))
        self.assertEqual(
            [line for line in self.notes if line.startswith("self_voice")],
            [line for line in notes if line.startswith("self_voice")],
        )

    def test_turn_report_outside_controller_lock(self):
        ctl = self.session(self_voice={"enabled": True})
        locks = []
        ctl.report = lambda line: locks.append(ctl._lock._is_owned())
        ctl.reply_window("r", 0, 32_000, 0)
        self.tag(0, 1000)
        self.assertEqual(locks, [])
        self.accept(heard("self", 0, 1000, "x"))
        self.accept(heard("person", 1000, 2000, "x"))
        self.assertEqual(locks, [False, False])

    def test_stalled_reporter_follows_person_publication(self):
        ctl = self.session(self_voice={"enabled": True})
        ctl.reply_window("r", 0, 16_000, 0)
        self.tag(0, 1000)
        self.tag(0, 100, "mixed")
        current = heard("entry", 0, 1000, " ".join(sorted(PRIVACY_ENTRY)[0]))
        ctl._received_ms = current.end_ms
        reporting, release = Event(), Event()

        def report(line):
            if line == "self_voice outcome=person reason=mixed fraction=1.00":
                reporting.set()
                release.wait()

        ctl.report = report
        with ThreadPoolExecutor(max_workers=1) as worker:
            future = worker.submit(
                ctl._turn, ctl._generation, ctl._decision_queue, ctl._memory, self.shadow, current
            )
            try:
                self.assertTrue(reporting.wait(timeout=5))
                self.assertEqual(
                    [(e["type"], e["turn"]["utterance_id"]) for e in self.events.drain()],
                    [("transcript", current.utterance_id)],
                )
            finally:
                release.set()
            future.result()

    def test_summary(self):
        base = (
            "speaker_id summary offered=1 scored=0 short=0 overlap=0 dropped=0"
            " clipped=0 inferred=0 unsegmented=0 labelled_ms=0"
        )
        for enabled in (False, True):
            ctl = self.session(self_voice={"enabled": enabled})
            expected = base
            if enabled:
                ctl.reply_window("r", 0, 16_000, 0)
                self.tag(0, 1000)
                self.tag(1500, 2000)
                self.tag(500, 501, "mixed")
                self.tag(1100, 1200, "mixed")
                self.accept(heard("x", 0, 100, "x"))
                ctl.reply_window("s", 0, 36_000 * 16, 0)
                expected += " self=1 tags_echo=1/1 tags_mixed=2/0 windows_dropped=1"
            shadow = ShadowSpeakerId(Mock(), report=self.notes.append)
            shadow.offered = 1
            with (
                patch.object(ctl, "_shadow_speaker_id", return_value=shadow),
                patch("rightyo.prototype.StdinPcmCapture") as capture,
            ):
                capture.return_value.read.return_value = b""
                capture.return_value.overrun = False
                capture.return_value.discarded_tail_bytes = capture.return_value.dropped_bytes = 0
                ctl._audio(
                    ctl._generation, ctl._stop, SESSION, ctl._decision_queue, ctl._memory, "stdin"
                )
            self.assertEqual(
                [line for line in self.notes if line.startswith("speaker_id summary")], [expected]
            )
            self.assertFalse(any(line.startswith("self_echo") for line in self.notes))
