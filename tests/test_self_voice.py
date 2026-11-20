"""Tag-only self exclusion with authored turns and inert workers."""

import hashlib
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from test_conversation import SESSION, decided, turn
from test_duck_mode import HOSTED_CONSUMERS, enrolled, heard

from rightyo.contracts import SpeakerPriority
from rightyo.live_audio import LiveConfig, LiveProcessor
from rightyo.live_speaker_id import ShadowSpeakerId
from rightyo.prototype import PCM_BYTES_PER_MS, PrototypeConfig, PrototypeController, PrototypeError
from rightyo.self_voice import HISTORY_MS, SelfVoice, SelfVoiceConfig
from rightyo.tool import _read_control
from rightyo.tool_events import PRIVACY_ENTRY, PRIVACY_EXIT, SpeechEvents, encode_json
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

    def test_window_cap_and_eviction(self):
        for idx in range(9):
            self.window(str(idx), 0, 100, 0, 100)
            self.tag(str(idx), idx, idx + 1, "mixed", 100)
        self.assertEqual(list(self.store.windows), list(map(str, range(1, 9))))
        self.assertEqual([s.reply for s in self.store.spans], list(self.store.windows))

    def test_lateness(self):
        self.window("r", 0, 200, 0, 200)
        self.tag("r", 0, 99, "mixed", 5100)
        self.tag("r", 0, 100, "echo_only", 5100)
        self.assertEqual(self.classify(0, 100, 5100), ("echo_only", 1))

    def test_outside_window(self):
        self.window("r", 10, 110, 0, 110)
        self.tag("s", 10, 110, "mixed", 110)
        self.tag("r", 0, 109, "mixed", 110)
        self.tag("r", 10, 110, "echo_only", 110)
        self.assertEqual(len(self.store.spans), 1)
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

    def test_submillisecond_mixed_and_retracted_tail(self):
        self.window("r", 0, None, 100, 100)
        self.tag("r", 0, 150, "echo_only", 100)
        self.tag("r", 99, 99, "mixed", 100, partial_end=True)
        self.assertEqual(self.classify(90, 110, 100)[0], "mixed")
        self.window("r", 0, 100, 5, 100)
        self.assertFalse(self.store.excludes_answer(105))
        self.assertTrue(self.store.excludes_answer(104))
        self.assertFalse(self.store.excludes_answer(99))

    def test_span_cap_and_pruning(self):
        self.window("r", 0, 2000, 0, 0)
        for idx in range(513):
            self.tag("r", idx * 2, idx * 2 + 1, "echo_only", 1100)
        self.assertEqual(len(self.store.spans), 512)
        self.assertEqual(self.store.spans[0].start, 2)
        self.classify(0, 2, HISTORY_MS + 2000)
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
            payloads = [self.started]
            for index, text in enumerate(("Unit", SpeakerPriority().stop_phrases[0])):
                payloads += self.accept(
                    turn(str(index), 1000 * index, 1000 * index + 500, text, "a")
                )
            self.events.end("stopped", 2000)
            payloads += self.events.drain()
            wire = "".join(encode_json(p) + "\n" for p in payloads).encode("ascii")
            # SHA-256 of the identical authored script at origin/main 2ae361c.
            self.assertEqual(
                hashlib.sha256(wire).hexdigest(),
                "d0d228b511d5d4a71b28fb19f9a9af39b17c8798e000e3e6f8c3395d4c7736fc",
            )

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
                packet=SimpleNamespace(discarded=False),
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

    def test_unchanged_publication(self):
        current = turn("x", 0, 1000, "x", "b")
        self.session()
        expected = self.accept(current)[0]
        self.session(self_voice={"enabled": True, "transcripts": "publish"})
        self.ctl.reply_window("r", 0, 16_000, 0)
        self.tag(0, 1000)
        (actual,) = self.accept(current)
        self.assertEqual(encode_json(actual), encode_json(expected))
        self.assertEqual(self.events._roles, {})

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

    def test_answer_point(self):
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
            self.assertEqual(
                "request" in [e["type"] for e in self.events.drain()], kind == "echo_only"
            )

    def test_self_entry_waits_for_host_duck(self):
        entry = " ".join(sorted(PRIVACY_ENTRY)[0])
        current = heard("entry", 0, 1000, entry)
        self.session()
        (expected,) = self.accept(current, decide=False)
        self.assertEqual(expected["type"], "transcript")
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

    def test_self_entry_matcher_error_stays_suppressed(self):
        ctl = self.session(self_voice={"enabled": True})
        ctl.reply_window("r", 0, 16_000, 0)
        self.tag(0, 1000)
        with patch("rightyo.tool_events.privacy_phrase", side_effect=ValueError):
            self.assertEqual(self.accept(heard("x", 0, 1000, "x")), [])
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

    def test_summary(self):
        for count in (None, 0, 2):
            notes = []
            shadow = ShadowSpeakerId(Mock(), report=notes.append)
            shadow.offered = int(count != 2)
            shadow.close(self_count=count)
            (line,) = notes
            self.assertEqual(line.endswith(f" self={count}"), count is not None)
