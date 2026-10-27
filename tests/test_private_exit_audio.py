"""#946 step 1: authored PCM and fake embeddings only; no speech or model assets."""

from __future__ import annotations

import contextlib
import io
import json
import os
import queue
import tempfile
import threading
import unittest
from argparse import Namespace
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from test_edge_attribution import Timeline, Units
from test_live_speaker_id import (
    GUEST_F0,
    OWNER_F0,
    Harness,
    ScriptedEmbedder,
    ToneEmbedder,
    settings,
    tone,
    turn,
    wait_for,
)

from rightyo.contracts import Addressing, SpeakerPriority
from rightyo.live_audio import LiveConfig, LiveProcessor
from rightyo.live_speaker_id import (
    PRIVATE_TURN_MAX_BYTES,
    EnrolledRoles,
    LabelState,
    PrivateTurnAudio,
)
from rightyo.memory import TranscriptMemory
from rightyo.prototype import PrototypeConfig, PrototypeController
from rightyo.speaker_id import SpeakerEmbedder, SpeakerIdError, _read_private_pcm
from rightyo.tool import listen
from rightyo.tool_events import SpeechEvents

ADDRESSING = Addressing(names=("Unit",))
OWNERS = SpeakerPriority(owners=("owner",))
EXIT = "Unit, privacy off"


class PrivateExitTests(unittest.TestCase):
    def harness(self, embedder=None, **options):
        harness = Harness(embedder or ToneEmbedder(), config=settings(**options))
        harness.shadow.start()
        self.addCleanup(harness.shadow.close)
        wait_for(lambda: bool(harness.shadow._compared))
        return harness

    def controller(self, harness, *, addressing=ADDRESSING):
        events = SpeechEvents(report=harness.lines.append)
        events.duck(True, now_ms=0)
        roles = EnrolledRoles(OWNERS)
        roles.source = harness.shadow
        events.start("shadow-test", now_ms=0, addressing=addressing, priority=roles, duck=True)
        events.drain()
        asset = Path("/synthetic/asset")
        controller = PrototypeController(
            PrototypeConfig(asset, asset, asset, asset, asset, asset), event_publisher=events
        )
        self.addCleanup(controller.close)
        controller._enrolled_roles = roles
        controller._shadow = harness.shadow
        roles.exit_audio = controller._private_sentence
        return controller, events, roles

    def hear(self, controller, text=EXIT, *, start=0, seconds=1, f0=OWNER_F0, inferred=False):
        current = replace(turn(start, start + round(seconds * 1000)), text=text)
        controller._received_bytes = current.end_ms * 32
        controller._received_ms = current.end_ms
        controller._private_turn_audio(
            controller._generation, current, memoryview(tone(seconds, f0))
        )
        old = controller._private_sentence.pcm
        if inferred:
            controller._inferred_turns.add(current.utterance_id)
        controller._turn(
            controller._generation, queue.Queue(), TranscriptMemory(), controller._shadow, current
        )
        return old

    def test_replacement_zeroes_the_old_object_and_holds_only_the_latest_turn(self):
        buffer = PrivateTurnAudio()
        first, second = turn(0, 1000), turn(1000, 2000)
        buffer.replace(first, memoryview(tone(1)))
        old = buffer.pcm
        buffer.replace(second, memoryview(tone(1, OWNER_F0)))
        self.assertTrue(old)
        self.assertFalse(any(old))
        self.assertIsNone(buffer.for_turn(first))
        self.assertIs(buffer.for_turn(second), buffer.pcm)
        self.assertEqual(len(buffer.pcm), 32000)
        latest = buffer.pcm
        buffer.clear()
        self.assertFalse(any(latest))
        self.assertEqual(buffer.pcm, bytearray())

    def test_cap_and_incomplete_spans_retain_nothing(self):
        buffer = PrivateTurnAudio()
        buffer.replace(turn(0, 10000), bytearray(PRIVATE_TURN_MAX_BYTES))
        self.assertEqual(len(buffer.pcm), PRIVATE_TURN_MAX_BYTES)
        old = buffer.pcm
        buffer.replace(turn(0, 10001), bytearray(PRIVATE_TURN_MAX_BYTES + 32))
        self.assertFalse(any(old))
        self.assertFalse(buffer.pcm)
        buffer.replace(turn(0, 1000), bytearray(31998))
        self.assertFalse(buffer.pcm)

    def test_start_private_owner_passes_and_guest_refuses_without_binding(self):
        for f0, allowed in ((OWNER_F0, True), (GUEST_F0, False)):
            with self.subTest(owner=allowed):
                harness = self.harness()
                controller, events, _roles = self.controller(harness)
                old = self.hear(controller, f0=f0)
                self.assertFalse(any(old))
                self.assertFalse(controller._private_sentence.pcm)
                self.assertEqual(
                    [e.get("action") for e in events.drain()], ["exit"] if allowed else []
                )
                self.assertEqual(
                    events._private_counts["exit_refused_voice_below_threshold"], int(not allowed)
                )
                self.assertEqual(harness.shadow._labels, {})
                self.assertEqual(harness.shadow._bound_ids, set())
                self.assertEqual(harness.shadow.scored_seconds, 0)
                self.assertEqual(harness.shadow.offered, 0)
                self.assertTrue(events.ducked)  # Only the host applies the off.

    def test_non_exit_status_and_unnamed_remainders_are_never_scored(self):
        harness = self.harness()
        controller, _events, _roles = self.controller(harness)
        with patch.object(harness.shadow, "exit_score", wraps=harness.shadow.exit_score) as score:
            previous = None
            for index, text in enumerate(("synthetic token", "status", "privacy off")):
                old = self.hear(controller, text, start=index * 1000)
                self.assertTrue(any(old))
                if previous is not None:
                    self.assertFalse(any(previous))
                previous = old
            score.assert_not_called()
        self.assertEqual(harness.embedder.calls, 0)
        self.assertEqual(harness.shadow._labels, {})

    def test_exit_voice_never_rebinds_or_accumulates_into_existing_labels(self):
        harness = self.harness()
        controller, events, _roles = self.controller(harness)
        previous = LabelState(state="bound", identity="guest", seconds=3, turns=2)
        harness.shadow._labels["Speaker A"] = previous
        harness.shadow._bound_ids = {"guest"}
        harness.shadow.scored_seconds = 3
        self.hear(controller, "synthetic token")
        self.hear(controller, start=1000)
        self.assertEqual([event.get("action") for event in events.drain()], ["exit"])
        self.assertIs(harness.shadow._labels["Speaker A"], previous)
        self.assertEqual(harness.shadow._bound_ids, {"guest"})
        self.assertEqual(harness.shadow.scored_seconds, 3)
        self.assertEqual(harness.shadow.offered, 0)

    def test_wipes_on_unavailable_short_and_embedder_exception(self):
        for embedder, seconds in ((ScriptedEmbedder(fail_on=1), 1), (ToneEmbedder(), 0.2)):
            with self.subTest(seconds=seconds):
                harness = self.harness(embedder)
                controller, events, _roles = self.controller(harness)
                old = self.hear(controller, seconds=seconds)
                self.assertFalse(any(old))
                self.assertFalse(controller._private_sentence.pcm)
                self.assertEqual(events.drain(), [])
                self.assertEqual(events._private_counts["exit_refused_voice_unavailable"], 1)
                self.assertNotIn("synthetic embedder failure", " ".join(harness.lines))

    def test_exception_from_exit_scorer_is_sanitized_and_wiped(self):
        harness = self.harness()
        controller, events, _roles = self.controller(harness)
        with patch.object(
            harness.shadow, "exit_score", side_effect=RuntimeError("synthetic token")
        ):
            old = self.hear(controller)
        self.assertFalse(any(old))
        self.assertEqual(events._private_counts["exit_refused_voice_unavailable"], 1)
        self.assertNotIn("synthetic token", " ".join(harness.lines))

    def test_oversize_and_missing_audio_refuse_without_scoring(self):
        harness = self.harness()
        controller, events, _roles = self.controller(harness)
        with patch.object(harness.shadow, "exit_score") as score:
            old = self.hear(controller, seconds=10.001)
            self.assertFalse(old)
            current = replace(turn(11000, 12000), text=EXIT)
            controller._turn(
                controller._generation, queue.Queue(), TranscriptMemory(), harness.shadow, current
            )
            score.assert_not_called()
        self.assertEqual(events.drain(), [])
        self.assertEqual(events._private_counts["exit_refused_voice_unavailable"], 2)

    def test_matcher_and_event_delivery_errors_leave_no_pcm_or_exception_content(self):
        for failure in ("matcher", "delivery"):
            with self.subTest(failure=failure):
                harness = self.harness()
                controller, events, _roles = self.controller(harness)
                old = self.hear(controller, "synthetic token")
                current = replace(turn(1000, 2000), text=EXIT)
                controller._private_turn_audio(
                    controller._generation, current, memoryview(tone(1, OWNER_F0))
                )
                self.assertFalse(any(old))
                candidate = controller._private_sentence.pcm
                target = "rightyo.tool_events.privacy_phrase" if failure == "matcher" else None
                manager = (
                    patch(target, side_effect=RuntimeError("synthetic token"))
                    if target
                    else patch.object(events, "_emit", side_effect=RuntimeError("synthetic token"))
                )
                with manager:
                    if failure == "matcher":
                        events.transcript(current, now_ms=2000)
                    else:
                        with self.assertRaises(RuntimeError):
                            events.transcript(current, now_ms=2000)
                self.assertFalse(any(candidate))
                self.assertFalse(controller._private_sentence.pcm)
                self.assertNotIn("synthetic token", " ".join(harness.lines))

    def test_binding_pass_inferred_and_playback_prechecks_wipe_without_scoring(self):
        for outcome in ("bound", "inferred", "playback"):
            with self.subTest(outcome=outcome):
                harness = self.harness()
                controller, events, _roles = self.controller(harness)
                if outcome == "bound":
                    harness.shadow._labels["Speaker A"] = LabelState(
                        state="bound", identity="owner"
                    )
                if outcome == "playback":
                    events._playbacks.append([0, 1000])
                with patch.object(harness.shadow, "exit_score") as score:
                    old = self.hear(controller, inferred=outcome == "inferred")
                    score.assert_not_called()
                self.assertFalse(any(old))
                self.assertEqual(
                    [e.get("action") for e in events.drain()],
                    ["exit"] if outcome == "bound" else [],
                )
                if outcome != "bound":
                    self.assertEqual(events._private_counts[f"exit_refused_{outcome}"], 1)

    def test_wipes_on_privacy_off_stop_session_end_and_forget(self):
        for action in ("off", "stop", "end", "forget", "renew"):
            with self.subTest(action=action):
                harness = self.harness()
                controller, events, _roles = self.controller(harness)
                old = self.hear(controller, "synthetic token")
                self.assertTrue(any(old))
                if action == "off":
                    controller.duck(False)
                elif action == "stop":
                    controller.stop()
                elif action == "end":
                    events.end(phase="stopped", now_ms=1000)
                elif action == "renew":
                    controller._speakers_renewed()
                else:
                    controller._forget_speakers()
                self.assertFalse(any(old))
                self.assertFalse(controller._private_sentence.pcm)

    def test_privacy_on_mid_turn_never_copies_public_prefix(self):
        harness = self.harness()
        controller, events, _roles = self.controller(harness)
        controller.duck(False)
        controller._received_bytes, controller._received_ms = 16000, 500
        controller.duck(True)
        events.drain()
        old = self.hear(controller)
        self.assertFalse(old)
        self.assertEqual(events.drain(), [])
        self.assertEqual(harness.embedder.calls, 0)
        # A turn wholly after the precise on-edge can authenticate normally.
        self.hear(controller, start=1000)
        self.assertEqual([e.get("action") for e in events.drain()], ["exit"])

    def test_bare_jarvis_uses_only_binding_and_zeroes_audio_without_a_score_or_log(self):
        for identity in (None, "guest", "owner"):
            with self.subTest(binding=identity):
                harness = self.harness()
                controller, events, roles = self.controller(
                    harness, addressing=Addressing(names=("Jarvis",))
                )
                if identity is not None:
                    harness.shadow._labels["Speaker A"] = LabelState(
                        state="bound", identity=identity
                    )
                with (
                    patch.object(roles, "exit_voice", wraps=roles.exit_voice) as voice,
                    patch.object(
                        harness.shadow, "exit_score", wraps=harness.shadow.exit_score
                    ) as score,
                ):
                    old = self.hear(controller, "Jarvis")
                    voice.assert_not_called()
                    score.assert_not_called()
                self.assertTrue(old)
                self.assertFalse(any(old))
                self.assertFalse(controller._private_sentence.pcm)
                self.assertEqual(events.drain(), [])
                self.assertEqual(harness.embedder.calls, 0)
                self.assertFalse(any("score=" in line for line in harness.lines))
                self.assertEqual(events._private_name_end, 1000 if identity == "owner" else None)
                self.assertEqual(
                    events._private_counts["exit_name_refused"], int(identity != "owner")
                )

    def test_unbound_name_cannot_start_a_split_exit_even_with_owner_audio(self):
        harness = self.harness()
        controller, events, _roles = self.controller(harness)
        with patch.object(harness.shadow, "exit_score", wraps=harness.shadow.exit_score) as score:
            first = self.hear(controller, "Unit", f0=OWNER_F0)
            self.hear(controller, "privacy off", start=1100, f0=OWNER_F0)
            score.assert_not_called()
        self.assertFalse(any(first))
        self.assertEqual(harness.embedder.calls, 0)
        self.assertEqual(events.drain(), [])
        self.assertEqual(events._private_counts["exit_name_refused"], 1)
        self.assertEqual(events._private_counts["exit_refused_unbound"], 1)
        self.assertEqual(events._private_counts["exit_unnamed"], 1)

    def test_split_name_requires_binding_and_scores_only_the_completing_turn(self):
        for f0, allowed in ((OWNER_F0, True), (GUEST_F0, False)):
            with self.subTest(owner=allowed):
                harness = self.harness()
                controller, events, _roles = self.controller(harness)
                harness.shadow._labels["Speaker A"] = LabelState(state="bound", identity="owner")
                with patch.object(harness.shadow, "exit_score") as score:
                    first = self.hear(controller, "Unit", f0=OWNER_F0)
                    score.assert_not_called()
                self.assertFalse(any(first))
                self.assertFalse(controller._private_sentence.pcm)
                self.assertEqual(harness.embedder.calls, 0)
                # The accepted fragment needs its binding; the completing turn can
                # authenticate its own audio without that binding still being present.
                harness.shadow._labels.clear()
                with patch.object(
                    harness.shadow, "exit_score", wraps=harness.shadow.exit_score
                ) as score:
                    second = self.hear(controller, "privacy off", start=1100, f0=f0)
                    score.assert_called_once()
                    self.assertIs(score.call_args.args[0], second)
                self.assertEqual(len(second), len(tone(1, f0)))
                self.assertFalse(any(second))
                self.assertFalse(controller._private_sentence.pcm)
                self.assertEqual(
                    [e.get("action") for e in events.drain()], ["exit"] if allowed else []
                )
                self.assertEqual(harness.embedder.calls, 1)

    def test_threshold_and_minimum_match_entry_configuration(self):
        harness = self.harness(ScriptedEmbedder(vectors=([0.6, 0.8, 0],)), entry_threshold=0.7)
        controller, events, _roles = self.controller(harness)
        self.hear(controller)
        self.assertEqual(events.drain(), [])
        self.assertEqual(events._private_counts["exit_refused_voice_below_threshold"], 1)
        self.assertIn("speaker_id exit_check score=0.600", harness.lines)

    def test_worker_timeout_cancels_queued_audio_before_a_late_score(self):
        gate = threading.Event()
        harness = Harness(ScriptedEmbedder(gate=gate))
        harness.shadow.start()
        self.addCleanup(harness.shadow.close)
        wait_for(lambda: bool(harness.shadow._compared))
        harness.speak(1)
        wait_for(lambda: harness.embedder.calls == 1)
        pcm = bytearray(tone(1))
        with patch("rightyo.live_speaker_id.ENTRY_CHECK_TIMEOUT_S", 0.01):
            result = harness.shadow.exit_score(pcm, ("owner",))
        self.assertEqual(result.outcome, "timeout")
        self.assertFalse(any(pcm))
        gate.set()
        wait_for(lambda: harness.shadow.scored == 1)
        harness.shadow.close(drain=True)
        self.assertEqual(harness.embedder.calls, 1)

    def test_private_embedding_views_are_wiped_even_on_failure(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                views = []
                embedder = ScriptedEmbedder()

                def embed(pcm):
                    self.assertIsInstance(pcm, memoryview)
                    self.assertFalse(pcm.readonly)
                    views.append(pcm)
                    if fail:
                        raise ValueError("synthetic token")
                    return [1.0, 0.0, 0.0]

                embedder.embed_private = embed
                harness = self.harness(embedder)
                pcm = bytearray(tone(4))
                result = harness.shadow.exit_score(pcm, ("owner",))
                self.assertEqual(result.outcome, "unavailable" if fail else "scored")
                self.assertFalse(any(pcm))
                self.assertTrue(views)
                self.assertTrue(all(not any(view) for view in views))

    def test_inflight_timeout_finishes_and_wipes_before_return_without_more_windows(self):
        gate, started = threading.Event(), threading.Event()
        views = []
        embedder = ScriptedEmbedder()

        def embed(pcm):
            views.append(pcm)
            started.set()
            gate.wait(3)
            return [1.0, 0.0, 0.0]

        embedder.embed_private = embed
        harness = self.harness(embedder)
        pcm, results = bytearray(tone(8)), []
        with patch("rightyo.live_speaker_id.ENTRY_CHECK_TIMEOUT_S", 0.2):
            caller = threading.Thread(
                target=lambda: results.append(harness.shadow.exit_score(pcm, ("owner",)))
            )
            caller.start()
            self.assertTrue(started.wait(1))
            caller.join(0.25)
            self.assertTrue(caller.is_alive())  # Cannot release an actively borrowed view.
            gate.set()
            caller.join(2)
        self.assertFalse(caller.is_alive())
        self.assertEqual(results[0].outcome, "timeout")
        self.assertEqual(len(views), 1)
        self.assertFalse(any(pcm))
        self.assertFalse(any(views[0]))

    def test_live_processor_retains_only_exact_final_turn_and_ring_stays_silent(self):
        harness = self.harness()
        controller, events, _roles = self.controller(harness)
        generation = controller._generation
        work, memory = queue.Queue(), TranscriptMemory()
        retained = []

        def audio(current, pcm):
            controller._private_turn_audio(generation, current, pcm)
            retained.append(controller._private_sentence.pcm)

        processor = LiveProcessor(
            LiveConfig(
                "shadow-test",
                transcriber=Units(("synthetic token", 0, 400), (EXIT, 500, 1500)),
                diarizer=Timeline((0, 400, 1), (500, 1500, 2)),
                private=controller._heard_private_span,
                on_private_turn_audio=audio,
                turn_merge_gap_ms=2000,
                reply_wait_ms=2000,
            ),
            lambda current: controller._turn(generation, work, memory, harness.shadow, current),
        )
        self.addCleanup(processor.close)
        pcm = tone(1.5, OWNER_F0) + bytes(32000 * 2)
        for at in range(0, len(pcm), 6400):
            controller._feed(generation, processor, pcm[at : at + 6400], "stdin", harness.shadow)
        processor.finish()
        self.assertEqual([len(pcm) for pcm in retained], [400 * 32, 1000 * 32])
        self.assertTrue(all(not any(pcm) for pcm in retained))
        self.assertEqual([e.get("action") for e in events.drain()], ["exit"])
        self.assertEqual(harness.embedder.calls, 1)
        self.assertEqual(harness.shadow._labels, {})
        self.assertEqual(memory.snapshot(4000)["turns"], [])
        self.assertEqual(work.qsize(), 0)
        self.assertFalse(any(harness.shadow._ring.span(0, 3500)[0]))


class PrivateListenTests(unittest.TestCase):
    def test_start_private_listen_uses_voice_without_files_or_content_in_logs(self):
        for f0, allowed in ((OWNER_F0, True), (GUEST_F0, False)):
            with self.subTest(owner=allowed), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                asset, config = root / "asset", root / "config.json"
                asset.touch()
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
                            "addressing": {"names": ["Unit"]},
                            "speakers": {"owner": ["owner"]},
                            "speaker_id": {
                                "python": str(asset),
                                "model": str(asset),
                                "live": True,
                                "roles": True,
                            },
                        }
                    )
                )
                expected = set(root.rglob("*"))
                harness = Harness(ToneEmbedder())
                controllers, buffers = [], []

                def shadow(_settings, *, report):
                    harness.shadow._report_to = report
                    return harness.shadow

                def processor(loaded, on_turn):
                    wait_for(lambda: bool(harness.shadow._compared))
                    callback = loaded.on_private_turn_audio

                    def audio(current, pcm):
                        callback(current, pcm)
                        buffers.append(controllers[0]._private_sentence.pcm)

                    return LiveProcessor(
                        replace(
                            loaded,
                            transcriber=Units((EXIT, 0, 1000)),
                            diarizer=Timeline((0, 1000, 1)),
                            on_private_turn_audio=audio,
                        ),
                        on_turn,
                    )

                def factory(loaded, **options):
                    controller = PrototypeController(
                        loaded,
                        processor_factory=processor,
                        speaker_id_factory=shadow,
                        capture_factory=Mock(side_effect=AssertionError("capture")),
                        provider_factory=Mock(side_effect=AssertionError("hosted")),
                        **options,
                    )
                    controllers.append(controller)
                    return controller

                args = Namespace(
                    config=config,
                    mode="stdin",
                    provenance="synthetic",
                    session_id="private-test",
                    use_jev=False,
                    allow_hosted=False,
                    start_private=True,
                )
                output, errors = io.StringIO(), io.StringIO()
                with (
                    patch.dict(os.environ, {"HOME": directory}),
                    contextlib.chdir(root),
                    patch("sys.stderr", errors),
                ):
                    result = listen(
                        args,
                        output=output,
                        controller_factory=factory,
                        audio_input=io.BytesIO(tone(1, f0) + bytes(32000 * 2)),
                    )
                self.assertEqual(result, 0)
                self.assertEqual(set(root.rglob("*")), expected)
                events = [json.loads(line) for line in output.getvalue().splitlines()]
                self.assertEqual(events[0]["duck"], {"version": 3, "private": True})
                self.assertEqual(
                    [e.get("action") for e in events if e["type"] == "privacy"],
                    ["exit"] if allowed else [],
                )
                self.assertTrue(buffers)
                self.assertTrue(all(not any(buffer) for buffer in buffers))
                self.assertEqual(harness.shadow._labels, {})
                self.assertEqual(harness.shadow.scored, 0)
                for line in errors.getvalue().splitlines():
                    self.assertRegex(
                        line,
                        r"^rightyo: (audio_retention speaker_ring_s=\d+ timeline_window_s=\d+"
                        r"|speaker_id start enrolled=\d+|speaker_id exit_check score=-?\d+\.\d+)$",
                    )
                self.assertNotIn(EXIT, output.getvalue() + errors.getvalue())


class PrivatePipeTests(unittest.TestCase):
    def test_worker_reads_mutable_pcm_and_wipes_partial_reads_on_failure(self):
        payload = tone(0.5)
        pcm = _read_private_pcm(io.BytesIO(payload), len(payload))
        self.assertIsInstance(pcm, bytearray)
        self.assertEqual(pcm, payload)
        for failure in ("eof", "exception"):
            with self.subTest(failure=failure):
                held = []

                def readinto(view):
                    held.append(view)
                    if len(held) == 1:
                        view[:2] = b"\x01\x02"
                        return 2
                    if failure == "exception":
                        raise OSError("synthetic token")
                    return 0

                with self.assertRaises((SpeakerIdError, OSError)):
                    _read_private_pcm(Mock(readinto=readinto), 32000)
                self.assertFalse(any(held[0]))
        for size in (None, True, -2, 0, 3, 320002):
            with self.assertRaises(SpeakerIdError):
                _read_private_pcm(io.BytesIO(), size)

    def test_private_request_has_fixed_header_and_borrowed_mutable_pcm_only(self):
        client = object.__new__(SpeakerEmbedder)
        client.process = Mock()
        client.process.stdin.write.side_effect = lambda view: len(view)
        client.timeout = 1
        client._receive = Mock(return_value={"embedding": [1.0, 0.0]})
        pcm = bytearray(tone(0.5))
        self.assertEqual(client.embed_private(memoryview(pcm)), [1.0, 0.0])
        header, payload = [call.args[0] for call in client.process.stdin.write.call_args_list]
        self.assertEqual(
            json.loads(header.tobytes()), {"command": "embed_private", "size": len(pcm)}
        )
        self.assertIs(payload.obj, pcm)
        for invalid in (
            memoryview(b"\0\0"),
            memoryview(bytearray(3)),
            memoryview(bytearray(320002)),
        ):
            with self.assertRaises(SpeakerIdError):
                client.embed_private(invalid)


if __name__ == "__main__":
    unittest.main()
