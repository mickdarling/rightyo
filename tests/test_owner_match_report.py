"""The #243 owner-match harness: authored shadow notes and tones with a stand-in embedder only."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from array import array
from pathlib import Path

from scripts import owner_match_report as harness

RATE = 16000


def note(
    label: str,
    speech_ms: int,
    score: float,
    state: str,
    route: str | None = None,
    identity: str = "owner",
) -> str:
    tail = f" route={route}" if route else ""
    return (
        f'rightyo: speaker_id label="{label}" turn_ms={speech_ms + 100} labelled_ms={speech_ms}'
        f" speech_ms={speech_ms} turn_score={score:.3f} acc_score={score:.3f} acc_s=4.0"
        f" state={state} id={identity}{tail}\n"
    )


def tone(seconds: float, amplitude: int) -> array:
    return array(
        "h",
        (
            int(amplitude * math.sin(2 * math.pi * 140 * n / RATE))
            for n in range(int(seconds * RATE))
        ),
    )


def unit(*values: float) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values]


OWNER = unit(1.0, 0.0)


def loudness_embedder(pcm: bytes) -> list[float]:
    """Near the owner print for loud audio, far from it for quiet audio."""
    samples = array("h")
    samples.frombytes(pcm)
    peak = max(abs(value) for value in samples)
    return unit(1.0, 0.1) if peak > 6000 else unit(0.1, 1.0)


class NoteParsingTests(unittest.TestCase):
    def test_turns_split_by_session_with_routes_and_short_counts(self):
        lines = [
            "attention: something else entirely\n",
            "rightyo: speaker_id start enrolled=1\n",
            note("Speaker A", 2400, 0.41, "unknown", "built_in_mic"),
            note("Speaker A", 4100, 0.72, "bound", "built_in_mic"),
            "rightyo: speaker_id summary offered=4 scored=2 short=2 overlap=0 dropped=0\n",
            "rightyo: speaker_id start enrolled=1\n",
            note("Speaker A", 1500, 0.30, "unknown"),
            'rightyo: speaker_id label="broken turn_ms=1\n',
            "rightyo: speaker_id label=X speech_ms=notanumber turn_score=0.1 state=unknown\n",
            "rightyo: speaker_id summary short=1\n",
        ]
        turns, counts = harness.parse_notes(lines)
        self.assertEqual((counts["short"], counts["offered"], counts["scored"]), (3, 4, 2))
        self.assertEqual(counts["started"], 2)
        self.assertEqual(counts["unsegmented"], 0)
        self.assertEqual([turn.session for turn in turns], [1, 1, 2])
        self.assertEqual(turns[0].label, "Speaker A")
        self.assertEqual(turns[0].route, "built_in_mic")
        self.assertEqual(turns[2].route, "unrecorded")
        self.assertEqual(
            (turns[1].speech_ms, turns[1].score, turns[1].state), (4100, 0.72, "bound")
        )


class TurnReportTests(unittest.TestCase):
    def report(self) -> dict:
        lines = [
            "rightyo: speaker_id start enrolled=1\n",
            note("Speaker A", 1500, 0.40, "unknown", "built_in_mic"),
            note("Speaker A", 2500, 0.50, "tentative", "built_in_mic"),
            note("Speaker A", 4000, 0.70, "bound", "built_in_mic"),
            note("Speaker A", 13000, 0.66, "bound", "usb_audio"),
            note("Speaker B", 3500, 0.20, "unknown"),
            "rightyo: speaker_id start enrolled=1\n",
            note("Speaker A", 5000, 0.30, "unknown"),
        ]
        turns, _ = harness.parse_notes(lines)
        return harness.turn_report(turns)

    def test_a_label_that_bound_counts_all_its_turns_as_the_owner(self):
        report = self.report()
        self.assertEqual(report["sessions_with_scored_turns"], 2)
        self.assertEqual(report["owner_labels"], 1)
        self.assertEqual(report["owner_turns"]["count"], 4)
        self.assertEqual(report["owner_turns"]["missed"], 2)
        self.assertEqual(report["owner_turns"]["miss_rate"], 0.5)
        self.assertEqual(report["owner_turns"]["at_or_above_bind"], 0.5)
        self.assertEqual(report["owner_turns"]["at_or_above_tentative"], 0.75)

    def test_labels_are_per_session_and_never_bound_ones_stay_apart(self):
        report = self.report()
        # Session 2's "Speaker A" is a different label from session 1's, and never bound.
        self.assertEqual(report["unconfirmed_labels"], 2)
        self.assertEqual(report["unconfirmed_turns"]["count"], 2)
        self.assertEqual(report["unconfirmed_turns"]["max"], 0.3)

    def test_misses_by_length_route_and_position(self):
        report = self.report()
        self.assertEqual(report["by_speech_length"]["1000-2000ms"]["miss_rate"], 1.0)
        self.assertEqual(report["by_speech_length"]["12000ms+"]["miss_rate"], 0.0)
        self.assertEqual(report["by_route"]["built_in_mic"]["count"], 3)
        self.assertEqual(report["by_route"]["usb_audio"]["missed"], 0)
        self.assertEqual(report["by_position_in_label"]["0-1turn"]["missed"], 1)
        self.assertEqual(report["by_position_in_label"]["1-3turn"]["count"], 2)
        self.assertEqual(report["by_position_in_label"]["3-10turn"]["miss_rate"], 0.0)

    def test_a_turn_bound_to_another_print_is_a_miss_and_its_score_is_left_out(self):
        lines = [
            "rightyo: speaker_id start enrolled=2\n",
            note("Speaker A", 2000, 0.52, "tentative", identity="guest"),
            note("Speaker A", 4000, 0.70, "bound"),
            note("Speaker A", 4000, 0.66, "bound", identity="guest"),
            note("Speaker B", 4000, 0.80, "bound", identity="guest"),
        ]
        report = harness.turn_report(harness.parse_notes(lines)[0])
        owner = report["owner_turns"]
        # Speaker B never bound to the owner: its 0.80 is against the guest print, not the owner's.
        self.assertEqual(report["unconfirmed_turns"]["count"], 0)
        self.assertEqual(report["unconfirmed_turns"]["scored_against_another_print"], 1)
        self.assertEqual(report["owner_labels"], 1)
        self.assertEqual((owner["turns"], owner["missed"], owner["count"]), (3, 2, 1))
        self.assertEqual(owner["scored_against_another_print"], 2)
        self.assertEqual(owner["max"], 0.7)

    def test_length_edges_are_half_open(self):
        lines = [
            "rightyo: speaker_id start enrolled=1\n",
            note("Speaker A", 2999, 0.70, "bound"),
            note("Speaker A", 3000, 0.70, "bound"),
        ]
        lengths = harness.turn_report(harness.parse_notes(lines)[0])["by_speech_length"]
        self.assertEqual(lengths["2000-3000ms"]["turns"], 1)
        self.assertEqual(lengths["3000-6000ms"]["turns"], 1)

    def test_a_logged_score_equal_to_a_threshold_is_ambiguous_not_above(self):
        # Notes are logged to three decimals: 0.600 may truly be 0.5996.
        lines = [
            "rightyo: speaker_id start enrolled=1\n",
            note("Speaker A", 4000, 0.600, "bound"),
            note("Speaker A", 4000, 0.601, "bound"),
            note("Speaker A", 4000, 0.450, "bound"),
            note("Speaker A", 4000, 0.599, "bound"),
        ]
        owner = harness.turn_report(harness.parse_notes(lines)[0])["owner_turns"]
        self.assertEqual(owner["at_or_above_bind"], 0.25)
        self.assertEqual(owner["ambiguous_at_bind"], 1)
        self.assertEqual(owner["at_or_above_tentative"], 0.75)
        self.assertEqual(owner["ambiguous_at_tentative"], 1)

    def test_only_allowlisted_routes_reach_the_report(self):
        lines = [
            "rightyo: speaker_id start enrolled=1\n",
            note("Speaker A", 4000, 0.70, "bound", route="Kitchen_iPhone"),
            note("Speaker A", 4000, 0.70, "bound", route="usb_audio"),
        ]
        routes = harness.turn_report(harness.parse_notes(lines)[0])["by_route"]
        self.assertEqual(sorted(routes), ["other", "usb_audio"])

    def test_a_session_with_no_scored_turn_still_counts_as_started(self):
        lines = [
            "rightyo: speaker_id start enrolled=1\n",
            note("Speaker A", 4000, 0.70, "bound"),
            "rightyo: speaker_id start enrolled=1\n",
            "rightyo: speaker_id summary offered=3 scored=0 short=3\n",
        ]
        turns, counts = harness.parse_notes(lines)
        self.assertEqual(counts["started"], 2)
        self.assertEqual(harness.turn_report(turns)["sessions_with_scored_turns"], 1)


class NegativeReportTests(unittest.TestCase):
    def test_segments_are_exact_non_overlapping_and_capped(self):
        samples = tone(7.5, 9000)
        self.assertEqual([len(s) for s in harness.segments(samples, 3.0)], [3 * RATE, 3 * RATE])
        self.assertEqual(len(list(harness.segments(samples, 1.0, limit=4))), 4)
        self.assertEqual(list(harness.segments(samples, 12.0)), [])

    def test_false_accepts_per_group_and_length_scored_as_a_live_turn(self):
        clips = [("near", tone(7.0, 9000)), ("far", tone(7.0, 3000)), ("far", tone(0.5, 3000))]
        report = harness.negative_report(clips, loudness_embedder, OWNER, durations=(1.0, 3.0))
        near, far = report["groups"]["near"], report["groups"]["far"]
        self.assertEqual((near["clips"], far["clips"]), (1, 2))
        self.assertEqual(
            (near["1s"]["count"], near["3s"]["count"], near["whole"]["count"]), (7, 2, 1)
        )
        self.assertEqual(near["1s"]["at_or_above_bind"], 1.0)
        self.assertEqual(far["3s"]["at_or_above_tentative"], 0.0)
        # The half-second clip is too short for any segment or a whole-clip score.
        self.assertEqual(far["whole"]["count"], 1)

    def test_nothing_shorter_than_the_configured_minimum_is_scored(self):
        clips = [("g", tone(7.0, 9000)), ("g", tone(1.5, 9000))]
        report = harness.negative_report(
            clips, loudness_embedder, OWNER, durations=(1.0, 3.0), min_seconds=2.0
        )
        group = report["groups"]["g"]
        self.assertEqual(report["lengths_below_min_turn"], ["1s"])
        self.assertNotIn("1s", group)
        self.assertEqual(group["3s"]["count"], 2)
        # The 1.5 s clip is under the 2 s minimum: only the 7 s clip is scored whole.
        self.assertEqual(group["whole"]["count"], 1)

    def test_exact_scores_keep_at_or_above_for_negatives(self):
        report = harness.negative_report(
            [("g", tone(3.0, 9000))], lambda pcm: OWNER, OWNER, durations=(3.0,)
        )
        self.assertEqual(report["groups"]["g"]["3s"]["at_or_above_bind"], 1.0)
        self.assertNotIn("ambiguous_at_bind", report["groups"]["g"]["3s"])

    def test_segments_come_from_trimmed_speech_capped_at_30_seconds(self):
        pause = array("h", bytes(2 * 3 * RATE))
        gapped = tone(3.0, 9000) + pause + tone(3.0, 9000)
        clips = [("gapped", gapped), ("long", tone(40.0, 9000))]
        report = harness.negative_report(clips, loudness_embedder, OWNER, durations=(3.0, 12.0))
        # The 3 s pause is cut to a quarter second: two 3 s segments, not three.
        self.assertEqual(report["groups"]["gapped"]["3s"]["count"], 2)
        # 40 s is capped at 30 s before cutting: two 12 s segments, not three.
        self.assertEqual(report["groups"]["long"]["12s"]["count"], 2)


class OutputTests(unittest.TestCase):
    def test_the_turns_report_holds_no_session_label(self):
        with tempfile.TemporaryDirectory() as scratch:
            log, output = Path(scratch, "child.log"), Path(scratch, "report.json")
            log.write_text(
                "rightyo: speaker_id start enrolled=1\n"
                + note("Kitchen Speaker", 4000, 0.71, "bound", "built_in_mic"),
                encoding="utf-8",
            )
            self.assertEqual(harness.main(["--output", str(output), "turns", "--log", str(log)]), 0)
            text = output.read_text(encoding="utf-8")
        self.assertNotIn("Kitchen", text)
        self.assertEqual(json.loads(text)["owner_turns"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
