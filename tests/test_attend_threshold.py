"""An optional, stricter attend bar on Jev's attend probability (rightyo#199); synthetic only."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import test_tool
from test_mvp import answer, turn

from rightyo.pipeline import ReplayRunner
from rightyo.prototype import DecisionConfigError, PrototypeConfig
from rightyo.providers import JEV_MODEL, MockProvider, ProviderError, build_request, parse_response


def response(request: dict, attend: float, label: str = "attend") -> dict:
    """An attention answer with `attend` probability (the rest on ignore) and recipient system."""
    criteria = set(request["questions"]["attention"]["criteria"])
    attention = answer(label, criteria)
    attention["probabilities"] = {
        option: attend if option == "attend" else (1 - attend if option == "ignore" else 0.0)
        for option in criteria
    }
    recipient = answer("system", set(request["questions"]["recipient"]["criteria"]))
    return {"model": JEV_MODEL, "answers": {"attention": attention, "recipient": recipient}}


class AttendBarTests(unittest.TestCase):
    def setUp(self):
        self.request = build_request(ReplayRunner(MockProvider())._state(turn()))

    def test_an_attend_below_the_bar_is_uncertain_and_at_or_above_it_stands(self):
        for attend, bar, label in (
            (0.85, None, "attend"),
            (0.85, 0.9, "uncertain"),
            (0.9, 0.9, "attend"),
            (0.95, 0.9, "attend"),
        ):
            with self.subTest(attend=attend, bar=bar):
                decision = parse_response(response(self.request, attend), self.request, 0.7, bar)
                self.assertEqual(decision.label, label)
                # The chosen label is still reported; only the action is withheld.
                self.assertEqual(decision.attention_choice, "attend")

    def test_the_bar_leaves_ignore_alone(self):
        raw = response(self.request, 0.1, label="ignore")
        raw["answers"]["recipient"] = answer(
            "other_human", set(self.request["questions"]["recipient"]["criteria"])
        )
        self.assertEqual(parse_response(raw, self.request, 0.7, 0.9).label, "ignore")

    def test_the_providers_decide_applies_its_bar(self):
        from rightyo.providers import JevProvider

        state = ReplayRunner(MockProvider())._state(turn())
        for bar, label in ((None, "attend"), (0.9, "uncertain"), (1, "uncertain"), (0, "attend")):
            provider = JevProvider(allow_hosted=True, attend_min_probability=bar)
            canned = response(build_request(state), 0.85)
            with self.subTest(bar=bar), patch.object(JevProvider, "_send", return_value=canned):
                self.assertEqual(provider.decide(state).label, label)

    def test_the_provider_validates_and_keeps_the_bar(self):
        from rightyo.providers import JevProvider

        self.assertIsNone(JevProvider(allow_hosted=True).attend_min_probability)
        self.assertEqual(
            JevProvider(allow_hosted=True, attend_min_probability=0.9).attend_min_probability, 0.9
        )
        for bad in (-0.1, 1.5):
            with self.subTest(bad=bad), self.assertRaises((ProviderError, ValueError)):
                JevProvider(allow_hosted=True, attend_min_probability=bad)


class AttendBarConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = test_tool.ConfigDecisionTests(
            "test_config_jev_with_consent_constructs_the_jev_provider"
        )
        self.config.setUp()
        self.addCleanup(self.config.doCleanups)

    def test_the_decision_section_sets_the_bar_for_listen(self):
        self.config.write({"provider": "jev", "allow_hosted": True, "attend_min_probability": 0.9})
        self.assertEqual(
            PrototypeConfig.load(self.config.tool.config).decision_attend_min_probability, 0.9
        )
        ((options, provider),) = self.config.run_listen()
        self.assertEqual(
            (options["attend_min_probability"], provider.attend_min_probability), (0.9, 0.9)
        )

    def test_absent_leaves_the_provider_as_before(self):
        self.config.write({"provider": "jev", "allow_hosted": True})
        ((options, provider),) = self.config.run_listen()
        self.assertNotIn("attend_min_probability", options)
        self.assertIsNone(provider.attend_min_probability)

    def test_a_bad_bar_or_one_without_jev_is_refused(self):
        for value in (-0.1, 1.01, "0.9", True, None):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(DecisionConfigError, "attend_min_probability"),
            ):
                self.config.write(
                    {"provider": "jev", "allow_hosted": True, "attend_min_probability": value}
                )
                PrototypeConfig.load(self.config.tool.config)
        self.config.write(
            {"provider": "mock", "allow_hosted": False, "attend_min_probability": 0.9}
        )
        with self.assertRaisesRegex(DecisionConfigError, "only to the jev"):
            PrototypeConfig.load(self.config.tool.config)


if __name__ == "__main__":
    unittest.main()
