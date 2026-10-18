"""Built-in scenes selectable per session with `listen --scene` (rightyo#199); synthetic only."""

from __future__ import annotations

import argparse
import unittest
from unittest.mock import patch

import test_turn_merge

from rightyo.addressedness import CLEAR_REQUESTS_SCENE, DEFAULT_SCENE, MAX_SCENE_CHARS, SCENES
from rightyo.cli import main


class ScenePresetTests(unittest.TestCase):
    # Borrowed by attribute, not imported by name, so unittest does not collect them twice.
    setUp = test_turn_merge.DecisionPathTests.setUp
    run_listen = test_turn_merge.DecisionPathTests.run_listen

    def listen_with(self, scene, **extra):
        """`run_listen` with `--scene` set as the parser would set it."""

        def namespace(**options):
            return argparse.Namespace(scene=scene, **options)

        with patch("test_turn_merge.Namespace", namespace):
            _, provider = self.run_listen(**extra)
        return provider.states[0].get("scene")

    def test_the_presets_are_the_default_and_clear_requests(self):
        self.assertEqual(SCENES, {"default": DEFAULT_SCENE, "clear-requests": CLEAR_REQUESTS_SCENE})
        self.assertLessEqual(len(CLEAR_REQUESTS_SCENE), MAX_SCENE_CHARS)

    def test_a_preset_reaches_the_decision_and_replaces_the_configured_scene(self):
        jev = {"provider": "jev", "allow_hosted": True, "scene": "A shared office."}
        self.assertEqual(self.listen_with(None, decision=jev), "A shared office.")
        self.assertEqual(self.listen_with("clear-requests", decision=jev), CLEAR_REQUESTS_SCENE)
        self.assertEqual(self.listen_with("default", decision=jev), DEFAULT_SCENE)

    def test_the_command_line_accepts_only_the_built_in_names(self):
        base = ["listen", "--config", "c.json", "--mode", "demo"]
        for argv in ([*base, "--scene", "A shared office."], [*base, "--scene"]):
            with self.subTest(argv=argv), self.assertRaises(SystemExit) as stopped:
                with patch("sys.stderr"):
                    main(argv)
            self.assertEqual(stopped.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
