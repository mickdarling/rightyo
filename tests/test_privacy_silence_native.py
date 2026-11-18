"""Opt-in native #291 qualification using explicitly supplied authored TTS only.

No downloads, capture or ASR. The default suite skips this test. Provisioned paths:
RIGHTYO_TEST_DIARIZER_LIBRARY, RIGHTYO_TEST_DIARIZER_MODEL, RIGHTYO_TEST_AUTHORED_VOICES
(the last contains authored-voice-0.wav and authored-voice-1.wav, mono PCM16 16 kHz).
"""

import os
import unittest
import wave
from pathlib import Path

from rightyo.live_audio import LiveConfig, _Diarizer

KEYS = (
    "RIGHTYO_TEST_DIARIZER_LIBRARY",
    "RIGHTYO_TEST_DIARIZER_MODEL",
    "RIGHTYO_TEST_AUTHORED_VOICES",
)


class UnusedASR:
    def transcribe(self, *_):
        raise AssertionError("native qualification never transcribes")


@unittest.skipUnless(
    all(os.environ.get(key) for key in KEYS), "explicit authored native assets only"
)
class NativeSilenceTests(unittest.TestCase):
    def test_long_private_substitution_keeps_clock_and_speaker_clusters(self):
        voices = []
        for index in (0, 1):
            path = Path(os.environ[KEYS[2]]) / f"authored-voice-{index}.wav"
            with wave.open(str(path), "rb") as audio:
                self.assertEqual(
                    (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()),
                    (16000, 1, 2),
                )
                voices.append(audio.readframes(audio.getnframes()))
        for gap in (60, 300):
            for different in (False, True):
                with self.subTest(silence_s=gap, different_voice=different):
                    config = LiveConfig(
                        "authored-native-silence",
                        transcriber=UnusedASR(),
                        diarization_library=os.environ[KEYS[0]],
                        diarization_model=os.environ[KEYS[1]],
                        timeout_seconds=120,
                    )
                    native = _Diarizer(config)
                    try:

                        def push(pcm):
                            for offset in range(0, len(pcm), 640):
                                native.push(pcm[offset : offset + 640])

                        first, last = voices[0], voices[int(different)]
                        push(first)
                        push(bytes(2000 * 32))  # native lookahead
                        initial = native.segments()  # before rolling timeline evicts A
                        push(bytes((gap * 1000 - 2000) * 32))
                        start = len(first) // 32 + gap * 1000
                        push(last)
                        push(bytes(3000 * 32))
                        final = native.finish()

                        def dominant(segments, left, right):
                            covered = {}
                            for segment in segments:
                                overlap = max(
                                    0,
                                    min(right, segment["end_ms"]) - max(left, segment["start_ms"]),
                                )
                                if overlap:
                                    label = segment["speaker"]
                                    covered[label] = covered.get(label, 0) + overlap
                            self.assertTrue(covered)
                            return max(covered, key=covered.get)

                        a = dominant(initial, 0, len(first) // 32)
                        b = dominant(final, start, start + len(last) // 32)
                        self.assertEqual(a != b, different)
                        # Native frame rounding/lookahead may extend a segment slightly.
                        self.assertLess(
                            abs(max(s["end_ms"] for s in final) - (start + len(last) // 32)), 200
                        )
                    finally:
                        native.close()
