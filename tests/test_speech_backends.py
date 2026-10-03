"""Offline backend-selection tests: mocked HTTP, no network, credentials, models or audio."""

import array
import io
import json
import os
import tempfile
import threading
import time
import unittest
import urllib.error
import wave
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock, patch

from rightyo.contracts import ContractError, identifier
from rightyo.credentials import CredentialError, load_diarizer_api_key, load_transcriber_api_key
from rightyo.live_audio import (
    BYTES_PER_MS,
    LiveAudioError,
    LiveConfig,
    LiveProcessor,
    NemotronCppDiarizer,
    WhisperCppTranscriber,
    _attribute,
)
from rightyo.pipeline import ReplayRunner
from rightyo.prototype import PrototypeConfig, PrototypeController, PrototypeError
from rightyo.providers import Diarizer, MockProvider, Transcriber
from rightyo.speech_backends import (
    DEEPGRAM_ENDPOINT,
    MAX_RESPONSE_BYTES,
    DeepgramDiarizer,
    HostedSpeechError,
    OpenAICompatibleTranscriber,
    _NoRedirect,
    deepgram_timeline,
    diarizer_factory,
    openai_units,
    provenance_id,
    speech_summary,
    transcriber_factory,
)
from rightyo.tool import listen
from rightyo.tool_events import SpeechEvents

KEY = "synthetic-test-key-do-not-use"
PRIVATE = "private utterance text"
ENDPOINT = "https://transcribe.example.test/v1/audio/transcriptions"
VOICE = array.array("h", [5000] * 320).tobytes()


def load_key(**_options):
    return KEY


class FakeProcess:
    """A `security` stand-in that hangs until terminated; never a real Keychain."""

    def __init__(self, args, **_options):
        self.args = args
        self.stdout = io.BytesIO(b"")
        self.terminated = False
        self.returncode = None

    def poll(self):
        if not self.terminated:
            return None
        self.returncode = -15
        return self.returncode

    def terminate(self):
        self.terminated = True

    kill = terminate

    def wait(self, timeout=None):
        return self.poll()


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def read(self, limit):
        chunk, self.payload = self.payload[:limit], self.payload[limit:]
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.closed = True
        return False

    def close(self):
        self.closed = True


def patched_opener(payload=None, error=None):
    """Replace the module's opener; returns the recorded requests."""
    requests = []

    def open_request(request, timeout):
        requests.append((request, timeout))
        if error is not None:
            raise error
        return FakeResponse(payload)

    opener = Mock(open=open_request)
    return patch("urllib.request.build_opener", return_value=opener), requests


def http_error(code, body=PRIVATE.encode()):
    return urllib.error.HTTPError("https://x.test", code, PRIVATE, {}, io.BytesIO(body))


class StubDiarizer:
    speaker = 2

    def __init__(self, config=None):
        self.frames = []
        self.closed = False

    def push(self, pcm):
        self.frames.append(pcm)

    def segments(self):
        pushed_ms = sum(map(len, self.frames)) // BYTES_PER_MS
        return [{"start_ms": 0, "end_ms": pushed_ms, "speaker": self.speaker}]

    def finish(self):
        return self.segments()

    def close(self):
        self.closed = True


class StubTranscriber:
    recognizer_id = "stub-recognizer"

    def __init__(self, config=None):
        self.calls = []

    def transcribe(self, pcm, register=None):
        self.calls.append((pcm, register))
        return [{"text": " Hi.", "start_ms": 0, "end_ms": len(pcm) // BYTES_PER_MS}]


class ProtocolAndSelectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.asset = Path(self.directory.name) / "supplied-local-asset"
        self.asset.touch()

    def local_config(self, **overrides):
        values = dict(
            session_id="selection-test",
            whisper_executable=self.asset,
            whisper_model=self.asset,
            diarization_library=self.asset,
            diarization_model=self.asset,
            provenance="causal-replay",
        )
        return LiveConfig(**(values | overrides))

    def test_local_wrappers_and_hosted_backends_satisfy_the_protocols(self):
        config = self.local_config()
        self.assertIsInstance(WhisperCppTranscriber(config), Transcriber)
        self.assertTrue(issubclass(NemotronCppDiarizer, Diarizer))
        self.assertIsInstance(
            OpenAICompatibleTranscriber(endpoint=ENDPOINT, model="m", allow_hosted=True),
            Transcriber,
        )
        self.assertIsInstance(DeepgramDiarizer(allow_hosted=True), Diarizer)
        self.assertEqual(WhisperCppTranscriber.recognizer_id, "whisper.cpp-live-window")

    def test_whisper_wrapper_derives_units_from_the_pinned_cli_document(self):
        config = self.local_config()
        document = {"transcription": [{"text": " Hello.", "offsets": {"from": 0, "to": 150}}]}
        with patch("rightyo.live_audio._transcribe", return_value=document) as transcribe:
            register = Mock()
            units = WhisperCppTranscriber(config).transcribe(bytes(200 * BYTES_PER_MS), register)
        self.assertEqual(units, [{"text": " Hello.", "start_ms": 0, "end_ms": 150}])
        self.assertEqual(transcribe.call_args.args[0], config)
        self.assertIs(transcribe.call_args.args[2], register)

    def test_processor_uses_supplied_instances_or_factories_and_labels_turns(self):
        turns = []
        transcriber, diarizer = StubTranscriber(), StubDiarizer()
        seen = []

        def diarizer_factory_(config):
            seen.append(config)
            return diarizer

        config = LiveConfig(
            "selection-test",
            provenance="causal-replay",
            transcriber=transcriber,
            diarizer=diarizer_factory_,
        )
        processor = LiveProcessor(config, turns.append)
        self.addCleanup(processor.close)
        for _ in range(10):
            processor.push_pcm16(VOICE)
        processor.finish()
        self.assertEqual(seen, [config])
        self.assertEqual(len(diarizer.frames), 10)
        self.assertEqual(len(transcriber.calls), 1)
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0].recognizer_id, "stub-recognizer")
        self.assertEqual(turns[0].speaker_id, "Speaker B")
        self.assertEqual(turns[0].text, "Hi.")
        self.assertEqual(turns[0].speaker_provenance, "diarization-timeline")
        self.assertTrue(diarizer.closed)

    def test_per_utterance_labels_are_declared_while_the_native_path_is_unchanged(self):
        turns = []
        per_utterance = StubDiarizer()
        per_utterance.speaker_provenance = "diarization-utterance"
        for diarizer, expected in (
            (per_utterance, "diarization-utterance"),
            (StubDiarizer(), "diarization-timeline"),
        ):
            with self.subTest(expected=expected):
                turns.clear()
                processor = LiveProcessor(
                    LiveConfig(
                        "selection-test",
                        provenance="causal-replay",
                        transcriber=StubTranscriber(),
                        diarizer=diarizer,
                    ),
                    turns.append,
                )
                for _ in range(10):
                    processor.push_pcm16(VOICE)
                processor.finish()
                self.assertEqual([turn.speaker_provenance for turn in turns], [expected])
                self.assertEqual(
                    turns[0].speaker_id,
                    "u1 Speaker B" if expected == "diarization-utterance" else "Speaker B",
                )
        self.assertEqual(DeepgramDiarizer.speaker_provenance, "diarization-utterance")
        with patch("rightyo.live_audio._transcribe", return_value={"transcription": []}):
            processor = LiveProcessor(self.local_config(diarizer=StubDiarizer()), turns.append)
            self.addCleanup(processor.close)
            self.assertEqual(
                getattr(processor._diarizer, "speaker_provenance", "diarization-timeline"),
                "diarization-timeline",
            )

    def test_backend_classes_lambdas_and_instances_are_all_accepted(self):
        turns = []
        for transcriber, diarizer in (
            (StubTranscriber, StubDiarizer),
            (lambda config: StubTranscriber(), lambda config: StubDiarizer()),
            (StubTranscriber(), StubDiarizer()),
        ):
            with self.subTest(transcriber=transcriber):
                turns.clear()
                processor = LiveProcessor(
                    LiveConfig(
                        "factory-test",
                        provenance="causal-replay",
                        transcriber=transcriber,
                        diarizer=diarizer,
                    ),
                    turns.append,
                )
                self.assertIsInstance(processor._transcriber, StubTranscriber)
                self.assertIsInstance(processor._diarizer, StubDiarizer)
                for _ in range(10):
                    processor.push_pcm16(VOICE)
                processor.finish()
                self.assertEqual(len(turns), 1)

    def test_adapter_timelines_are_validated_at_the_common_boundary(self):
        cases = {
            "end past the stream": [{"start_ms": 0, "end_ms": 900000, "speaker": 1}],
            "negative start": [{"start_ms": -1, "end_ms": 100, "speaker": 1}],
            "reversed": [{"start_ms": 150, "end_ms": 100, "speaker": 1}],
            "float": [{"start_ms": 0.0, "end_ms": 100, "speaker": 1}],
            "speaker zero": [{"start_ms": 0, "end_ms": 100, "speaker": 0}],
            "speaker too large": [{"start_ms": 0, "end_ms": 100, "speaker": 703}],
            "not a dict": ["segment"],
            "not a list": {"start_ms": 0, "end_ms": 100, "speaker": 1},
            "over count": [{"start_ms": 0, "end_ms": 1, "speaker": 1}] * 18001,
        }
        for name, timeline in cases.items():
            with self.subTest(case=name):
                turns = []
                diarizer = StubDiarizer()
                diarizer.segments = lambda timeline=timeline: timeline
                processor = LiveProcessor(
                    LiveConfig(
                        "timeline-test",
                        provenance="causal-replay",
                        transcriber=StubTranscriber(),
                        diarizer=diarizer,
                    ),
                    turns.append,
                )
                for _ in range(10):
                    processor.push_pcm16(VOICE)
                with self.assertRaisesRegex(LiveAudioError, "diarizer timeline"):
                    processor.finish()
                self.assertTrue(processor.failed)
                self.assertEqual(turns, [])

    def test_speaker_numbers_beyond_z_become_column_style_labels(self):
        from rightyo.live_audio import _speaker_label

        expected = {1: "A", 26: "Z", 27: "AA", 28: "AB", 52: "AZ", 53: "BA", 702: "ZZ"}
        for number, label in expected.items():
            self.assertEqual(_speaker_label(number), label)
        turns = []
        for number in (26, 27, 52, 53, 702):
            with self.subTest(speaker=number):
                turns.clear()
                diarizer = StubDiarizer()
                diarizer.speaker = number
                processor = LiveProcessor(
                    LiveConfig(
                        "label-test",
                        provenance="causal-replay",
                        transcriber=StubTranscriber(),
                        diarizer=diarizer,
                    ),
                    turns.append,
                )
                for _ in range(10):
                    processor.push_pcm16(VOICE)
                processor.finish()
                self.assertEqual(turns[0].speaker_id, "Speaker " + expected[number])
                identifier(turns[0].speaker_id, "speaker_id")

    def test_adapter_units_are_validated_at_the_common_boundary(self):
        duration = 10 * 640 // BYTES_PER_MS
        cases = {
            "negative start": [{"text": " a", "start_ms": -1, "end_ms": 10}],
            "end past the utterance": [{"text": " a", "start_ms": 0, "end_ms": duration + 1}],
            "reversed interval": [{"text": " a", "start_ms": 50, "end_ms": 40}],
            "reordered units": [
                {"text": " b", "start_ms": 100, "end_ms": 150},
                {"text": " a", "start_ms": 0, "end_ms": 50},
            ],
            "float timestamps": [{"text": " a", "start_ms": 0.0, "end_ms": 50}],
            "not a list": {"text": " a"},
        }
        for name, units in cases.items():
            with self.subTest(case=name):
                turns = []
                transcriber = StubTranscriber()
                transcriber.transcribe = lambda pcm, register=None, units=units: units
                processor = LiveProcessor(
                    LiveConfig(
                        "boundary-test",
                        provenance="causal-replay",
                        transcriber=transcriber,
                        diarizer=StubDiarizer(),
                    ),
                    turns.append,
                )
                for _ in range(10):
                    processor.push_pcm16(VOICE)
                with self.assertRaisesRegex(LiveAudioError, "recognizer") as caught:
                    processor.finish()
                self.assertTrue(processor.failed)
                self.assertEqual(turns, [])
                self.assertNotIn(" a", str(caught.exception))

    def test_local_assets_are_required_only_by_the_local_defaults(self):
        with self.assertRaisesRegex(LiveAudioError, "existing"):
            LiveConfig("selection-test")
        with self.assertRaisesRegex(LiveAudioError, "existing"):
            LiveConfig("selection-test", transcriber=StubTranscriber())
        with self.assertRaisesRegex(LiveAudioError, "existing"):
            LiveConfig("selection-test", diarizer=StubDiarizer())
        config = LiveConfig(
            "selection-test", transcriber=StubTranscriber(), diarizer=StubDiarizer()
        )
        self.assertIsNone(config.whisper_executable)
        # A supplied but missing asset is still rejected, and a backend must be usable.
        with self.assertRaisesRegex(LiveAudioError, "existing"):
            self.local_config(transcriber=StubTranscriber(), whisper_model="/missing")
        for name in ("transcriber", "diarizer"):
            with self.assertRaisesRegex(LiveAudioError, name):
                self.local_config(**{name: object()})

    def test_per_utterance_labels_never_merge_into_one_participant(self):
        turns = []
        per_utterance = StubDiarizer()
        per_utterance.speaker_provenance = "diarization-utterance"
        per_utterance.speaker = 1
        processor = LiveProcessor(
            LiveConfig(
                "namespace-test",
                provenance="causal-replay",
                transcriber=StubTranscriber(),
                diarizer=per_utterance,
            ),
            turns.append,
        )
        self.addCleanup(processor.close)
        for _ in range(2):
            for _ in range(10):
                processor.push_pcm16(VOICE)
            for _ in range(72):
                processor.push_pcm16(bytes(640))
        # The service labelled both utterances "Speaker A"; the turns must not match.
        self.assertEqual([turn.speaker_id for turn in turns], ["u1 Speaker A", "u2 Speaker A"])
        for turn in turns:
            identifier(turn.speaker_id, "speaker_id")
        states = []

        class Recorder:
            def decide(self, state):
                states.append(state)
                return MockProvider().decide(state)

        runner = ReplayRunner(Recorder())
        for turn in turns:
            runner.process(turn)
        self.assertEqual(states[-1]["known_participants"], ["u1 Speaker A", "u2 Speaker A"])
        self.assertEqual(len(states[-1]["known_participants"]), 2)


class HostedTranscriberTests(unittest.TestCase):
    def transcriber(self, **overrides):
        values = dict(endpoint=ENDPOINT, model="whisper-1", allow_hosted=True, load_key=load_key)
        return OpenAICompatibleTranscriber(**(values | overrides))

    def test_hosted_use_requires_explicit_consent_and_https(self):
        with self.assertRaisesRegex(HostedSpeechError, "consent"):
            OpenAICompatibleTranscriber(endpoint=ENDPOINT, model="whisper-1")
        with self.assertRaisesRegex(HostedSpeechError, "consent"):
            OpenAICompatibleTranscriber(endpoint=ENDPOINT, model="whisper-1", allow_hosted=1)
        for endpoint in (
            "http://transcribe.example.test/v1",
            "https://u:p@x.test/v1",
            ENDPOINT + "?model=x",
            ENDPOINT + "#fragment",
            ENDPOINT + "?",
            ENDPOINT + "#",
            5,
        ):
            with self.subTest(endpoint=endpoint), self.assertRaisesRegex(LiveAudioError, "https"):
                self.transcriber(endpoint=endpoint)
        with self.assertRaises(LiveAudioError):
            self.transcriber(model="bad model")
        with self.assertRaises(LiveAudioError):
            self.transcriber(timeout_seconds=0)

    def test_request_shape_follows_the_documented_multipart_schema(self):
        pcm = bytes(500 * BYTES_PER_MS)
        payload = json.dumps(
            {
                "text": "Hello, world.",
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 0.2},
                    {"word": "world", "start": 0.25, "end": 0.6},
                ],
            }
        ).encode()
        opener, requests = patched_opener(payload)
        with opener:
            units = self.transcriber(language="en", timeout_seconds=7).transcribe(pcm, Mock())
        self.assertEqual(
            units,
            [
                {"text": " Hello,", "start_ms": 0, "end_ms": 200},
                {"text": " world.", "start_ms": 250, "end_ms": 500},
            ],
        )
        (request, timeout), *_rest = requests
        self.assertEqual(len(requests), 1)
        # The transport gets the budget left after the credential lookup, never more.
        self.assertLessEqual(timeout, 7)
        self.assertGreater(timeout, 6.5)
        self.assertEqual(request.full_url, ENDPOINT)
        self.assertEqual(request.get_method(), "POST")
        # The credential is removed from the request object after the exchange.
        self.assertFalse(request.has_header("Authorization"))
        content_type = request.get_header("Content-type")
        self.assertTrue(content_type.startswith("multipart/form-data; boundary="))
        boundary = content_type.split("boundary=")[1].encode()
        parts = request.data.split(b"--" + boundary)
        fields = {}
        for part in parts[1:-1]:
            header, _, value = part.partition(b"\r\n\r\n")
            name = header.split(b'name="')[1].split(b'"')[0].decode()
            fields.setdefault(name, []).append(value[:-2])
        self.assertEqual(fields["model"], [b"whisper-1"])
        self.assertEqual(fields["response_format"], [b"verbose_json"])
        self.assertEqual(fields["timestamp_granularities[]"], [b"word", b"segment"])
        self.assertEqual(fields["language"], [b"en"])
        self.assertIn(b'filename="utterance.wav"', parts[-2])
        with wave.open(io.BytesIO(fields["file"][0]), "rb") as audio:
            self.assertEqual(
                (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()),
                (1, 2, 16000),
            )
            self.assertEqual(audio.readframes(audio.getnframes()), pcm)
        self.assertEqual(parts[-1], b"--\r\n")

    def test_authorization_header_carries_the_bearer_key_at_point_of_use_only(self):
        seen = []

        def open_request(request, timeout):
            seen.append(request.get_header("Authorization"))
            return FakeResponse(b'{"text": ""}')

        with patch("urllib.request.build_opener", return_value=Mock(open=open_request)):
            self.assertEqual(self.transcriber().transcribe(bytes(640)), [])
        self.assertEqual(seen, ["Bearer " + KEY])

    def test_failures_are_sanitized_and_bounded(self):
        pcm = bytes(640)
        cases = [
            (http_error(401), "HTTP 401"),
            (http_error(429), "temporarily unavailable"),
            (urllib.error.URLError(PRIVATE), "connection failed"),
            (TimeoutError(PRIVATE), "connection failed"),
            (HostedSpeechError("Hosted speech redirect refused"), "redirect refused"),
        ]
        for error, expected in cases:
            opener, _ = patched_opener(error=error)
            with self.subTest(expected=expected), opener:
                with self.assertRaisesRegex(HostedSpeechError, expected) as caught:
                    self.transcriber().transcribe(pcm)
                self.assertNotIn(PRIVATE, str(caught.exception))
                self.assertNotIn(KEY, str(caught.exception))
                self.assertNotIn(ENDPOINT, str(caught.exception))
                self.assertIsNone(caught.exception.__context__)
        opener, _ = patched_opener(b"{" * (MAX_RESPONSE_BYTES + 1))
        with opener, self.assertRaisesRegex(HostedSpeechError, "size limit"):
            self.transcriber().transcribe(pcm)
        for payload in (b"not json", b"[]", json.dumps({"text": PRIVATE}).encode()):
            opener, _ = patched_opener(payload)
            with self.subTest(payload=payload), opener:
                with self.assertRaises(HostedSpeechError) as caught:
                    self.transcriber().transcribe(pcm)
                self.assertNotIn(PRIVATE, str(caught.exception))

    def test_missing_credential_or_cancellation_sends_nothing(self):
        def missing(**_options):
            raise CredentialError("Synthetic missing credential")

        opener, requests = patched_opener(b"{}")
        with opener:
            with self.assertRaisesRegex(HostedSpeechError, "credential is unavailable") as caught:
                self.transcriber(load_key=missing).transcribe(bytes(640))
            self.assertIsNone(caught.exception.__context__)
            with self.assertRaisesRegex(HostedSpeechError, "cancelled"):
                self.transcriber(cancelled=lambda: True).transcribe(bytes(640))
            with self.assertRaisesRegex(HostedSpeechError, "Invalid utterance audio"):
                self.transcriber().transcribe(b"")
        self.assertEqual(requests, [])

    def test_recognizer_id_names_the_configured_model_safely(self):
        first = self.transcriber(model="whisper-1").recognizer_id
        second = self.transcriber(model="gpt-4o-transcribe").recognizer_id
        self.assertRegex(first, r"^hosted-openai-compatible whisper-1 [0-9a-f]{12}$")
        self.assertNotEqual(first, second)
        odd = self.transcriber(model="org/model:v1.2").recognizer_id
        self.assertRegex(odd, r"^hosted-openai-compatible org-model-v1.2 [0-9a-f]{12}$")
        # Sanitization alone would collide; the hash of the original name keeps them apart.
        self.assertNotEqual(
            self.transcriber(model="org/model:v1").recognizer_id,
            self.transcriber(model="org-model-v1").recognizer_id,
        )
        long = self.transcriber(model="m" * 128).recognizer_id
        self.assertEqual(len(long), 96)
        self.assertNotEqual(
            provenance_id("p", "a" * 199 + "b"), provenance_id("p", "a" * 199 + "c")
        )
        self.assertLessEqual(len(provenance_id("p", "a" * 200)), 96)
        for value in (first, second, odd, long):
            identifier(value, "recognizer_id")
            self.assertNotIn("example.test", value)
        self.assertRegex(
            provenance_id("hosted-deepgram", "nova-3", "latest"),
            r"^hosted-deepgram nova-3 latest [0-9a-f]{12}$",
        )
        diarizer = DeepgramDiarizer(allow_hosted=True, model="nova-2", diarize_model="v2")
        self.assertRegex(diarizer.diarizer_id, r"^hosted-deepgram nova-2 v2 [0-9a-f]{12}$")
        self.assertEqual(diarizer.speaker_provenance, "diarization-utterance")
        turns = []
        document = {"text": "Hi.", "words": [{"word": "Hi", "start": 0.0, "end": 0.1}]}
        opener, _ = patched_opener(json.dumps(document).encode())
        with opener:
            processor = LiveProcessor(
                LiveConfig(
                    "provenance-test",
                    provenance="causal-replay",
                    transcriber=self.transcriber(model="whisper-1"),
                    diarizer=StubDiarizer(),
                ),
                turns.append,
            )
            for _ in range(10):
                processor.push_pcm16(VOICE)
            processor.finish()
        self.assertEqual([turn.recognizer_id for turn in turns], [first])

    def blocking_response(self, payload=b'{"text": ""}', pause=None):
        """A response whose read blocks until the connection is shut (or `pause` elapses)."""
        released = threading.Event()

        class Socket:
            def __init__(self):
                self.shut = False

            def shutdown(self, _how):
                self.shut = True
                released.set()

        class Blocking(FakeResponse):
            def __init__(self):
                super().__init__(payload)
                # Mock auto-attributes would satisfy `shutdown`; the chain must be strict.
                self.fp = type("Layer", (), {})()
                self.fp.raw = type("Layer", (), {})()
                self.fp.raw._sock = Socket()

            def read(self, limit):
                if pause is None:
                    released.wait(5)
                    raise OSError("connection shut")
                time.sleep(pause)
                return FakeResponse.read(self, limit)

        return Blocking()

    def test_blocked_body_is_abandoned_at_the_wall_clock_deadline_and_closed(self):
        clock = [0.0]

        def monotonic():
            clock[0] += 10.0
            return clock[0]

        response = self.blocking_response()
        with patch("urllib.request.build_opener", return_value=Mock(open=lambda *a, **k: response)):
            with patch("rightyo.speech_backends.time.monotonic", monotonic):
                with self.assertRaisesRegex(HostedSpeechError, "deadline") as caught:
                    self.transcriber(timeout_seconds=30).transcribe(bytes(640))
        self.assertTrue(response.fp.raw._sock.shut)
        self.assertTrue(response.closed)
        self.assertIsNone(caught.exception.__context__)
        self.assertNotIn(ENDPOINT, str(caught.exception))

    def test_a_pause_shorter_than_the_budget_is_tolerated_mid_body(self):
        response = self.blocking_response(pause=0.3)
        with patch("urllib.request.build_opener", return_value=Mock(open=lambda *a, **k: response)):
            self.assertEqual(self.transcriber(timeout_seconds=30).transcribe(bytes(640)), [])
        self.assertFalse(response.fp.raw._sock.shut)
        self.assertTrue(response.closed)

    def test_cancellation_mid_read_stops_the_request_and_closes_it(self):
        checks = []
        response = self.blocking_response()
        with patch("urllib.request.build_opener", return_value=Mock(open=lambda *a, **k: response)):
            with self.assertRaisesRegex(HostedSpeechError, "cancelled"):
                self.transcriber(
                    cancelled=lambda: checks.append(1) is None and len(checks) >= 4
                ).transcribe(bytes(640))
        self.assertTrue(response.fp.raw._sock.shut)
        self.assertTrue(response.closed)

    def test_keychain_lookup_is_cancelled_or_bounded_by_the_hosted_deadline(self):
        """`post` forwards the cancellation guard and the hosted budget into the lookup."""
        polls = []
        spawned = []

        def popen(args, **options):
            spawned.append(FakeProcess(args, **options))
            return spawned[-1]

        opener, requests = patched_opener(b"{}")
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("rightyo.credentials.sys.platform", "darwin"),
            patch("rightyo.credentials.subprocess.Popen", side_effect=popen),
            patch("rightyo.credentials.time.sleep", lambda _s: polls.append(1)),
            opener,
        ):
            transcriber = self.transcriber(
                load_key=load_transcriber_api_key, cancelled=lambda: len(polls) >= 2
            )
            with self.assertRaisesRegex(HostedSpeechError, "cancelled") as caught:
                transcriber.transcribe(bytes(640))
            self.assertIsNone(caught.exception.__context__)
            self.assertTrue(spawned[0].terminated)
            self.assertEqual(len(polls), 2)
            clock = [0.0]

            def monotonic():
                clock[0] += 20.0
                return clock[0]

            with patch("rightyo.credentials.time.monotonic", monotonic):
                slow = self.transcriber(load_key=load_transcriber_api_key, timeout_seconds=30)
                with self.assertRaisesRegex(HostedSpeechError, "credential is unavailable"):
                    slow.transcribe(bytes(640))
            # A 120-second default would have survived three 20-second ticks; the hosted
            # 30-second budget ended the lookup on the second check.
            self.assertTrue(spawned[1].terminated)
        self.assertEqual(requests, [])
        self.assertNotIn(KEY, "".join(str(process.args) for process in spawned))

    def test_transport_receives_only_the_budget_the_credential_lookup_left(self):
        clock = [0.0]

        def monotonic():
            return clock[0]

        def slow_key(**_options):
            clock[0] += 25.0  # the Keychain lookup consumed most of a 30-second budget
            return KEY

        class Layer:
            def __init__(self, **attributes):
                self.__dict__.update(attributes)

        class Socket:
            def __init__(self):
                self.timeouts = []

            def settimeout(self, value):
                self.timeouts.append(value)

        socket_ = Socket()
        response = FakeResponse(b'{"text": ""}')
        # urllib's chain: response.fp (HTTPResponse).fp (BufferedReader).raw (SocketIO)._sock
        response.fp = Layer(fp=Layer(raw=Layer(_sock=socket_)))
        seen = []

        def open_request(request, timeout):
            seen.append(timeout)
            return response

        with patch("urllib.request.build_opener", return_value=Mock(open=open_request)):
            with patch("rightyo.speech_backends.time.monotonic", monotonic):
                self.transcriber(load_key=slow_key, timeout_seconds=30).transcribe(bytes(640))
        self.assertEqual(seen, [5.0])

        def exhausting_key(**_options):
            clock[0] += 31.0
            return KEY

        seen.clear()
        with patch("urllib.request.build_opener", return_value=Mock(open=open_request)):
            with patch("rightyo.speech_backends.time.monotonic", monotonic):
                with self.assertRaisesRegex(HostedSpeechError, "deadline") as caught:
                    self.transcriber(load_key=exhausting_key, timeout_seconds=30).transcribe(
                        bytes(640)
                    )
        self.assertEqual(seen, [])
        self.assertNotIn(KEY, str(caught.exception))

    def test_redirects_are_refused_before_forwarding_credentials(self):
        source = Mock()
        with self.assertRaisesRegex(HostedSpeechError, "redirect refused"):
            _NoRedirect().redirect_request(Mock(), source, 302, "Found", {}, "https://o.test")
        source.close.assert_called_once()

    def test_units_restore_punctuation_only_when_words_align_and_fall_back_to_segments(self):
        words = [
            {"word": "I'm", "start": 0.1, "end": 0.3},
            {"word": "here", "start": 0.3, "end": 0.5},
        ]
        self.assertEqual(
            [u["text"] for u in openai_units({"text": '"I\'m here!"', "words": words}, 600)],
            [" \"I'm", ' here!"'],
        )
        # Words that do not reproduce the text in order are never used as the transcript:
        # complete, consistent segments are used instead, or the response is rejected.
        complete = [
            {"text": "I'm here.", "start": 0, "end": 0.4},
            {"text": "Are you?", "start": 0.4, "end": 0.6},
        ]
        incomplete = [{"word": "I'm", "start": 0.1, "end": 0.3}]
        misordered = [
            {"word": "here", "start": 0.3, "end": 0.5},
            {"word": "I'm", "start": 0.1, "end": 0.3},
        ]
        for bad_words in (incomplete, words):
            with self.subTest(words=bad_words):
                units = openai_units(
                    {"text": "I'm here. Are you?", "words": bad_words, "segments": complete},
                    600,
                )
                self.assertEqual([u["text"] for u in units], [" I'm here.", " Are you?"])
                with self.assertRaisesRegex(HostedSpeechError, "inconsistent word timing"):
                    openai_units({"text": "I'm here. Are you?", "words": bad_words}, 600)
        # Words running backwards in time are rejected outright, segments or not.
        for document in (
            {"text": "I'm here. Are you?", "words": misordered, "segments": complete},
            {"text": "here I'm", "words": misordered},
        ):
            with self.subTest(document=document):
                with self.assertRaisesRegex(HostedSpeechError, "inconsistent word timing"):
                    openai_units(document, 600)
        backwards_segments = [
            {"text": "A.", "start": 0.3, "end": 0.5},
            {"text": "B.", "start": 0.0, "end": 0.2},
        ]
        with self.assertRaisesRegex(HostedSpeechError, "inconsistent segments"):
            openai_units({"text": "A. B.", "segments": backwards_segments}, 600)
        overlapping = [
            {"word": "I'm", "start": 0.1, "end": 0.32},
            {"word": "here", "start": 0.3, "end": 0.5},
        ]
        self.assertEqual(len(openai_units({"text": "I'm here", "words": overlapping}, 600)), 2)
        # An empty transcript beside timed units is a contradiction, not silence.
        timed_word = [{"word": "x", "start": 0.0, "end": 0.1}]
        timed_segment = [{"text": "x", "start": 0.0, "end": 0.1}]
        for contradictory in (
            {"text": "", "words": timed_word},
            {"text": "   ", "segments": timed_segment},
            {"text": "", "words": timed_word, "segments": timed_segment},
        ):
            with self.subTest(contradictory=contradictory):
                with self.assertRaisesRegex(HostedSpeechError, "inconsistent"):
                    openai_units(contradictory, 600)
        for silent in ({"text": ""}, {"text": " ", "words": [], "segments": []}):
            with self.subTest(silent=silent):
                self.assertEqual(openai_units(silent, 600), [])
        segments = {"text": "A. B.", "segments": [{"text": "A. B.", "start": 0, "end": 0.4}]}
        self.assertEqual(
            openai_units(segments, 300), [{"text": " A. B.", "start_ms": 0, "end_ms": 300}]
        )
        with self.assertRaisesRegex(HostedSpeechError, "inconsistent segments"):
            openai_units(
                {"text": "A. B.", "segments": [{"text": "A.", "start": 0, "end": 0.4}]}, 300
            )
        for bad in (
            {"text": "x"},
            {"text": "x", "words": [{"word": "x", "start": 0.7, "end": 0.8}]},
            {"text": "x", "words": [{"word": "x", "start": 0.2, "end": 0.1}]},
            {"text": "x", "words": [{"word": "x", "start": True, "end": 0.1}]},
            {"text": "x", "words": [{"word": "x", "start": 0, "end": 2.0}]},
            {"text": "x", "words": [{"word": "", "start": 0, "end": 0.1}]},
        ):
            with self.subTest(bad=bad), self.assertRaises(HostedSpeechError):
                openai_units(bad, 600)


DIARIZE_INFO = {"diarize_info": {"model_uuid": "synthetic-uuid", "arch": "v2"}}
DIARIZED_EMPTY = {
    "metadata": DIARIZE_INFO,
    "results": {"channels": [{"alternatives": [{"words": []}]}]},
}


def diarized(words):
    return {
        "metadata": DIARIZE_INFO,
        "results": {"channels": [{"alternatives": [{"words": words}]}]},
    }


class HostedDiarizerTests(unittest.TestCase):
    def diarizer(self, **overrides):
        values = dict(allow_hosted=True, window_ms=2000, load_key=load_key)
        return DeepgramDiarizer(**(values | overrides))

    def test_hosted_use_requires_explicit_consent_and_valid_settings(self):
        with self.assertRaisesRegex(HostedSpeechError, "consent"):
            DeepgramDiarizer()
        for endpoint in (
            "http://api.deepgram.com/v1/listen",
            DEEPGRAM_ENDPOINT + "?diarize=true",
            DEEPGRAM_ENDPOINT + "?",
            DEEPGRAM_ENDPOINT + "#",
        ):
            with self.subTest(endpoint=endpoint), self.assertRaisesRegex(LiveAudioError, "https"):
                self.diarizer(endpoint=endpoint)
        with self.assertRaises(LiveAudioError):
            self.diarizer(diarize_model="v9")
        with self.assertRaises(LiveAudioError):
            self.diarizer(window_ms=100)

    def test_trailing_window_is_sent_as_wav_with_documented_query_and_token_header(self):
        diarizer = self.diarizer()
        payload = json.dumps(
            {
                "metadata": {"diarize_info": {"model_uuid": "synthetic-uuid", "arch": "v2"}},
                "results": {
                    "channels": [
                        {
                            "alternatives": [
                                {
                                    "words": [
                                        {"word": "a", "start": 0.0, "end": 0.4, "speaker": 0},
                                        {"word": "b", "start": 0.5, "end": 0.9, "speaker": 0},
                                        {"word": "c", "start": 1.0, "end": 1.4, "speaker": 1},
                                        {"word": "d", "start": 1.5, "end": 1.9},
                                        {"word": "e", "start": 1.9, "end": 2.0, "speaker": 1},
                                    ]
                                }
                            ]
                        }
                    ]
                },
            }
        ).encode()
        seen = []

        def open_request(request, timeout):
            seen.append(request)
            return FakeResponse(payload)

        with patch("urllib.request.build_opener", return_value=Mock(open=open_request)):
            diarizer = self.diarizer()
            for index in range(5):
                diarizer.push(bytes([index]) * (1000 * BYTES_PER_MS))
            self.assertEqual(diarizer.pushed_bytes, 5000 * BYTES_PER_MS)
            timeline = diarizer.segments()
            finished = diarizer.finish()
        self.assertEqual(len(seen), 2)
        request = seen[0]
        self.assertEqual(
            request.full_url,
            DEEPGRAM_ENDPOINT + "?model=nova-3&diarize_model=latest&mip_opt_out=true",
        )
        # Consented audio is excluded from the provider's model improvement program.
        self.assertIn("mip_opt_out=true", request.full_url)
        self.assertEqual(request.get_header("Content-type"), "audio/wav")
        self.assertFalse(request.has_header("Authorization"))
        with wave.open(io.BytesIO(request.data), "rb") as audio:
            frames = audio.readframes(audio.getnframes())
        # Only the last two seconds (the window) leave the process, never older audio.
        self.assertEqual(
            frames, bytes([3]) * (1000 * BYTES_PER_MS) + bytes([4]) * (1000 * BYTES_PER_MS)
        )
        self.assertEqual(
            timeline,
            [
                {"speaker": 1, "start_ms": 3000, "end_ms": 3900},
                {"speaker": 2, "start_ms": 4000, "end_ms": 4400},
                {"speaker": 2, "start_ms": 4900, "end_ms": 5000},
            ],
        )
        self.assertEqual(finished, timeline)
        diarizer.close()
        with self.assertRaisesRegex(HostedSpeechError, "closed"):
            diarizer.push(bytes(640))

    def test_authorization_uses_the_documented_token_scheme(self):
        seen = []

        def open_request(request, timeout):
            seen.append(request.get_header("Authorization"))
            return FakeResponse(json.dumps(DIARIZED_EMPTY).encode())

        with patch("urllib.request.build_opener", return_value=Mock(open=open_request)):
            diarizer = self.diarizer()
            diarizer.push(bytes(640))
            self.assertEqual(diarizer.segments(), [])
        self.assertEqual(seen, ["Token " + KEY])

    def test_finish_without_a_pending_utterance_sends_no_audio(self):
        opener, requests = patched_opener(json.dumps(DIARIZED_EMPTY).encode())
        turns = []
        with opener:
            silent = LiveProcessor(
                LiveConfig(
                    "silent-finish",
                    provenance="causal-replay",
                    transcriber=StubTranscriber(),
                    diarizer=lambda config: self.diarizer(cancelled=config.cancelled),
                ),
                turns.append,
            )
            for _ in range(100):
                silent.push_pcm16(bytes(640))
            silent.finish()
            self.assertEqual(requests, [])
            voiced = LiveProcessor(
                LiveConfig(
                    "voiced-finish",
                    provenance="causal-replay",
                    transcriber=StubTranscriber(),
                    diarizer=lambda config: self.diarizer(cancelled=config.cancelled),
                ),
                turns.append,
            )
            for _ in range(10):
                voiced.push_pcm16(VOICE)
            voiced.finish()
        self.assertEqual(len(requests), 1)
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0].speaker_provenance, "unknown")

    def test_silence_without_pushed_audio_and_failures_send_or_reveal_nothing(self):
        opener, requests = patched_opener(b"{}")
        with opener:
            self.assertEqual(self.diarizer().segments(), [])
            self.assertEqual(self.diarizer().finish(), [])
        self.assertEqual(requests, [])
        for error, expected in ((http_error(403), "HTTP 403"), (OSError(PRIVATE), "connection")):
            opener, _ = patched_opener(error=error)
            with self.subTest(expected=expected), opener:
                diarizer = self.diarizer()
                diarizer.push(bytes(640))
                with self.assertRaisesRegex(HostedSpeechError, expected) as caught:
                    diarizer.segments()
                self.assertNotIn(PRIVATE, str(caught.exception))
                self.assertNotIn(KEY, str(caught.exception))
        for bad in (
            {},
            {"results": {"channels": []}},
            {"results": {"channels": [{"alternatives": [{"words": PRIVATE}]}]}},
            diarized([{"speaker": 0}]),
            diarized([{"start": 0, "end": 0.1, "speaker": 26}]),
            diarized([{"start": 0, "end": 9.0, "speaker": 0}]),
        ):
            with self.subTest(bad=bad), self.assertRaises(HostedSpeechError) as caught:
                deepgram_timeline(bad, 1000, 0)
            self.assertNotIn(PRIVATE, str(caught.exception))

    def test_a_response_without_diarizer_evidence_is_refused_not_treated_as_unknown(self):
        labelled = [{"word": "a", "start": 0.0, "end": 0.2, "speaker": 0}]
        unlabelled = [{"word": "a", "start": 0.0, "end": 0.2}]
        for document in (
            {"results": {"channels": [{"alternatives": [{"words": labelled}]}]}},
            {"metadata": {}, "results": {"channels": [{"alternatives": [{"words": labelled}]}]}},
            {
                "metadata": {"diarize_info": {"arch": "v2"}},
                "results": {"channels": [{"alternatives": [{"words": labelled}]}]},
            },
            {
                "metadata": DIARIZE_INFO,
                "results": {"channels": [{"alternatives": [{"words": unlabelled}]}]},
            },
            {"results": {"channels": [{"alternatives": [{"words": unlabelled}]}]}},
        ):
            with self.subTest(document=document):
                if document.get("metadata") == DIARIZE_INFO:
                    # Diarizer ran, but it labelled nothing: honestly unknown, not an error.
                    self.assertEqual(deepgram_timeline(document, 1000, 0), [])
                    continue
                with self.assertRaisesRegex(HostedSpeechError, "diarization unavailable"):
                    deepgram_timeline(document, 1000, 0)
        self.assertEqual(
            deepgram_timeline(diarized(labelled), 1000, 0),
            [{"speaker": 1, "start_ms": 0, "end_ms": 200}],
        )

    def test_same_speaker_words_across_a_wide_gap_stay_separate(self):
        words = [
            {"word": "a", "start": 0.0, "end": 0.2, "speaker": 0},
            {"word": "b", "start": 2.0, "end": 2.2, "speaker": 0},
        ]
        timeline = deepgram_timeline(diarized(words), 3000, 0)
        self.assertEqual(
            timeline,
            [
                {"speaker": 1, "start_ms": 0, "end_ms": 200},
                {"speaker": 1, "start_ms": 2000, "end_ms": 2200},
            ],
        )
        # A unit transcribed inside the gap is unknown rather than "Speaker A".
        self.assertEqual(_attribute(1000, 1100, timeline), (None, False))
        close = [
            {"word": "a", "start": 0.0, "end": 0.2, "speaker": 0},
            {"word": "b", "start": 0.5, "end": 0.7, "speaker": 0},
        ]
        self.assertEqual(
            deepgram_timeline(diarized(close), 3000, 0),
            [{"speaker": 1, "start_ms": 0, "end_ms": 700}],
        )


class ConfigurationSelectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.asset = Path(self.directory.name) / "local-asset"
        self.asset.touch()
        self.path = Path(self.directory.name) / "config.json"
        self.local = {
            name: str(self.asset)
            for name in (
                "whisper_executable",
                "whisper_model",
                "diarization_library",
                "diarization_model",
                "microphone_helper",
            )
        }
        self.hosted_transcriber = {
            "kind": "hosted-openai-compatible",
            "endpoint": ENDPOINT,
            "model": "whisper-1",
        }
        self.hosted_diarizer = {"kind": "hosted-deepgram", "model": "nova-3"}

    def load(self, raw):
        self.path.write_text(json.dumps(raw))
        return PrototypeConfig.load(self.path)

    def test_existing_configuration_selects_the_local_defaults(self):
        config = self.load(self.local)
        self.assertEqual(config.transcriber, {"kind": "whisper.cpp"})
        self.assertEqual(config.diarizer, {"kind": "nemotron.cpp"})
        self.assertFalse(config.hosted_speech)
        self.assertIsNone(transcriber_factory(config.transcriber))
        self.assertIsNone(diarizer_factory(config.diarizer))
        explicit = self.local | {"transcriber": {"kind": "whisper.cpp"}}
        self.assertEqual(self.load(explicit).transcriber, {"kind": "whisper.cpp"})

    def test_hosted_kinds_are_selected_by_name_and_need_no_local_assets(self):
        raw = {
            "microphone_helper": str(self.asset),
            "transcriber": self.hosted_transcriber,
            "diarizer": self.hosted_diarizer,
        }
        config = self.load(raw)
        self.assertTrue(config.hosted_speech)
        self.assertIsNone(config.whisper_executable)
        self.assertEqual(config.diarizer["diarize_model"], "latest")
        self.assertEqual(config.diarizer["endpoint"], DEEPGRAM_ENDPOINT)
        self.assertEqual(config.transcriber["timeout_seconds"], 30)
        live = LiveConfig(
            "selection-test",
            provenance="causal-replay",
            transcriber=transcriber_factory(config.transcriber, allow_hosted=True),
            diarizer=diarizer_factory(config.diarizer, allow_hosted=True),
        )
        self.assertIsInstance(live.transcriber(live), OpenAICompatibleTranscriber)
        self.assertIsInstance(live.diarizer(live), DeepgramDiarizer)
        # Without consent the factories build backends that refuse to send anything.
        with self.assertRaisesRegex(HostedSpeechError, "consent"):
            transcriber_factory(config.transcriber)(live)
        with self.assertRaisesRegex(HostedSpeechError, "consent"):
            diarizer_factory(config.diarizer)(live)
        # Mixing one hosted backend with one local one keeps that side's assets required.
        mixed = self.local | {"diarizer": self.hosted_diarizer}
        self.assertTrue(self.load(mixed).hosted_speech)
        del mixed["whisper_model"]
        with self.assertRaises(PrototypeError):
            self.load(mixed)

    def test_invalid_selections_are_rejected(self):
        for invalid in (
            {"kind": "unknown"},
            {"kind": "hosted-deepgram"},
            "whisper.cpp",
            {"kind": "whisper.cpp", "endpoint": ENDPOINT},
            {"kind": "hosted-openai-compatible", "model": "whisper-1"},
            {"kind": "hosted-openai-compatible", "endpoint": "http://x.test", "model": "m"},
            {**self.hosted_transcriber, "extra": 1},
            {**self.hosted_transcriber, "timeout_seconds": 1000},
            {**self.hosted_transcriber, "language": "english language"},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(PrototypeError) as error:
                self.load(self.local | {"transcriber": invalid})
            self.assertNotIn(ENDPOINT, str(error.exception))
        for invalid in (
            {"kind": "unknown"},
            {"kind": "hosted-openai-compatible", "endpoint": ENDPOINT, "model": "m"},
            {"kind": "nemotron.cpp", "model": "nova-3"},
            {"kind": "hosted-deepgram", "diarize_model": "v3"},
            {"kind": "hosted-deepgram", "endpoint": "http://api.deepgram.com/v1/listen"},
            {"kind": "hosted-deepgram", "extra": True},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(PrototypeError):
                self.load(self.local | {"diarizer": invalid})

    def test_snapshot_reports_selected_backends_without_endpoints_or_credentials(self):
        local = PrototypeController(self.load(self.local), capture_factory=Mock())
        self.addCleanup(local.close)
        snapshot = local.snapshot()
        self.assertEqual(
            snapshot["transcriber"],
            {"kind": "whisper.cpp", "hosted": False, "service": "Whisper (configured model)"},
        )
        self.assertEqual(
            snapshot["diarizer"],
            {"kind": "nemotron.cpp", "hosted": False, "service": "Nemotron 3 (configured GGUF)"},
        )
        self.assertEqual(snapshot["models"]["asr"], "Whisper (configured model)")
        hosted = PrototypeController(
            self.load(
                {
                    "microphone_helper": str(self.asset),
                    "transcriber": self.hosted_transcriber | {"language": "en"},
                    "diarizer": self.hosted_diarizer,
                }
            ),
            capture_factory=Mock(),
            allow_hosted_speech=True,
        )
        self.addCleanup(hosted.close)
        snapshot = hosted.snapshot()
        self.assertEqual(
            snapshot["transcriber"],
            {
                "kind": "hosted-openai-compatible",
                "hosted": True,
                "service": "OpenAI-compatible hosted",
            },
        )
        self.assertEqual(
            snapshot["diarizer"],
            {"kind": "hosted-deepgram", "hosted": True, "service": "Deepgram hosted"},
        )
        self.assertEqual(snapshot["models"]["diarization"], "Deepgram hosted")
        with patch.dict(os.environ, {"RIGHTYO_TRANSCRIBER_API_KEY": KEY}):
            encoded = json.dumps(hosted.snapshot())
        for secret in (ENDPOINT, DEEPGRAM_ENDPOINT, "whisper-1", "nova-3", KEY, "api.deepgram"):
            self.assertNotIn(secret, encoded)

    def test_started_event_advertises_selected_backends_with_display_safe_ids(self):
        # Authored replay publishers advertise nothing, so shared fixtures are unchanged.
        plain = SpeechEvents()
        plain.start("fixture-session")
        self.assertNotIn("speech", plain.drain()[0])
        local = speech_summary({"kind": "whisper.cpp"}, {"kind": "nemotron.cpp"})
        self.assertEqual(
            local,
            {
                "transcriber": {"kind": "whisper.cpp", "id": "whisper.cpp-live-window"},
                "diarizer": {"kind": "nemotron.cpp", "id": "nemotron.cpp v3-streaming"},
            },
        )
        config = self.load(
            {
                "microphone_helper": str(self.asset),
                "transcriber": self.hosted_transcriber,
                "diarizer": self.hosted_diarizer | {"diarize_model": "v2"},
            }
        )
        hosted = speech_summary(config.transcriber, config.diarizer)
        self.assertEqual(hosted["transcriber"]["kind"], "hosted-openai-compatible")
        self.assertEqual(hosted["diarizer"]["kind"], "hosted-deepgram")
        self.assertEqual(
            hosted["transcriber"]["id"],
            OpenAICompatibleTranscriber(
                endpoint=ENDPOINT, model="whisper-1", allow_hosted=True
            ).recognizer_id,
        )
        self.assertEqual(
            hosted["diarizer"]["id"],
            DeepgramDiarizer(allow_hosted=True, diarize_model="v2").diarizer_id,
        )
        for secret in (ENDPOINT, "example.test", "api.deepgram", str(self.asset), KEY):
            self.assertNotIn(secret, json.dumps(hosted))
        events = SpeechEvents()
        controller = PrototypeController(
            config,
            processor_factory=Mock(side_effect=LiveAudioError("stop before audio")),
            capture_factory=Mock(),
            event_publisher=events,
            allow_hosted_speech=True,
        )
        self.addCleanup(controller.close)
        controller.start({"mode": "microphone"})
        started = controller.drain_events()[0]
        self.assertEqual(started["phase"], "started")
        self.assertEqual(started["speech"], hosted)
        self.assertNotIn("speech", started["capabilities"])
        with self.assertRaises(ContractError):
            SpeechEvents().start("bad", speech={"transcriber": {"kind": "x"}})

    def test_configured_roles_are_refused_with_an_utterance_local_diarizer(self):
        roles = {"owner": ["Speaker A"], "trusted": [], "owner_only": False}
        processors = []

        def processor(live, on_turn):
            processors.append(live)
            raise LiveAudioError("stop before any audio")

        for speakers in (
            roles,
            {"owner": ["Speaker A"], "trusted": [], "owner_only": True},
            {"owner": [], "trusted": ["Speaker B"], "owner_only": False},
        ):
            with self.subTest(speakers=speakers):
                config = self.load(
                    self.local | {"diarizer": self.hosted_diarizer, "speakers": speakers}
                )
                controller = PrototypeController(
                    config,
                    processor_factory=processor,
                    capture_factory=Mock(),
                    allow_hosted_speech=True,
                )
                self.addCleanup(controller.close)
                with self.assertRaisesRegex(PrototypeError, "session-stable diarizer") as error:
                    controller.start({"mode": "microphone"})
                self.assertNotIn(ENDPOINT, str(error.exception))
        self.assertEqual(processors, [])
        self.path.write_text(
            json.dumps(self.local | {"diarizer": self.hosted_diarizer, "speakers": roles})
        )
        args = Namespace(
            config=self.path, mode="microphone", session_id=None, use_jev=False, allow_hosted=True
        )
        with self.assertRaisesRegex(PrototypeError, "session-stable diarizer"):
            listen(args, controller_factory=PrototypeController)
        # The native, session-stable diarizer still starts with the same roles.
        native = PrototypeController(
            self.load(self.local | {"speakers": roles}),
            processor_factory=processor,
            capture_factory=Mock(),
        )
        self.addCleanup(native.close)
        native.start({"mode": "microphone"})
        deadline = 100
        while not processors and deadline:
            deadline -= 1
            native._audio_thread.join(0.05)
        self.assertEqual(len(processors), 1)

    def test_controller_and_listen_require_explicit_hosted_consent(self):
        config = self.load(self.local | {"transcriber": self.hosted_transcriber})
        processors = []

        def processor(live, on_turn):
            processors.append(live)
            raise LiveAudioError("stop before any audio")

        controller = PrototypeController(
            config, processor_factory=processor, capture_factory=Mock()
        )
        self.addCleanup(controller.close)
        with self.assertRaisesRegex(PrototypeError, "consent"):
            controller.start({"mode": "microphone"})
        self.assertEqual(processors, [])
        consented = PrototypeController(
            config, processor_factory=processor, capture_factory=Mock(), allow_hosted_speech=True
        )
        self.addCleanup(consented.close)
        consented.start({"mode": "microphone"})
        deadline = 100
        while not processors and deadline:
            deadline -= 1
            consented._audio_thread.join(0.05)
        self.assertEqual(len(processors), 1)
        self.assertIsInstance(processors[0].transcriber(processors[0]), OpenAICompatibleTranscriber)
        self.assertIsNone(processors[0].diarizer)
        args = Namespace(
            config=self.path, mode="demo", session_id=None, use_jev=False, allow_hosted=False
        )
        with self.assertRaisesRegex(PrototypeError, "allow-hosted"):
            listen(args, controller_factory=Mock())
        args.allow_hosted = True
        factory = Mock(side_effect=PrototypeError("stop at construction"))
        with self.assertRaisesRegex(PrototypeError, "construction"):
            listen(args, controller_factory=factory)
        self.assertTrue(factory.call_args.kwargs["allow_hosted_speech"])


class SpeechCredentialTests(unittest.TestCase):
    def test_each_backend_has_its_own_environment_variable_and_keychain_item(self):
        with patch.dict(os.environ, {"RIGHTYO_TRANSCRIBER_API_KEY": KEY}, clear=True):
            with patch("rightyo.credentials.subprocess.Popen") as runner:
                self.assertEqual(load_transcriber_api_key(), KEY)
                runner.assert_not_called()
            with self.assertRaises(CredentialError) as error:
                with patch("rightyo.credentials.sys.platform", "linux"):
                    load_diarizer_api_key()
            self.assertIn("RIGHTYO_DIARIZER_API_KEY", str(error.exception))

        class Completed(FakeProcess):
            def __init__(self, args, status, output, **options):
                super().__init__(args, **options)
                self.stdout = io.BytesIO(output)
                self.status = status

            def poll(self):
                self.returncode = self.status
                return self.returncode

        spawned = []

        def completed(status, output):
            def popen(args, **options):
                spawned.append(Completed(args, status, output, **options))
                return spawned[-1]

            return popen

        with (
            patch.dict(os.environ, {"TYPESAFE_API_KEY": "unrelated-jev-key"}, clear=True),
            patch("rightyo.credentials.sys.platform", "darwin"),
            patch(
                "rightyo.credentials.subprocess.Popen",
                side_effect=completed(0, (KEY + "\n").encode()),
            ),
        ):
            self.assertEqual(load_diarizer_api_key(), KEY)
            self.assertIn("rightyo.diarizer", spawned[-1].args)
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("rightyo.credentials.sys.platform", "darwin"),
            patch("rightyo.credentials.subprocess.Popen", side_effect=completed(1, KEY.encode())),
        ):
            with self.assertRaises(CredentialError) as error:
                load_transcriber_api_key()
            self.assertIn("rightyo.transcriber", str(error.exception))
            self.assertNotIn(KEY, str(error.exception))


if __name__ == "__main__":
    unittest.main()
