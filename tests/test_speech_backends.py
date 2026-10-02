"""Offline backend-selection tests: mocked HTTP, no network, credentials, models or audio."""

import array
import io
import json
import os
import subprocess
import tempfile
import unittest
import urllib.error
import wave
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock, patch

from rightyo.credentials import CredentialError, load_diarizer_api_key, load_transcriber_api_key
from rightyo.live_audio import (
    BYTES_PER_MS,
    LiveAudioError,
    LiveConfig,
    LiveProcessor,
    NemotronCppDiarizer,
    WhisperCppTranscriber,
)
from rightyo.prototype import PrototypeConfig, PrototypeController, PrototypeError
from rightyo.providers import Diarizer, Transcriber
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
    transcriber_factory,
)
from rightyo.tool import listen

KEY = "synthetic-test-key-do-not-use"
PRIVATE = "private utterance text"
ENDPOINT = "https://transcribe.example.test/v1/audio/transcriptions"
VOICE = array.array("h", [5000] * 320).tobytes()


def load_key():
    return KEY


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def read(self, limit):
        return self.payload[:limit]

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


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
    def __init__(self):
        self.frames = []
        self.closed = False

    def push(self, pcm):
        self.frames.append(pcm)

    def segments(self):
        return [{"start_ms": 0, "end_ms": 900000, "speaker": 2}]

    def finish(self):
        return self.segments()

    def close(self):
        self.closed = True


class StubTranscriber:
    recognizer_id = "stub-recognizer"

    def __init__(self):
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
        self.assertTrue(diarizer.closed)

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


class HostedTranscriberTests(unittest.TestCase):
    def transcriber(self, **overrides):
        values = dict(endpoint=ENDPOINT, model="whisper-1", allow_hosted=True, load_key=load_key)
        return OpenAICompatibleTranscriber(**(values | overrides))

    def test_hosted_use_requires_explicit_consent_and_https(self):
        with self.assertRaisesRegex(HostedSpeechError, "consent"):
            OpenAICompatibleTranscriber(endpoint=ENDPOINT, model="whisper-1")
        with self.assertRaisesRegex(HostedSpeechError, "consent"):
            OpenAICompatibleTranscriber(endpoint=ENDPOINT, model="whisper-1", allow_hosted=1)
        for endpoint in ("http://transcribe.example.test/v1", "https://u:p@x.test/v1", 5):
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
        self.assertEqual(timeout, 7)
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
        def missing():
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
        # Misaligned text keeps bare words rather than guessing punctuation.
        self.assertEqual(
            [u["text"] for u in openai_units({"text": "I am here.", "words": words}, 600)],
            [" I'm", " here"],
        )
        segments = {"text": "A. B.", "segments": [{"text": "A.", "start": 0, "end": 0.4}]}
        self.assertEqual(
            openai_units(segments, 300), [{"text": " A.", "start_ms": 0, "end_ms": 300}]
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


class HostedDiarizerTests(unittest.TestCase):
    def diarizer(self, **overrides):
        values = dict(allow_hosted=True, window_ms=2000, load_key=load_key)
        return DeepgramDiarizer(**(values | overrides))

    def test_hosted_use_requires_explicit_consent_and_valid_settings(self):
        with self.assertRaisesRegex(HostedSpeechError, "consent"):
            DeepgramDiarizer()
        with self.assertRaisesRegex(LiveAudioError, "https"):
            self.diarizer(endpoint="http://api.deepgram.com/v1/listen")
        with self.assertRaises(LiveAudioError):
            self.diarizer(diarize_model="v9")
        with self.assertRaises(LiveAudioError):
            self.diarizer(window_ms=100)

    def test_trailing_window_is_sent_as_wav_with_documented_query_and_token_header(self):
        diarizer = self.diarizer()
        payload = json.dumps(
            {
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
                }
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
        self.assertEqual(request.full_url, DEEPGRAM_ENDPOINT + "?model=nova-3&diarize_model=latest")
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
            return FakeResponse(b'{"results":{"channels":[{"alternatives":[{"words":[]}]}]}}')

        with patch("urllib.request.build_opener", return_value=Mock(open=open_request)):
            diarizer = self.diarizer()
            diarizer.push(bytes(640))
            self.assertEqual(diarizer.segments(), [])
        self.assertEqual(seen, ["Token " + KEY])

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
            {"results": {"channels": [{"alternatives": [{"words": [{"speaker": 0}]}]}]}},
            {
                "results": {
                    "channels": [
                        {"alternatives": [{"words": [{"start": 0, "end": 0.1, "speaker": 26}]}]}
                    ]
                }
            },
            {
                "results": {
                    "channels": [
                        {"alternatives": [{"words": [{"start": 0, "end": 9.0, "speaker": 0}]}]}
                    ]
                }
            },
        ):
            with self.subTest(bad=bad), self.assertRaises(HostedSpeechError) as caught:
                deepgram_timeline(bad, 1000, 0)
            self.assertNotIn(PRIVATE, str(caught.exception))


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
            with patch("rightyo.credentials.subprocess.run") as runner:
                self.assertEqual(load_transcriber_api_key(), KEY)
                runner.assert_not_called()
            with self.assertRaises(CredentialError) as error:
                with patch("rightyo.credentials.sys.platform", "linux"):
                    load_diarizer_api_key()
            self.assertIn("RIGHTYO_DIARIZER_API_KEY", str(error.exception))
        response = subprocess.CompletedProcess([], 0, (KEY + "\n").encode(), b"")
        with (
            patch.dict(os.environ, {"TYPESAFE_API_KEY": "unrelated-jev-key"}, clear=True),
            patch("rightyo.credentials.sys.platform", "darwin"),
            patch("rightyo.credentials.subprocess.run", return_value=response) as runner,
        ):
            self.assertEqual(load_diarizer_api_key(), KEY)
            self.assertIn("rightyo.diarizer", runner.call_args.args[0])
        failed = subprocess.CompletedProcess([], 1, KEY.encode(), KEY.encode())
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("rightyo.credentials.sys.platform", "darwin"),
            patch("rightyo.credentials.subprocess.run", return_value=failed),
        ):
            with self.assertRaises(CredentialError) as error:
                load_transcriber_api_key()
            self.assertIn("rightyo.transcriber", str(error.exception))
            self.assertNotIn(KEY, str(error.exception))


if __name__ == "__main__":
    unittest.main()
