"""Write a deterministic started-only host fixture from a prototype config.

Uses the real event producer without starting a controller, audio, model workers or
inference providers. Host options are fixed to stdin, ordinary controls, initially
private (when local speech permits it), and owner-voice status at 20 seconds with roles.
No environment opt-ins or enrollment data are read. This is a wire fixture, not evidence
that any configured backend or live capability works.

    PYTHONPATH=src python3 scripts/emit_started_event.py --config config.json --out started.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rightyo.live_speaker_id import EnrolledRoles
from rightyo.prototype import STDIN_AUDIO_INPUT, PrototypeConfig, PrototypeError
from rightyo.providers import ConfiguredPriorityProvider, request_former_for
from rightyo.speech_backends import speech_summary
from rightyo.tool_events import OwnerVoice, SpeechEvents, encode_json


def started_event(config: PrototypeConfig) -> dict:
    """Construct only the real producer's first event; no runtime is started."""
    priority = None
    if config.speakers is not None:
        priority = (
            EnrolledRoles(
                config.speakers,
                config.speaker_id.enrolled_follow_up_min_probability,
                emit_names=config.speaker_id.emit_names,
            )
            if config.voiceprint_roles
            else ConfiguredPriorityProvider(config.speakers)
        )
    events = SpeechEvents(request_control=True)
    if not config.hosted_speech:
        events.duck(True, now_ms=0)
    events.start(
        "host-contract",
        attention_enabled=config.hosted_decisions,
        addressing=config.addressing,
        priority=priority,
        former=request_former_for(config.request_former),
        speech=speech_summary(config.transcriber, config.diarizer),
        audio_input=STDIN_AUDIO_INPUT,
        dismissal=config.dismissal,
        conversation=config.conversation,
        self_voice=config.self_voice,
        acknowledgement=config.acknowledgement,
        meeting_ask_anywhere=config.meeting_ask_anywhere,
        other_agents=config.other_agents,
        owner_voice=OwnerVoice() if config.voiceprint_roles else None,
        duck=not config.hosted_speech,
        meeting=True,
    )
    return events.drain()[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    try:
        config = PrototypeConfig.load(args.config)
        args.out.write_text(encode_json(started_event(config)) + "\n", encoding="utf-8")
    except (PrototypeError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
