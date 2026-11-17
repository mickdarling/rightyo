# RightyO

RightyO decides when speech is meant for an assistant. It listens to a live room,
transcribes each finished turn, works out who spoke, and decides whether the turn
was addressed to the system. Only then does it hand on a request. Everything else
stays context.

**Status:** RightyO is in daily use as the hands-free speech front end of
[Hailing Station](https://github.com/mickdarling/hailing-station), on a Mac.

What runs today:

- **Live listening:** microphone audio streamed from a phone or Mac to a Mac host,
  with local [whisper.cpp](https://github.com/ggml-org/whisper.cpp) transcription and
  a local end-of-turn model.
- **Who spoke:** local speaker diarization, plus optional on-device voice enrollment
  that recognizes the owner and trusted speakers. Voiceprints never leave the machine.
- **Was it meant for the assistant:** a decision model (Jev, opt-in and hosted, given
  text only) reads each finished turn in context. It labels the turn `attend`,
  `ignore` or `uncertain` and names the likely recipient.
- **Conversation rules on top:**
  - follow-ups without repeating the name;
  - natural "stop" and "never mind" dismissals;
  - meeting, media and privacy modes, which hold back or exclude speech.
- **For hosts:** a versioned JSONL event stream, with a contract test against Hailing
  Station's decoder so the two stay in step.

What is still experimental:

- **Audio attention:** there is no trained attention model yet. Addressee decisions
  come from a general text model over transcripts. The audio-first design (a frozen
  encoder, a small attention head and a gated decoder; see
  [architecture](docs/architecture.md)) is still a research direction. It waits on
  measured quality and cost.
- **Quality:** decisions and transcription quality are not yet benchmarked on a
  public evaluation set. Live behavior is tuned from the author's own use.
- **Platform:** a Mac host with separately provisioned local models. There is no
  packaged release and no mobile on-device deployment.

Project name: **RightyO**; repository/package spelling: **rightyo**.

## Try the microphone lab

See [the local prototype guide](docs/prototype.md) for Start/Stop controls, persistent
Nemotron speaker labels, local Whisper text, five minutes of context, and opt-in Jev.
The lab starts idle and uses separately provisioned native runtimes and models.

```sh
.venv/bin/rightyo prototype --config local/prototype.json --port 8766
```

## Use RightyO as a local tool

The website is a test/demo rig. A host application can instead consume the versioned
JSONL event stream directly, without a browser. Install from this source checkout with
Python 3.11+; no published package or bundled native models are assumed:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install .
.venv/bin/rightyo tool-replay --input examples/synthetic-turns.json --provider mock
```

That first command path needs no microphone, model weights, API key, or hosted calls.
It emits authored fixture decisions, not model-quality evidence. After separately
provisioning the assets in [the prototype guide](docs/prototype.md), run the explicit
foreground microphone tool:

```sh
.venv/bin/rightyo listen --config local/prototype.json --mode microphone \
  --session-id local-session-001
```

Without hosted opt-in it emits transcript/session events only. Add both `--use-jev`
and `--allow-hosted`, or the configuration file's
`"decision": {"provider": "jev", "allow_hosted": true}`, to send bounded finalized
transcript context to Jev and emit attention/request events. Stop with Ctrl-C; restart with a fresh session ID. The stream
contains transcript text: pipe it to a local consumer and avoid logging real conversations.
See [the tool guide](docs/tool-quickstart.md) for the event contract, Hailing Station handoff,
limits, and contribution checks. The configuration's optional `audio_retention` object sets
how much audio is kept in memory for speaker identification and attribution (60 s each by
default; see [the prototype guide](docs/prototype.md)); a change takes effect at restart. Replies, acknowledgement voices, routing, and tool
execution belong to the consuming application.

## Enroll speakers locally

`rightyo enroll` computes a voiceprint for each person you enroll and stores it only on
your machine (mode 700, never inside a repository checkout), with `list`, `verify` and
`delete`. You provision the model yourself; RightyO never downloads it. With
`"live": true`, live sessions run identification. By default that is shadow mode: scores
are logged on stderr, and turns, roles and events don't change. With `"roles": true` as
well (opt-in), a voice bound to an enrolled identifier carries the role the `speakers`
section gives it, the session advertises `speakers: "enrolled"`, and enrolled owners and
trusted speakers take precedence. See [speaker enrollment](docs/speaker-enrollment.md).

## Replay private coaching

An explicitly enabled, owner-private coaching component can accept final tool transcripts
under a host-qualified private output generation. It uses bounded context, a replaceable
provider and short-lived advice tickets. It grants no command or playback authority.
Try its authored mock fixture without a microphone, model, API key or audio output:

```sh
PYTHONPATH=src python3 -m rightyo coach-replay --input examples/coaching-replay.json
```

For continuous host composition, [the coaching listener](docs/coaching-listen.md)
documents opt-in raw PCM input, separate trusted controls and an executable authored
speech backend. It requires no model assets for synthetic qualification; actual
recognition/coaching quality and private hardware behavior remain unvalidated.
Mock advice is the default. A separately consented Jev provider can select bounded
locally authored catalog suggestions; its credential/HTTPS path is qualified with
fakes, and hosted usefulness/latency remain unvalidated. See the listener guide for
the separate consent and request-budget requirements.

See [the coaching guide](docs/conversation-coaching.md) for the replay and library
adapter, and [the wire contract](docs/coaching-wire-v1.md) for the host boundary. This
is a separate opt-in experiment; actual coaching inference and device audio wiring
remain unvalidated.

## Try the replay MVP

Python 3.11 or newer is sufficient for synthetic replay; the runtime has no Python dependencies. See [the MVP guide](docs/mvp.md) for secure credentials, local audio and speaker imports, and the limits of the experiment.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install .
.venv/bin/rightyo evaluate --input examples/synthetic-turns.json --provider mock
```

The mock is a fixture rule, not a trained model. Hosted mode is explicit and sends bounded transcript context to TypeSafe. Transcript text and source IDs are omitted from output by default. File ASR and imported speaker timelines are offline experiments; they do not establish live latency or diarization accuracy.

## Product boundary

The attention component answers one narrowly scoped question: should the system attend to this speech? Its result is typed (`attend`, `ignore`, or `uncertain`). It does not interpret commands, choose tools, summarize, or generate replies. The transcription component emits what was actually said, with stream and utterance provenance. The MVP consumes completed transcript turns; the following diagram describes the later audio-first research option.

```text
Consented audio stream -> bounded rolling buffer -> streaming attention detector
                                                        |
                                                      attend
                                                        v
                                 buffered onset + live audio -> ASR -> transcript events
```

The buffer preserves the opening words spoken before the detector can make its decision. Full speech-to-text decoding begins when attention is established. Feature extraction may share an encoder with ASR in a later compact implementation.

The first research rig targets a Mac. This project does not change Hailing Station's current iPhone/iPad on-device speech recognition or introduce a semantic model on those devices. Future deployments are a separate decision.

## Project records

GitHub issues manage features, research questions, acceptance criteria, and process work. They serve as feature/story tickets without role-based story boilerplate. Implementation proceeds through issue-linked pull requests with documented verification and independent review of the exact head. A failed or missing review is not approval.

The [execution roadmap (#1)](https://github.com/mickdarling/rightyo/issues/1) links the work records. Begin with the [fast MVP (#23)](https://github.com/mickdarling/rightyo/issues/23), [anonymous speakers (#24)](https://github.com/mickdarling/rightyo/issues/24), [comparison (#25)](https://github.com/mickdarling/rightyo/issues/25), and [CI/review enforcement (#13)](https://github.com/mickdarling/rightyo/issues/13). Dataset acquisition, training runs, and public model releases each have their own gates; the issue list is not blanket permission to start them.

Start with [architecture](docs/architecture.md), [training process](docs/training-process.md), [evaluation](docs/evaluation.md), [related research](docs/research.md), and the [roadmap](docs/roadmap.md). See [contribution workflow](CONTRIBUTING.md), [privacy/security](SECURITY.md), and [licensing](docs/licensing.md).

Audio recordings, transcripts from real conversations, credentials, and downloaded model weights do not belong in Git. No live capture, external inference, or dataset collection runs implicitly.

## Relationship to Hailing Station

[Hailing Station](https://github.com/mickdarling/hailing-station) runs RightyO as its live
ambient-listening pipeline. Its host daemon streams device audio to a RightyO child process
and reads the JSONL events. It then routes attended requests to the chosen agent session and
plays replies. Routing, target selection, replies and tool execution stay in the host.
RightyO only decides what was said, who said it and whether it was meant for the assistant.
Other hosts can consume the same event stream. Hailing Station's tap-to-talk path still uses
on-device speech recognition and does not go through RightyO.

Jev's input is text only. It serves as the working transcript-based decision layer and as the
baseline that any future audio attention model has to beat. See [research notes](docs/research.md).

## License

RightyO's original code and documentation are licensed under **GNU Affero GPL version 3 or later** (`AGPL-3.0-or-later`); see [LICENSE](LICENSE). Copyright (C) 2026 Mick Darling. There is no warranty.

Alternative commercial licensing may be made available by the copyright holder. The commercial option permits distributions that cannot or do not wish to comply with the AGPL; it does not change the availability of this public source tree.

External code contributions are not yet being accepted because the contributor agreement needed to preserve dual licensing has not been finalized. Bug reports, design discussion, and reproducible test reports are welcome.

We intend to release project-owned trained weights openly under AGPL-3.0-or-later too, accompanied by the training code, configurations, provenance, and reproducibility materials we have the right to distribute. No weights are released yet. Upstream code/weights retain their original notices, and datasets retain their own licenses and privacy restrictions. A repository license does not grant rights to somebody else's data or guarantee that every trained checkpoint can be redistributed. See the [release licensing policy](docs/licensing.md).
