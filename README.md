# RightyO

A compact audio attention system: recognize when speech is addressed to the system, then transcribe that speech and emit the text.

**Status:** a transcript-first experiment MVP, with local whisper.cpp file transcription, explicit local Nemotron 3 diarization or imported anonymous speaker timelines, mock decisions and opt-in Jev decisions. An explicit local Mac microphone lab is available; no trained attention model is included. Project name: **RightyO**; repository/package spelling: **rightyo**.

The first experiment transcribes locally and asks Jev whether the completed turn addresses the system and who its recipient is. Speaker A/B labels are sufficient initially; diarization alone cannot establish the addressee. This quickly tests whether existing models solve the product problem. A frozen audio encoder, compact attention head and gated decoder remain conditional research directions after quality and compute measurements justify them.

## Try the microphone lab

See [the local prototype guide](docs/prototype.md) for Start/Stop controls, persistent
Nemotron speaker labels, local Whisper text, five minutes of context, and opt-in Jev.
The lab starts idle and uses separately provisioned native runtimes and models.

```sh
.venv/bin/rightyo prototype --config local/prototype.json --port 8766
```

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

[Hailing Station](https://github.com/mickdarling/hailing-station) is a possible downstream consumer of the attended transcript. Its host remains responsible for programmatic routing, target selection, and output handling. RightyO can also serve other consumers. Multi-host negotiation and automatic tool/session selection are outside this project's first scope.

Jev's documented input is text only. It can be evaluated as a transcript-based comparison baseline, but cannot directly implement this audio-before-transcription attention detector. See [research notes](docs/research.md).

## License

RightyO's original code and documentation are licensed under **GNU Affero GPL version 3 or later** (`AGPL-3.0-or-later`); see [LICENSE](LICENSE). Copyright (C) 2026 Mick Darling. There is no warranty.

We intend to release project-owned trained weights openly under AGPL-3.0-or-later too, accompanied by the training code, configurations, provenance, and reproducibility materials we have the right to distribute. No weights are released yet. Upstream code/weights retain their original notices, and datasets retain their own licenses and privacy restrictions. A repository license does not grant rights to somebody else's data or guarantee that every trained checkpoint can be redistributed. See the [release licensing policy](docs/licensing.md).
