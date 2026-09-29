# RightyO

A compact audio attention system: recognize when speech is addressed to the system, then transcribe that speech and emit the text.

**Status:** public model-research bootstrap. No trained attention model, live microphone capture, or working inference/training pipeline is included yet. Project name: **RightyO**; repository/package spelling: **rightyo**.

The initial direction is a Whisper-based audio attention model with a narrowly trained decision head and gated speech-to-text decoding. We will build the training infrastructure, consented/licensed dataset process, evaluation harness, and model-release process in this repository. Begin with a frozen small Whisper encoder and train a compact attention head; fine-tuning, distillation, and a shared compact model follow only if measurements justify them.

## Product boundary

The attention component answers one narrowly scoped question: should the system attend to this ongoing speech? Its result is typed (`attend`, `ignore`, or `uncertain`). It does not interpret commands, choose tools, summarize, or generate replies. The transcription component emits what was actually said, with stream and utterance provenance.

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

The [execution roadmap (#1)](https://github.com/mickdarling/rightyo/issues/1) links all initial work records. Begin with [architecture and rights (#2)](https://github.com/mickdarling/rightyo/issues/2), [experiment infrastructure (#3)](https://github.com/mickdarling/rightyo/issues/3), and [CI/review enforcement (#13)](https://github.com/mickdarling/rightyo/issues/13). Dataset acquisition, training runs, and public model releases each have their own gates; the issue list is not blanket permission to start them.

Start with [architecture](docs/architecture.md), [training process](docs/training-process.md), [evaluation](docs/evaluation.md), [related research](docs/research.md), and the [roadmap](docs/roadmap.md). See [contribution workflow](CONTRIBUTING.md), [privacy/security](SECURITY.md), and [licensing](docs/licensing.md).

Audio recordings, transcripts from real conversations, credentials, and downloaded model weights do not belong in Git. No live capture, external inference, or dataset collection runs implicitly.

## Relationship to Hailing Station

[Hailing Station](https://github.com/mickdarling/hailing-station) is a possible downstream consumer of the attended transcript. Its host remains responsible for programmatic routing, target selection, and output handling. RightyO can also serve other consumers. Multi-host negotiation and automatic tool/session selection are outside this project's first scope.

Jev's documented input is text only. It can be evaluated as a transcript-based comparison baseline, but cannot directly implement this audio-before-transcription attention detector. See [research notes](docs/research.md).

## License

RightyO's original code and documentation are licensed under **GNU Affero GPL version 3 or later** (`AGPL-3.0-or-later`); see [LICENSE](LICENSE). Copyright (C) 2026 Mick Darling. There is no warranty.

We intend to release project-owned trained weights openly under AGPL-3.0-or-later too, accompanied by the training code, configurations, provenance, and reproducibility materials we have the right to distribute. No weights are released yet. Upstream code/weights retain their original notices, and datasets retain their own licenses and privacy restrictions. A repository license does not grant rights to somebody else's data or guarantee that every trained checkpoint can be redistributed. See the [release licensing policy](docs/licensing.md).
