# Related research and naming

Checked September 29, 2026. These are research leads, not imported implementations, verified performance reproductions, or license clearance.

## Directly relevant prior art

- [Apple: Device-Directed Speech Detection, 2022](https://machinelearning.apple.com/research/device-directed-speech) studies detecting directed speech without a specific wake word in touch-invoked sessions. An acoustics-only student is trained with distillation from an ASR-based teacher. This supports investigating an audio-only inference head; its invocation setting differs from unrestricted ambient observation.
- [Streaming device-directed speech detection, 2021](https://arxiv.org/abs/2110.04656) examines efficient causal decision layers for acoustic false-trigger mitigation after voice/touch invocation. Evaluate the methodology and limits rather than assuming that post-invocation results transfer to continuous listening.
- [Attention Labs research](https://attentionlabs.ai/research) describes a wake-wordless pre-ASR addressee detection layer and points to [arXiv:2604.08412](https://arxiv.org/abs/2604.08412). This is especially close conceptually. Paper/site titles differ; inspect the exact paper version, available software, evaluation setup, and licensing before comparison or reuse. Its published claims have not been reproduced here.

There is meaningful prior art in precisely this problem. The project should evaluate reuse and a small composable implementation, not claim invention or uniqueness without further evidence. Public papers do not imply usable open-source code, weights, or patent clearance.

## Comparison baselines

[TypeSafe's model documentation](https://docs.typesafe.ai/models) specifies text-only input for Jev. A Jev decision over already-transcribed text is a comparison baseline or an existing Hailing Station option, not the proposed pre-ASR audio detector. No hosted requests have been made for this project.

General voice activity detectors, phrase wake-word detectors, and bounded speech-to-intent engines solve adjacent questions. Evaluate their components where useful, but require addressedness metrics and causal audio-only acquisition for this project's core claim.

## Whisper starting point

[OpenAI's Whisper repository](https://github.com/openai/whisper) documents MIT licensing for code and model weights and sliding 30-second windows in the standard transcribe path. This permits investigation of a Whisper-derived AGPL project with upstream notices preserved; any chosen checkpoint, fork, dataset, and dependency must be audited individually. None has been imported or downloaded in this bootstrap.

Whisper is the initial encoder/ASR family, not an already trained attention detector. Full-file transcription is not the desired attention-first runtime. The first experiment compares a trained head on frozen small-encoder features, limited causal audio windows, and a separately gated decoder. See the training process for staged fine-tuning/distillation options.

## Selected name

The owner selected **RightyO**, repository **rightyo**, on September 29, 2026. A preliminary web/GitHub/indexed app-store search found no obvious exact-name app or voice/AI-tool collision for RightyO/Righty-O/Righty O. The phrase has unrelated artistic uses; this is not trademark or store-name clearance. The repository is public with original code/documentation licensed AGPL-3.0-or-later.
