# Licensing code, models, and data

## Original work

Copyright (C) 2026 Mick Darling. RightyO's original code and documentation are offered under GNU Affero GPL version 3 or, at your option, any later version (`AGPL-3.0-or-later`). The full text is in LICENSE. No warranty is offered.

The intended license for project-owned trained weights is also AGPL-3.0-or-later. No weights exist in this bootstrap and none are licensed by implication as an existing release. A model release must explicitly identify its covered artifacts, upstream notices, and the available source/reproduction materials. We will publish training and inference code, exact configs, preprocessing/label specifications, dataset provenance and permitted reconstruction instructions, and evaluation/model-card materials to the extent distribution rights permit. Do not promise full reconstruction if inaccessible or restricted data makes that impossible; disclose the limitation and assess whether the proposed release satisfies its obligations.

Model-weight copyright and the application of software-license source obligations to trained artifacts are legally nuanced. Before releasing checkpoints, document the intended licensing treatment and obtain qualified advice where material uncertainty remains. Do not treat a LICENSE file alone as proof of an open, redistributable model or make categorical claims about downstream obligations. Review AGPL section 13 for any hosted modified-program deployment and plan an appropriate corresponding-source offer; do not assume a model endpoint automatically settles those obligations.

## Upstream work

The [official Whisper README](https://github.com/openai/whisper#license) states its code and model weights use MIT. Preserve its copyright/license notices in redistributed upstream material and derivatives as required. MIT upstream material does not become exclusively owned by this project when used in an AGPL distribution. No Whisper code or weights are vendored yet.

Audit each actual base model revision and checksum, fork/runtime, dependency, teacher model, dataset, and generated training corpus. Do not assume all Whisper-named models have OpenAI Whisper's license. Reject incompatible or noncommercial dependencies for the intended open release, or record a different explicit disposition before reuse.

## Speaker embedding model

`rightyo enroll` ([speaker enrollment](speaker-enrollment.md)) uses a separately provisioned
WeSpeaker ResNet34-LM speaker-embedding model. RightyO does not bundle or redistribute it.

- **Work:** `voxceleb_resnet34_LM.onnx`, revision `f0c48c298fd835726c27956a5d617bad7115627e`,
  SHA-256 `7bb2f06e9df17cdf1ef14ee8a15ab08ed28e8d0ef5054ee135741560df2ec068`
  ([model card](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM)).
- **Creators:** the [WeSpeaker](https://github.com/wenet-e2e/wespeaker) project (Hongji Wang,
  Chengdong Liang, Shuai Wang, Zhengyang Chen, Binbin Zhang, Xu Xiang, Yanlei Deng and Yanmin
  Qian, "WeSpeaker: A Research and Production Oriented Speaker Embedding Learning Toolkit",
  ICASSP 2023); the r-vector architecture follows Zeinali et al., "BUT System Description to
  VoxCeleb Speaker Recognition Challenge 2019".
- **Licence:** [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/), per the model card
  and WeSpeaker's [pretrained-model notes](https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md).
  RightyO uses the weights unmodified. Its fbank front end is an independent numpy
  reimplementation and imports no WeSpeaker code (Apache-2.0).
- **Training data:** VoxCeleb2 dev (Chung, Nagrani and Zisserman, VGG, University of Oxford),
  CC-BY-4.0, which its creators describe as available for research purposes. Mick decided on
  [#137](https://github.com/mickdarling/rightyo/issues/137) that this wording is not a
  blocker: RightyO is research for personal use, and any release would be a free public
  tool. Keep these attributions with any setup that provisions the model.
- **Runtime:** onnxruntime (MIT) and numpy (BSD-3-Clause), in an interpreter the user
  provides.

## Datasets and consent

Training permission, redistribution permission, personal-data consent, and model-release permission are separate checks. Record them separately. Public availability is not permission. Public GitHub issues must not contain participant identity, recordings, transcripts, paths exposing identity, or private consent records.

Public dataset documentation may describe lawful access/reconstruction without republishing raw speech. Publish data only when its license and consent explicitly permit it. Never relicense another party's dataset merely by storing it under this repository.

## Release gate

Before publishing weights: complete the license inventory; record base/teacher/checkpoint provenance; inspect for private content and memorization risk; include upstream notices, declared artifact license, model card, checksums, version, source/config links, and honest reproduction limits. A restricted or uncertain artifact must not be uploaded to a public registry until resolved. AGPL publication does not by itself settle patents, trademark availability, or training-data rights.
