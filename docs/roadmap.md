# Initial roadmap

GitHub issues are the execution records; [roadmap #1](https://github.com/mickdarling/rightyo/issues/1) tracks completion. This document establishes dependency order.

1. **Fast CI and review setup.** Install the secretless PR lane and test its actual GitHub runs. Subscription-backed Codex reviews must bind to the current head; automated trusted review enforcement remains a separate setup task.
2. **Transcript-first MVP.** Evaluate local Whisper file transcription plus Jev before training. Support anonymous speakers through a replaceable diarization boundary. Compare plain transcripts, inferred speakers and explicitly labeled oracle speakers; synthetic fixtures prove plumbing, not accuracy.
3. **Consented evaluation manifest and metrics.** Establish held-out grouping, ambiguous labels, hard negatives, causal replay, privacy controls and the quality/compute break-even comparison. Raw data stays outside Git.
4. **Conditional audio attention baseline and gated transcription.** Audit usable upstream models, weights and rights, then evaluate a compact audio model only if the MVP measurements justify it. Decide whether a shared encoder is justified from measured compute/latency.
5. **Live Mac pilot and downstream integration design.** Only after feasibility, add opt-in visible capture and plan Hailing Station integration. No mobile semantic deployment is part of the first pilot.

## Work records

| Stage | Issues |
| --- | --- |
| Fast baseline and anonymous speakers | [#23 transcript-first MVP](https://github.com/mickdarling/rightyo/issues/23), [#24 diarization](https://github.com/mickdarling/rightyo/issues/24), [#25 comparison](https://github.com/mickdarling/rightyo/issues/25) |
| CI and agent workflow | [#15 fast checks](https://github.com/mickdarling/rightyo/issues/15), [#16 workflow/artifact hygiene](https://github.com/mickdarling/rightyo/issues/16), [#17 dual reviews](https://github.com/mickdarling/rightyo/issues/17), [#18 trusted merge gate](https://github.com/mickdarling/rightyo/issues/18), [#22 agent protocol](https://github.com/mickdarling/rightyo/issues/22) |
| Research, rights, and setup | [#2 architecture/rights](https://github.com/mickdarling/rightyo/issues/2), [#3 experiment environment](https://github.com/mickdarling/rightyo/issues/3), [#13 CI/review enforcement](https://github.com/mickdarling/rightyo/issues/13) |
| Data and benchmarks | [#4 consent/manifests](https://github.com/mickdarling/rightyo/issues/4), [#5 annotation/splits/causal benchmark](https://github.com/mickdarling/rightyo/issues/5) |
| First working pipeline | [#6 replaceable ports/controller](https://github.com/mickdarling/rightyo/issues/6), [#7 frozen-Whisper attention head](https://github.com/mickdarling/rightyo/issues/7), [#8 gated ASR](https://github.com/mickdarling/rightyo/issues/8) |
| Evidence and optimization | [#9 calibration/held-out results](https://github.com/mickdarling/rightyo/issues/9), [#10 conditional distillation/export](https://github.com/mickdarling/rightyo/issues/10) |
| Release and integration | [#11 model registry/cards/source/privacy](https://github.com/mickdarling/rightyo/issues/11), [#12 local Mac pilot/Hailing Station adapter](https://github.com/mickdarling/rightyo/issues/12) |

Every issue includes acceptance criteria, dependencies, and verification requirements. The baseline workflow is described in [CI](ci.md); Claude rollout and stronger publisher enforcement remain tracked in #17/#18. The MVP is documented in [the experiment guide](mvp.md). No TestFlight build change follows from this repository.
