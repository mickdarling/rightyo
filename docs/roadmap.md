# Initial roadmap

GitHub issues are the execution records; [roadmap #1](https://github.com/mickdarling/rightyo/issues/1) tracks completion. This document establishes dependency order.

1. **Feasibility and prior-art audit.** Find usable audio attention implementations/weights, inspect the close Attention Labs work, audit datasets/licenses, and decide a first candidate. Avoid training from scratch before checking reuse.
2. **Typed stream contract and deterministic replay rig.** Implement replaceable attention/ASR ports, bounded pre-roll, causal windows, provenance, stop/restart, and validation with stubs. Stubs prove plumbing, not model accuracy.
3. **Consented evaluation manifest and metrics.** Establish held-out grouping, ambiguous labels, hard negatives, causal replay, and privacy controls. Raw data stays outside Git.
4. **Compact attention baseline and gated transcription.** Evaluate a real audio model and gated ASR against the benchmark. Decide whether a shared encoder is justified from measured compute/latency.
5. **Live Mac pilot and downstream integration design.** Only after feasibility, add opt-in visible capture and plan Hailing Station integration. No mobile semantic deployment is part of the first pilot.

## Work records

| Stage | Issues |
| --- | --- |
| Research, rights, and setup | [#2 architecture/rights](https://github.com/mickdarling/rightyo/issues/2), [#3 experiment environment](https://github.com/mickdarling/rightyo/issues/3), [#13 CI/review enforcement](https://github.com/mickdarling/rightyo/issues/13) |
| Data and benchmarks | [#4 consent/manifests](https://github.com/mickdarling/rightyo/issues/4), [#5 annotation/splits/causal benchmark](https://github.com/mickdarling/rightyo/issues/5) |
| First working pipeline | [#6 replaceable ports/controller](https://github.com/mickdarling/rightyo/issues/6), [#7 frozen-Whisper attention head](https://github.com/mickdarling/rightyo/issues/7), [#8 gated ASR](https://github.com/mickdarling/rightyo/issues/8) |
| Evidence and optimization | [#9 calibration/held-out results](https://github.com/mickdarling/rightyo/issues/9), [#10 conditional distillation/export](https://github.com/mickdarling/rightyo/issues/10) |
| Release and integration | [#11 model registry/cards/source/privacy](https://github.com/mickdarling/rightyo/issues/11), [#12 local Mac pilot/Hailing Station adapter](https://github.com/mickdarling/rightyo/issues/12) |

Every issue includes acceptance criteria, dependencies, and verification requirements. Model-aware CI and exact-head review enforcement are tracked in #13; they are not already configured by this design-only bootstrap. No TestFlight build change follows from this repository bootstrap.
