# Pipecat components: evaluation (#107)

We adopt Pipecat **components and patterns** inside RightyO's own pipeline; we do not rebuild on Pipecat. This note records each component's evaluation. Smart Turn comes first because it sits in front of the end-of-turn decision (#84) and the instant acknowledgement target (#106).

## 1. Smart Turn v3 (end of turn): spike results, 2026-10-08

| | |
| --- | --- |
| What it does | Reads the last ≤ 8 s of audio and returns P(turn complete) from intonation and phrasing, not from the transcript |
| Model | `smart-turn-v3.2-cpu.onnx` (int8, 8.7 MB), Whisper Tiny encoder + linear head, HF `pipecat-ai/smart-turn-v3` @ `f766f81` |
| License | BSD-2-Clause (model and code); the test data is CC BY 4.0 |
| Runtime | `onnxruntime` 1.30 on CPU, plus Whisper log-mel features (numpy). The reference code names v3.1; the harness swaps in the v3.2 CPU file with the same input and output |
| Verdict | **Adapt.** Fast enough with room to spare. Accuracy on the owner's speech is unmeasured; it needs real recordings (#115) before it can gate dispatch |

### What it would replace

Today a request reaches the decision only after two fixed waits that overlap (`live_audio.py`, `turn_merge.py`):

1. An energy VAD closes the utterance after `hangover_ms` = **1,440 ms** of silence; transcription runs then.
2. The turn merger holds the fragment until `turn_merge_gap_ms` = **2,000 ms** after its last word, in case the same speaker continues (#73). The hold counts from the last word, so it adds about 560 ms past the hangover. The `reply_wait_ms` (1,200 ms) hold for request-shaped fragments falls inside it.

So with the `listen` defaults, a finished request is decided about **2 s** after the last word (longer only if transcription takes more than the remaining ~560 ms). That 2 s is a large part of the "still a little slow" acknowledgement Mick heard live (#106). Smart Turn's role is to say "complete" about 200 ms after speech stops, so both waits can be skipped when it is confident, and kept as the fallback when it is not.

### Latency (M4 Max, CPU)

| Setting | Median | p95 |
| --- | --- | --- |
| ONNX only, `intra_op_num_threads=4` | 17 ms | 24 ms |
| ONNX only, 1 thread | 38 ms | 43 ms |
| ONNX only, ORT default threads | 26 ms | 71 ms |
| Features + ONNX (4 threads), 181 clips, Voicebox rendering concurrently | 36 ms | 63 ms |

That is well inside the ≤ 300 ms budget. Use a fixed small thread count: the default thread pool has a worse tail.

### Harness check: the vendor test set

The same harness, run on the 873 English rows of one shard (`data/train-00000-of-00009.parquet`, 1 of 9, sha256 prefix `272d19c7374fade9`) of `pipecat-ai/smart-turn-data-v3.1-test` @ `22b4ec7`:

| Source | Label | n | Correct |
| --- | --- | --- | --- |
| Human | complete | 312 | 97.8% |
| Human | incomplete | 295 | 95.6% |
| Synthetic | complete | 118 | 94.9% |
| Synthetic | incomplete | 148 | 92.6% |
| **All English** | | **873** | **95.8%** |

This is consistent with the published English figure (94.3% on the full set). It is a sanity check, not proof of equivalence: one shard is not the full set, and outputs were not compared example by example with the reference `inference.py`. The harness follows that code's preprocessing (last 8 s, left padding, Whisper features with `chunk_length=8`, threshold 0.5).

### Authored clips: 24 requests in Mick's style, 4 TTS voices (synthetic, not live speech)

Method:
- Each sentence has a marked mid-sentence pause point.
- Each sentence was rendered whole in local Voicebox: three Qwen 1.7B cloned voices and the Kokoro preset `am_michael`.
- whisper.cpp word timings located the pause point, and the render was cut there with 200 ms of silence added. That keeps the continuation intonation that a separately rendered prefix would lose.
- Result: 96 complete clips and 85 cuts. Three cuts were dropped where alignment failed ("Pipecat" was transcribed as "peeper cat").
- Cuts are split into *dangling* ones, which end mid-phrase (for example on and, the, to or should) and are never finished, and *complete-sounding* ones ("Turn the volume down"), which a listener could also take as finished.

At threshold 0.5:

| Measure | Result |
| --- | --- |
| Dangling cut treated as complete (would dispatch mid-sentence) | 23 / 52 |
| Complete-sounding cut treated as complete | 17 / 33 (arguably correct) |
| Finished request held as incomplete | 15 / 96 |

Raising the threshold trades one error for the other. Neither error rate gets near the vendor numbers:

| Threshold | Dangling dispatched | Finished held |
| --- | --- | --- |
| 0.5 | 23 / 52 | 15 / 96 |
| 0.7 | 17 / 52 | 18 / 96 |
| 0.9 | 13 / 52 | 29 / 96 |

The trailing silence length moves the result a lot. The model has learned that silence means done:

| Silence after the last word | Dangling dispatched | Finished held |
| --- | --- | --- |
| 0 ms | 11 / 52 | 23 / 96 |
| 200 ms | 23 / 52 | 15 / 96 |
| 500 ms | 35 / 52 | 6 / 96 |

(Adding a random −55 dB noise floor instead of digital silence changed these by at most 6.)

### Reading

- The gap between 95.8% on the vendor set and these clips is most likely the **clips**. Cutting fluent TTS at a word boundary does not sound like a person pausing mid-thought: there is no lengthening, no filler and no held pitch, and the TTS voices were never asked to pause. Vendor *synthetic* incompletes still score 92.6%, but they were generated to be incomplete, not cut from a fluent sentence.
- Authored clips therefore **cannot** tell us how Smart Turn handles Mick's speech. They do settle latency and the harness, and they show that the evaluation point (how much silence the model sees) must be fixed and reported.
- Mick's speech has long thinking pauses and trailing conjunctions. Measuring it needs real audio (#115: an opt-in local recorder, then a personal fine-tune or threshold).

### What can be adjusted

Each decision is opaque (one probability, no explanation), but the model is fully open, so it is tunable at several depths. From cheapest to deepest:

| Lever | Effect | Cost |
| --- | --- | --- |
| Threshold | Trades false dispatches against held requests (table above) | One number; could be live-adjustable once integrated |
| Evaluation point | How much silence the model hears before it is asked; the biggest single effect measured here | One number |
| Fallback wait | How long to keep waiting after an "incomplete" before dispatching anyway | Today's hangover and merge hold |
| Transcript guard | Never dispatch when the transcript ends in and / the / to and so on | A small rule after ASR |
| Fine-tune on the owner's speech | Learns the owner's pauses and phrasing; `train.py`, weights and data are open (#115) | Owner recordings, then (estimated, not measured) minutes to hours on the M4 Max |

### Integration (#117)

Implemented behind the prototype's `end_of_turn` switch (see [prototype](prototype.md)), off by default. One finding from replay changed the design: finalizing 200 ms after speech left the last words outside the streaming diarizer's timeline (it trails the audio by 0.4–1.2 s), so they lost their speaker and split the turn. A turn judged complete therefore waits until the timeline covers its last voiced audio. On an authored replay with the real diarizer, finished requests were emitted 300–740 ms after their last word instead of about 2,000 ms, and turn splitting matched the silence-only run exactly (synthetic voices, not live speech).

The original proposal follows.

### Proposed integration (follow-up issue)

1. Run Smart Turn when the VAD has seen ~200 ms of silence, on the utterance audio already in the window (no extra capture).
2. If P(complete) ≥ threshold, finalize at once and skip the merge hold. This also gives #106 its early end-of-turn signal for the acknowledgement.
3. Otherwise keep today's 1,440 ms hangover and merge hold as the fallback, re-checking at each further pause.
4. Behind a config switch, off by default, so the owner can turn it on in live use and judge it directly; the threshold and evaluation point are config values, tuned on recorded sessions later.
5. Log `p`, the evaluation point and the outcome per turn so false cuts can be counted live.

## 2–6. Remaining components

To be evaluated in order: interruption and barge-in (#98), filler and "thinking" audio (#105, #335 in hailing-station), per-stage metrics, streaming STT partials, and speculative generation. Each gets a section like the one above, with an adopt, adapt or skip verdict.

## Reproducing

The spike scripts (clip generator, clip and test-set evaluators) are kept outside the repository with the audio for now. A public-safe harness, sentence list, cut points and manifest hashes will be committed with the integration work (#117). The reference inference code is `pipecat-ai/smart-turn` `inference.py` @ `4786657`.
