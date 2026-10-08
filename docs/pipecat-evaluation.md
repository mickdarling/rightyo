# Pipecat components: evaluation (#107)

We adopt Pipecat **components and patterns** inside RightyO's own pipeline; we do not rebuild on Pipecat. This note records each component's evaluation. Smart Turn comes first because it sits in front of the end-of-turn decision (#84) and the instant acknowledgement target (#106).

## 1. Smart Turn v3 (end of turn): spike results, 2026-10-08

| | |
| --- | --- |
| What it does | Reads the last ≤ 8 s of audio and returns P(turn complete) from intonation and phrasing, not from the transcript |
| Model | `smart-turn-v3.2-cpu.onnx` (int8, 8.7 MB), Whisper Tiny encoder + linear head, HF `pipecat-ai/smart-turn-v3` @ `f766f81` |
| License | BSD-2-Clause (model and code); the test data is CC BY 4.0 |
| Runtime | `onnxruntime` 1.30 on CPU, plus Whisper log-mel features (numpy) |
| Verdict | **Adapt.** Fast enough with room to spare. Accuracy on the owner's speech is unmeasured; it needs real recordings (#115) before it can gate dispatch |

### What it would replace

Today a request reaches the decision only after two fixed waits (`live_audio.py`, `turn_merge.py`):

1. An energy VAD closes the utterance after `hangover_ms` = **1,440 ms** of silence.
2. The turn merger holds the fragment for `turn_merge_gap_ms` = **2,000 ms** in case the same speaker continues (#73); a request-shaped fragment is also held for `reply_wait_ms` = 1,200 ms, which overlaps the merge hold.

So a finished request waits about 3.4 s of silence plus transcription before it is decided. That is most of the "still a little slow" acknowledgement Mick heard live (#106). Smart Turn's role is to say "complete" about 200 ms after speech stops, so both waits can be skipped when it is confident, and kept as the fallback when it is not.

### Latency (M4 Max, CPU, Voicebox rendering concurrently)

| Setting | Median | p95 |
| --- | --- | --- |
| ONNX only, `intra_op_num_threads=4` | 21 ms | 40 ms |
| ONNX only, 1 thread | 38 ms | 47 ms |
| ONNX only, ORT default threads | 41 ms | 85 ms |
| Features + ONNX (4 threads), 181 clips | 36 ms | 63 ms |

That is well inside the ≤ 300 ms budget. Use a fixed small thread count: the default thread pool has a worse tail.

### Harness check: the vendor test set

The same harness, run on the 873 English rows of one shard (1 of 9) of `pipecat-ai/smart-turn-data-v3.1-test`:

| Source | Label | n | Correct |
| --- | --- | --- | --- |
| Human | complete | 312 | 97.8% |
| Human | incomplete | 295 | 95.6% |
| Synthetic | complete | 118 | 94.9% |
| Synthetic | incomplete | 148 | 92.6% |
| **All English** | | **873** | **95.8%** |

This reproduces the published English figure (94.3% on the full set), so the harness matches the reference `inference.py`.

### Authored clips: 24 requests in Mick's style, 4 TTS voices (synthetic, not live speech)

Method:
- Each sentence has a marked mid-sentence pause point.
- Each sentence was rendered whole in Voicebox: Feynman, Jarvis and Fry (Qwen 1.7B clones) and Kokoro `am_michael`.
- whisper.cpp word timings located the pause point, and the render was cut there with 200 ms of silence added. That keeps the continuation intonation that a separately rendered prefix would lose.
- Result: 96 complete clips and 85 cuts. Three cuts were dropped where alignment failed ("Pipecat" was transcribed as "peeper cat").
- Cuts are split into *dangling* ones, which end on and / the / to / is and are never finished, and *complete-sounding* ones ("Turn the volume down"), which a listener could also take as finished.

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

(Adding a −55 dB noise floor instead of digital silence changed these by at most 4.)

### Reading

- The gap between 95.8% on the vendor set and these clips is most likely the **clips**. Cutting fluent TTS at a word boundary does not sound like a person pausing mid-thought: there is no lengthening, no filler and no held pitch, and the TTS voices were never asked to pause. Vendor *synthetic* incompletes still score 92.6%, but they were generated to be incomplete, not cut from a fluent sentence.
- Authored clips therefore **cannot** tell us how Smart Turn handles Mick's speech. They do settle latency and the harness, and they show that the evaluation point (how much silence the model sees) must be fixed and reported.
- Mick's speech has long thinking pauses and trailing conjunctions. Measuring it needs real audio (#115: an opt-in local recorder, then a personal fine-tune or threshold).

### Proposed integration (follow-up issue, after #115 has data)

1. Run Smart Turn when the VAD has seen ~200 ms of silence, on the utterance audio already in the window (no extra capture).
2. If P(complete) ≥ threshold, finalize at once and skip the merge hold. This also gives #106 its early end-of-turn signal for the acknowledgement.
3. Otherwise keep today's 1,440 ms hangover and merge hold as the fallback, re-checking at each further pause.
4. The threshold is per owner and is tuned on recorded sessions.
5. Log `p`, the evaluation point and the outcome per turn so false cuts can be counted live.

## 2–6. Remaining components

To be evaluated in order: interruption and barge-in (#98), filler and "thinking" audio (#105, #335 in hailing-station), per-stage metrics, streaming STT partials, and speculative generation. Each gets a section like the one above, with an adopt, adapt or skip verdict.

## Reproducing

The spike scripts (clip generator, clip and test-set evaluators) are kept outside the repository with the audio. The method above is enough to rebuild them. The reference inference code is `pipecat-ai/smart-turn` `inference.py` @ `4786657`.
