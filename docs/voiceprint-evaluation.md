# Voiceprint model spike: synthetic voices only

Date: 2026-10-09. This is step 1 of [#137](https://github.com/mickdarling/rightyo/issues/137):
an offline comparison of local speaker-embedding models for enrollment-based identification.
**All audio was authored text rendered by built-in macOS `say` voices.** No microphone, real
person's voice, cloned voice, downloaded speech dataset or hosted processing was used. Audio,
embeddings and per-run metrics stayed in a scratch directory outside Git and were deleted after
the run. Nothing here changes the live path.

**Caveat first.** Synthetic TTS voices are much easier to separate than real people: every
clip from one voice is acoustically consistent, with no colds, mood, distance or microphone
changes. These numbers measure mechanics (short-turn behaviour, noise sensitivity, latency,
memory, integration cost). They are **not** real-world accuracy, and the thresholds below are
starting points only. Real-voice validation must happen locally, by the user, with the
enrollment tool from step 3; no real voice data may ever leave the host (#109).

## Result and recommendation

All four models separate ten synthetic voices almost perfectly from 1 s up when clean. Short
(0.5 s) turns and noisy, reverberant turns are where they differ, and where all of them degrade.

- **Use WeSpeaker ResNet34-LM (ONNX) for steps 3 and 4**, in a private worker process like
  Smart Turn's. It is the same network as SpeakerKit's embedder (see below), scored the same,
  runs in the **existing Smart Turn interpreter with no new dependency** (numpy + onnxruntime,
  verified), loads in 0.03 s, and embeds a 1–2 s turn in 8–16 ms on four CPU threads.
- **Keep SpeakerKit's Core ML embedder as an equivalent alternative**, not the first choice:
  driving it needs coremltools or a new Swift helper, it always processes a fixed 30 s window,
  and it gives chance-level results if short clips are zero-padded (a trap for step 4).
- **SpeechBrain ECAPA-TDNN is the most noise-robust** here (2 s noisy EER 1.6% versus 3.8%),
  but needs PyTorch: about 0.45 GiB RSS after load, 1.2 GiB peak, a ~0.9 GiB environment.
  Re-test it if real-voice validation shows noise problems.
- **CAM++ is fastest and about as robust as ResNet34**, but its licence is inconsistent
  between primary sources (see Licences). Don't adopt it until that is resolved.

Suggested starting thresholds (cosine against a 30 s clean voiceprint, ResNet34-LM):
**bind a session label to an enrolled identity only at ≥ 0.60**; treat 0.45–0.60 as tentative
(no role authority); below that, `unknown`. Don't use turns shorter than 1 s of speech for
identity on their own; accumulate them per label. Rationale and caveats are under
[Thresholds](#thresholds).

## Method

Hardware: Apple M4 Max, macOS 15.8, arm64. Python 3.11.14, numpy 2.4.6.
Harness: [`scripts/voiceprint_eval.py`](../scripts/voiceprint_eval.py) (in this PR).

- **Speakers.** Ten enrolled voices: Samantha, Daniel, Karen, Moira, Rishi, Tessa, Fred,
  Kathy, Ralph, Reed (US). Four unenrolled impostors: Shelley (US), Flo (US) and Albert for
  open-set rejection, and Reed (UK), a near-twin of the enrolled Reed (US) (same voice
  family, different accent), reported separately.
- **Enrollment.** One authored passage at the default rate, silences trimmed, cut to exactly
  30.0 s of audio; one embedding of the whole 30 s, L2-normalised.
- **Held-out test.** A different passage rendered at default, 150 and 220 words per minute.
  From it, 24 non-overlapping clips per speaker per duration (0.5, 1, 2, 4 s), each at least
  70% voiced. Per duration: 240 target trials and 2,160 non-target trials; 72 open-set and 24
  near-twin clips.
- **Room condition.** Each test clip again with synthetic reverb (RT60 0.4 s, exponentially
  decaying noise tail) plus pink noise at 10 dB SNR. Enrollment stays clean (mismatched).
- **Scores.** Cosine between test embedding and every voiceprint. EER from the
  target/non-target distributions; top-1 is closed-set identification accuracy.
- **Cost.** A separate fresh process per model (`footprint`): load time, RSS after load, peak
  RSS (including one 30 s enrollment embed) and warm median latency over 20 runs after three
  warm-ups. Timings include feature extraction and Python call overhead.

## Separation and short turns

EER in percent (lower is better). Clean / room (reverb + 10 dB pink noise).

| Model | 0.5 s | 1 s | 2 s | 4 s |
| --- | --- | --- | --- | --- |
| SpeakerKit embedder (Core ML, W8A16) | 4.5 / 17.1 | 1.3 / 9.2 | 0.4 / 3.8 | 0.0 / 3.0 |
| WeSpeaker ResNet34-LM (ONNX) | 3.8 / 17.9 | 1.3 / 9.2 | 0.4 / 3.8 | 0.0 / 2.9 |
| WeSpeaker CAM++ (ONNX) | 3.3 / 18.3 | 0.8 / 7.5 | 0.0 / 4.2 | 0.0 / 2.9 |
| SpeechBrain ECAPA-TDNN | 2.1 / 16.7 | 0.8 / 5.8 | 0.1 / 1.6 | 0.0 / 0.9 |

Closed-set top-1 was 97.5–98.8% at 0.5 s and 100% from 1 s when clean; in the room condition
it was 64–75% at 0.5 s, 89–95% at 1 s and 96–99% at 2 s.

Score distributions are similar across models (ResNet34-LM shown; others within ~0.05):

| ResNet34-LM | 0.5 s | 1 s | 2 s | 4 s |
| --- | --- | --- | --- | --- |
| Target mean (5th pct), clean | 0.54 (0.35) | 0.69 (0.53) | 0.80 (0.69) | 0.88 (0.76) |
| Non-target mean (95th pct), clean | 0.14 (0.32) | 0.16 (0.36) | 0.19 (0.40) | 0.20 (0.41) |
| Target mean, room | 0.24 | 0.34 | 0.45 | 0.52 |
| Non-target mean, room | 0.07 | 0.08 | 0.10 | 0.11 |
| EER threshold, clean / room | 0.34 / 0.14 | 0.46 / 0.21 | 0.54 / 0.29 | 0.61 / 0.36 |

Two effects matter for live use. Target scores collapse under mismatched noise and reverb
(0.80 → 0.45 at 2 s), so a threshold chosen on clean audio rejects most noisy turns. And
non-target scores rise with clip length, so the EER threshold drifts upward with duration.

**Hard cases.** The near-twin Reed (UK) scored 0.65–0.68 on average against Reed (US),
maximum 0.85–0.89, inside the target range for every model: no threshold separates them.
The unenrolled voices from the same synthesis family (Shelley, Flo) reached best scores of
0.50–0.61 clean against some enrolled voice. Real relatives or similar voices may behave
like this; it is the main open question for real-voice validation.

## Cost

Fresh process, warm median per call. CPU models use four threads (one-thread figures in
brackets). SpeakerKit uses Core ML `cpuAndNeuralEngine`, the SpeakerKit default.

| Model | On disk | Load | RSS after load | Peak RSS | 0.5 s | 1 s | 2 s | 4 s | 30 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SpeakerKit | 9.3 MB | 0.27 s | 120 MiB | 211 MiB | 24 ms | 24 ms | 25 ms | 24 ms | 24 ms |
| ResNet34-LM | 26.5 MB | 0.03 s | 101 MiB | 510 MiB | 7 ms | 8 (17) ms | 16 (29) ms | 28 (60) ms | 188 ms |
| CAM++ | 29.3 MB | 0.25 s | 119 MiB | 453 MiB | 6 ms | 8 (8) ms | 10 (13) ms | 20 (23) ms | 85 ms |
| ECAPA-TDNN | 83.3 MB | 1.1 s | 452 MiB | 1,182 MiB | 13 ms | 17 (13) ms | 28 (24) ms | 59 (41) ms | 175 ms |

- SpeakerKit costs the same for any clip length because the Core ML model takes a fixed 30 s
  window. With `all` compute units it took about 19 ms, with `cpuOnly` 63–68 ms. The first
  load on this machine took 5.6 s (Core ML compilation/caching); later loads 0.27 s. RSS does
  not capture Neural Engine or GPU allocations.
- Peak RSS for the CPU models is dominated by the one 30 s enrollment embed; live turns of a
  few seconds stay much nearer the after-load figure. ECAPA's first cold import took 14.5 s.
- ResNet34-LM ran unchanged under the existing Smart Turn interpreter (onnxruntime 1.30.0):
  0.087 s load, 6/8/13/28 ms for 0.5/1/2/4 s.

## SpeakerKit embedder: how to drive it, and a trap

The cached `speaker_embedder/pyannote-v3/W8A16` is two Core ML programs:
`SpeakerEmbedderPreprocessor` (480,000 samples, 30 s at 16 kHz, to 2,998 × 80 fbank) and
`SpeakerEmbedder` (fbank plus 64 speaker masks over 1,767 segmentation frames, to 64 × 256
embeddings). Its `README.txt` points to the WeSpeaker model licence, and it is evidently the
`pyannote/wespeaker-voxceleb-resnet34-LM` network with 8-bit weights; its embeddings match
ResNet34-LM's closely (below).

The least-effort path for a spike is `coremltools.models.CompiledMLModel` on the `.mlmodelc`
directories, filling mask slot 0 for the clip and leaving the other 63 slots at zero. For
production, a small Swift helper would avoid a Python Core ML dependency;
argmax-oss-swift's `SpeakerEmbedderModel` shows the masking scheme but is internal to the
diarization pipeline and doesn't expose an "embed this clip" API.

- **Don't zero-pad short clips.** Placing a 2 s clip in a 30 s window of digital silence gave
  chance-level results (target 0.05, non-target 0.05), whatever the mask. Tiling the clip to
  fill the window works (target 0.82, non-target 0.18), as does surrounding real audio with a
  partial mask (tested with tiling). Low-level noise padding partly recovers (0.63 / 0.13).
  The harness tiles. Live use should pass the clip with real surrounding context.
- **Same network as ResNet34-LM.** For the same clip, the two models' embeddings had mean
  cosine 0.93 (minimum 0.57, on very short clips). Enrolling with the ONNX model and testing
  with Core ML still separated speakers (2 s target 0.80, non-target 0.18). Even so, store the
  model identity with each voiceprint and re-enroll if the model changes.

## Licences and provenance

The harness never downloads anything; models were fetched once into the scratch directory
for this spike and deleted afterwards. No weights are redistributed by RightyO.

| Model | Weights source | Weights licence | Runtime |
| --- | --- | --- | --- |
| SpeakerKit embedder | `argmaxinc/speakerkit-coreml` @ `556fc52`, already provisioned ([smoke record](mvp-smoke.md)) | CC-BY-4.0 (Argmax card); derived from pyannote/WeSpeaker ResNet34-LM, CC-BY-4.0 | Core ML via coremltools 9.0 (BSD-3-Clause) or Swift |
| ResNet34-LM | `Wespeaker/wespeaker-voxceleb-resnet34-LM` @ `f0c48c2`, `voxceleb_resnet34_LM.onnx`, 26,530,309 B, SHA-256 `7bb2f06e…c068` | CC-BY-4.0 (card and WeSpeaker docs) | onnxruntime (MIT) + numpy (BSD-3-Clause) |
| CAM++ | `Wespeaker/wespeaker-voxceleb-campplus` @ `acf623a`, `voxceleb_CAM++.onnx`, 29,292,449 B, SHA-256 `b5081049…efad` | **Conflict:** card says Apache-2.0; WeSpeaker docs say VoxCeleb models follow CC-BY-4.0 | onnxruntime + numpy |
| ECAPA-TDNN | `speechbrain/spkrec-ecapa-voxceleb` @ `0f99f2d`, `embedding_model.ckpt` SHA-256 `0575cb64…26a2` | Apache-2.0 (card) | PyTorch 2.14.1 (BSD-3-Clause), SpeechBrain 1.1.1 (Apache-2.0) |

- WeSpeaker's [pretrained-model page](https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md)
  says its VoxCeleb models follow the dataset licence, CC-BY-4.0. WeSpeaker and SpeechBrain
  code are Apache-2.0. The harness reimplements Kaldi fbank in numpy and imports no WeSpeaker
  code.
- All four were trained on [VoxCeleb](https://mm.kaist.ac.kr/datasets/voxceleb/), which is
  CC-BY-4.0 but describes itself as "available to download for research purposes", with video
  copyright staying with the owners. Keep attribution notices when provisioning; whether any
  of these weights could be bundled or redistributed is a separate release decision.
- The WeSpeaker ResNet34-LM card itself has an incorrect `summarization` pipeline tag;
  that doesn't affect the licence.

## Integration into RightyO

- **Step 3 (enrollment).** Read up to 30–60 s, embed the whole clip (or the mean of 3–4 s
  windows), L2-normalise, store with model ID and SHA-256, mode 700, outside the repository.
  The voice-data guard from #138 already refuses voiceprints in Git.
- **Step 4 (live).** A worker modelled on `src/rightyo/smart_turn.py`: same configured
  interpreter (numpy + onnxruntime), an explicit local `.onnx` path, PCM in, a 256-float
  vector or just scores out, nothing persisted. About 60 lines of fbank code plus the
  worker protocol. Per finalised turn: embed, compare with voiceprints, then update a
  per-label running mean so short turns accumulate evidence before the label binds.
- **SpeakerKit instead** would need a coremltools environment or a Swift helper,
  window tiling or real context, and the fixed 24 ms per call.

## Thresholds

Starting points for ResNet34-LM only, from synthetic voices. Recalibrate on the user's own
enrollment before relying on them. At a fixed per-turn threshold:

| Threshold | Clean 1 s FRR / FAR | Clean 2 s | Clean 4 s | Room 2 s | Room 4 s |
| --- | --- | --- | --- | --- | --- |
| 0.40 | 0.4% / 3.2% | 0% / 5.1% | 0% / 6.3% | 29% / 1.1% | 12% / 1.5% |
| 0.50 | 2.1% / 0.8% | 0.4% / 1.5% | 0% / 2.8% | 71% / 0.1% | 43% / 0.05% |
| 0.60 | 14% / 0% | 0.4% / 0% | 0% / 0.05% | 97% / 0% | 71% / 0% |

At 0.50, 10–21% of 2–4 s clips from unenrolled same-family voices (Shelley, Flo, Albert)
were accepted as some enrolled voice; at 0.60, none were. The near-twin's scores overlap the
target range, so no threshold separates it.

Because an enrolled role gains precedence (#113), a false accept is the costlier error. So
bind at ≥ 0.60 (sticky, with hysteresis), give 0.45–0.60 only tentative status, and fall back
to `unknown`. In noisy rooms with a quiet enrollment this mostly fails safe (the speaker
stays `unknown`), which is acceptable but not good. Two untested improvements: score margin
over the second-best voiceprint, and enrolling in the room where RightyO runs.

## Open questions

- Real-voice behaviour: same-person variation, relatives, TV and podcast voices. This can
  only be measured locally by the user, with the step 3 tool; record scores only.
- Should enrollment include room audio, or several sessions, to close the clean/noisy gap?
- Score normalisation (for example adaptive s-norm with a synthetic cohort) to stabilise the
  duration-dependent threshold.
- Whether the live diarizer can expose its own per-label embeddings, avoiding a second model.
  SpeakerKit's are compatible with ResNet34-LM; Nemotron's are not evaluated here.
- CAM++ licence conflict, and VoxCeleb's "research purposes" wording, before any bundling.

## Reproduce

Generation needs macOS. Pass a directory outside every Git checkout; the harness refuses
otherwise and never overwrites a metrics file. Metrics hold aggregate numbers only: no
audio, embeddings, text or local paths. Delete the directory afterwards.

```sh
python3 scripts/voiceprint_eval.py generate --out /private/tmp/vp/audio
/path/to/smart-turn-venv/bin/python scripts/voiceprint_eval.py evaluate \
  --data /private/tmp/vp/audio --backend wespeaker \
  --model /local/voxceleb_resnet34_LM.onnx --output /private/tmp/vp/r34.json
/path/to/coremltools-venv/bin/python scripts/voiceprint_eval.py evaluate \
  --data /private/tmp/vp/audio --backend speakerkit \
  --model ~/Library/Caches/rightyo/speakerkit/speaker_embedder/pyannote-v3/W8A16 \
  --output /private/tmp/vp/speakerkit.json
/path/to/speechbrain-venv/bin/python scripts/voiceprint_eval.py footprint \
  --data /private/tmp/vp/audio --backend speechbrain --model /local/spkrec-ecapa-voxceleb
rm -rf /private/tmp/vp
```

Defaults: 24 clips per duration, seed 137, four threads. The same seed reproduces the same
clips and noise from the same rendered audio. macOS voice updates can change the renders;
the manifest records their hashes.
