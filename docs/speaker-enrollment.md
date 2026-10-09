# Speaker enrollment

`rightyo enroll` builds a voiceprint for each person you want RightyO to recognise, and keeps
it only on your machine. This is step 3 of
[#137](https://github.com/mickdarling/rightyo/issues/137). **Live identification is not wired
yet** (step 4): enrolling changes nothing in `listen` or the lab today. `enroll verify` lets
you check locally that enrollment separates the people you enrolled.

## Privacy

- RightyO ships code and synthetic test signals only. It contains no voices, voiceprints or
  enrollment data, and each user enrolls their own speakers on their own machine
  ([#109](https://github.com/mickdarling/rightyo/issues/109)).
- Audio is read from the file you name, or recorded from the microphone when you pass
  `--record`, into memory only. It is embedded and discarded, never written.
- The store holds one JSON file per enrolled speaker: the voiceprint vector and minimal
  metadata (identifier, display name, model identity, creation time, seconds of speech used,
  window count). The directory is mode 700 and each file mode 600; RightyO tightens looser
  modes when it opens the store.
- The default store is `~/Library/Application Support/RightyO/enrollment/`. RightyO refuses a
  store inside any Git checkout, a symlinked store, or one owned by another user.
- Command output carries identifiers, display names, durations and scores only: no audio,
  embeddings, transcript text or local paths. Don't paste scores of real people into public
  issues; summarise them.
- A voiceprint is still biometric data about a person. Enroll someone only with their
  agreement, and delete their voiceprint when they ask.

## Provision the model

The voiceprint model is WeSpeaker ResNet34-LM (ONNX, CC-BY-4.0), chosen in
[the voiceprint evaluation](voiceprint-evaluation.md). RightyO never downloads it. Download
it once yourself, outside any Git checkout, and check its hash:

| Field | Value |
| --- | --- |
| Source | [`Wespeaker/wespeaker-voxceleb-resnet34-LM`](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM) |
| Revision | `f0c48c298fd835726c27956a5d617bad7115627e` |
| File | `voxceleb_resnet34_LM.onnx`, 26,530,309 bytes |
| SHA-256 | `7bb2f06e9df17cdf1ef14ee8a15ab08ed28e8d0ef5054ee135741560df2ec068` |
| Licence | CC-BY-4.0; see [licensing](licensing.md#speaker-embedding-model) |

```sh
mkdir -p ~/Library/Caches/rightyo/wespeaker
curl -L -o ~/Library/Caches/rightyo/wespeaker/voxceleb_resnet34_LM.onnx \
  https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM/resolve/f0c48c298fd835726c27956a5d617bad7115627e/voxceleb_resnet34_LM.onnx
shasum -a 256 ~/Library/Caches/rightyo/wespeaker/voxceleb_resnet34_LM.onnx
```

RightyO checks the SHA-256 itself and refuses any other file. The model runs in a separate
worker process under an interpreter that has `numpy` and `onnxruntime`; the Smart Turn
interpreter from the [end-of-turn setup](prototype.md) is enough, and RightyO installs
neither.

## Configure

Add a `speaker_id` section to your local configuration (for example the gitignored
`local/prototype.json`):

```json
"speaker_id": {
  "python": "/Users/you/Library/Caches/rightyo/smart-turn/venv/bin/python",
  "model": "/Users/you/Library/Caches/rightyo/wespeaker/voxceleb_resnet34_LM.onnx"
}
```

Optional keys: `store` (an absolute directory, default
`~/Library/Application Support/RightyO/enrollment`), `bind_threshold` (default 0.60),
`tentative_threshold` (default 0.45, below the bind threshold), `min_turn_seconds` (default
1.0, 0.5 to 10), `threads` (1 to 8, default 4) and `enabled` (default true). The section is
off when absent or `"enabled": false`.

The thresholds come from the evaluation's per-turn scores on synthetic voices. They are
starting points: recalibrate them on your own enrollment with `verify`. Step 4 binds on a
running per-label average, where non-target scores rise with more audio, so it will
recalibrate them for accumulated scores
([#141](https://github.com/mickdarling/rightyo/issues/141)).

## Enroll, list, verify and delete

Enrollment needs at least 20 s of speech after silence is trimmed, and warns below 30 s;
30 to 60 s of one person talking naturally, in a quiet room, works best. At most 90 s is
used. Files must be 16 kHz mono 16-bit PCM WAV, up to five minutes; convert other
recordings with macOS `afconvert`:

```sh
afconvert -f WAVE -d LEI16@16000 -c 1 input.m4a /private/tmp/enroll.wav
```

```sh
# From a file (read into memory; the file itself is untouched)
.venv/bin/rightyo enroll add --config local/prototype.json \
  --id owner --name "Your Name" --from /private/tmp/enroll.wav

# Or record 45 s from the microphone (may prompt for microphone permission)
.venv/bin/rightyo enroll add --config local/prototype.json \
  --id owner --name "Your Name" --record 45

.venv/bin/rightyo enroll list
.venv/bin/rightyo enroll verify --config local/prototype.json --record 5
.venv/bin/rightyo enroll delete --id owner
.venv/bin/rightyo enroll delete --all
```

- Identifiers are 1 to 32 lowercase letters, digits, `-` or `_`. Adding an existing
  identifier is refused unless you pass `--replace`.
- `list` and `delete` need no configuration; pass `--config` or `--store` if you moved the
  store. `delete --all` removes every voiceprint and then the directory.
- `verify` (1 to 30 s of speech) prints one cosine score per enrolled speaker with its band:
  `bind` at or above `bind_threshold`, `tentative` at or above `tentative_threshold`,
  otherwise `below`. Voiceprints made with a different model file are counted, not scored;
  re-enroll them.
- How it works: speech is cut into 3 s windows with 1.5 s hop. The voiceprint is the mean
  of the L2-normalised window embeddings, renormalised. `verify` scores its clip the same
  way, so its scores use more audio than one short live turn and run somewhat higher.
- Delete any WAV you converted or recorded for enrollment once it is enrolled; RightyO
  never keeps a copy, but it does not remove your files either.
- The speech gate is a level gate, not a voice detector: background music or steady noise
  counts as speech. Record one person, close to the microphone.

To remove everything without RightyO, delete the directory:
`rm -rf ~/Library/Application\ Support/RightyO/enrollment`.
