# Speaker enrollment

`rightyo enroll` builds a voiceprint for each person you want RightyO to recognise, and keeps
it only on your machine. This is step 3 of
[#137](https://github.com/mickdarling/rightyo/issues/137). `enroll verify` lets you check
locally that enrollment separates the people you enrolled. Live sessions can run
identification in **shadow mode** only (step 4a, [below](#shadow-identification-in-live-sessions)):
it logs scores and changes nothing else. Turns, roles, requests and events stay exactly as
without enrollment until a later step applies roles.

## Privacy

- RightyO ships code and synthetic test signals only. It contains no voices, voiceprints or
  enrollment data, and each user enrolls their own speakers on their own machine
  ([#109](https://github.com/mickdarling/rightyo/issues/109)).
- Audio is read from the file you name, or recorded from the microphone when you pass
  `--record`, into memory only. It is embedded and discarded, never written.
- The store holds one JSON file per enrolled speaker: the voiceprint vector and minimal
  metadata (identifier, display name, model identity, creation time, seconds of speech used,
  window count), plus an empty `.rightyo-voice-store` marker. The directory is mode 700 and
  each voiceprint mode 600; RightyO tightens looser modes only in a directory carrying the
  marker, so a mistaken `--store` never has its permissions changed. `add` creates the store,
  or adopts an empty directory, and refuses an existing directory that holds other files.
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
1.0, 0.5 to 10), `threads` (1 to 8, default 4), `live` (default false; see
[shadow identification](#shadow-identification-in-live-sessions)), `bind_min_seconds`
(default 3.0, 0.5 to 120) and `enabled` (default true). The section is off when absent or
`"enabled": false`.

The thresholds come from the evaluation's per-turn scores on synthetic voices. They are
starting points: recalibrate them on your own enrollment with `verify`. Live shadow
identification binds on a running per-label average, where non-target scores rise with more
audio, so the thresholds will be recalibrated for accumulated scores from the shadow-mode
logs ([#141](https://github.com/mickdarling/rightyo/issues/141)).

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
  store. `delete` removes only RightyO voiceprints (and RightyO's own temporary files),
  never other files; `delete --all` removes every voiceprint, then the marker and the
  directory once nothing else is left.
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

## Shadow identification in live sessions

Step 4a of #137 runs identification alongside a `listen` session, in any mode (microphone,
demo or stdin), without acting on it. Its only output is lines on `listen`'s stderr (each
prefixed `rightyo: `), so the web lab, which has no such channel, doesn't run it in shadow
mode. Turn it on with `"live": true` in the `speaker_id` section:

```json
"speaker_id": {
  "python": "/Users/you/Library/Caches/rightyo/smart-turn/venv/bin/python",
  "model": "/Users/you/Library/Caches/rightyo/wespeaker/voxceleb_resnet34_LM.onnx",
  "live": true
}
```

What it does:

- Each finalized turn with a session speaker label from the diarizer timeline (`Speaker A`)
  is scored. RightyO keeps the last 60 s of the session's audio in memory to cut the turn's
  span from; nothing is written, and the buffer is cleared when the session ends.
- The span is embedded the way `verify` does it: silence trimmed, 3 s windows, renormalised
  mean. Turns with less than `min_turn_seconds` of speech, or with overlapping speakers, are
  counted, not scored. So are turns with inferred-label words (edge attribution or a tail
  join): their audio may be another speaker's, and accumulating it could bind one voice's
  label to someone else's enrolled identity.
- Each label accumulates a duration-weighted mean of its turn embeddings, renormalised, and
  that mean is scored against every voiceprint enrolled with the same model file.
- A label is `bound` to an enrolled identifier once its accumulated score reaches
  `bind_threshold` with at least `bind_min_seconds` of speech, and then stays bound until
  that score falls below `tentative_threshold`. Otherwise it is `tentative` at or above
  `tentative_threshold`, else `unknown`.
- Embedding runs on its own thread behind a small queue, never on the audio thread. When it
  falls behind, turns are dropped and counted. If the model fails, RightyO says so once and
  turns shadow identification off for the session; the session itself carries on.
- Nothing else changes: no roles are applied, `session.started` still advertises the same
  `speakers`, and no event gains a field. Edge attribution and tail join stay available.
  To apply roles, see [roles from identification](#roles-from-identification) below.

Each scored turn adds one stderr line, for example:

```text
speaker_id label="Speaker A" turn_ms=2140 speech_ms=1880 turn_score=0.712 acc_score=0.781 acc_s=8.4 state=bound id=owner
```

| Field | Meaning |
| --- | --- |
| `label` | The session speaker label (changes every session), always in double quotes |
| `turn_ms` | The turn's span; `speech_ms` is the speech left after trimming silence |
| `turn_score` | Cosine score of this turn alone against `id` |
| `acc_score` | Cosine score of the label's accumulated mean against `id` |
| `acc_s` | Seconds of speech accumulated for the label |
| `state` | `bound`, `tentative` or `unknown` |
| `id` | The bound identifier, or else the closest enrolled one |

The session also logs `speaker_id start enrolled=N` at start, and at the end one
`speaker_id final label=…` line per label and a `speaker_id summary` line counting offered,
scored, short, overlapping, dropped, clipped and inferred turns (clipped: part of the span
was already outside the 60 s buffer; inferred: the turn had inferred-label words and was
not scored). When a session ends, turns still queued are scored if the
stream ended normally, but that is best effort: `listen` stops the session moments after
the end of input, so the last turn or two may be discarded rather than scored. A stopped or
failed session discards them.

Every line is `speaker_id`, an optional word naming the line (`start`, `final`, `summary`),
then `key=value` pairs. The label is the only value that can contain a space, and it is
always double-quoted (labels never contain quotes), so `shlex.split` parses a line into
words and each `key=value` word splits at its first `=`. The `final` line prints `-` for a
value it does not have. Other notes, such as the model being unavailable, are plain
sentences.

Lines carry labels, enrolled identifiers, durations and
scores only: never audio, embeddings, transcript text or paths. They are calibration data:
the bind and tentative thresholds will be recalibrated for accumulated scores from these
logs (#141). Like `verify` scores, keep them local and only summarise them in public
issues.

## Roles from identification

Step 4b of #137 acts on the bindings. It is off unless the `speaker_id` section sets both
`"live": true` and `"roles": true`, and the `speakers` section names enrolled identifiers:

```json
"speakers": {"owner": ["owner"]},
"speaker_id": {
  "python": "/Users/you/Library/Caches/rightyo/smart-turn/venv/bin/python",
  "model": "/Users/you/Library/Caches/rightyo/wespeaker/voxceleb_resnet34_LM.onnx",
  "live": true,
  "roles": true
}
```

- The session advertises `speakers: "enrolled"`, and each turn carries a role: `owner` or
  `trusted` once its label is bound to an identifier `speakers` names, `participant` for a
  label bound to another enrolled identifier or scored as matching no enrolled voice, and
  `unknown` before a label is scored, while it is `tentative`, or when identification is
  off. The full rules, including precedence and inferred labels, are in
  [the API notes](tool-api.md#roles-from-live-speaker-identification).
- Binding never holds up a turn. A turn's role reflects the bindings when it is emitted,
  so your first few seconds of speech in a session (until `bind_min_seconds` of it has
  been scored) are `unknown`.
- Loading refuses `"roles": true` without `"live": true`, or without an owner or trusted
  identifier in `speakers`. A diarizer with utterance-local labels is refused at start.
- With roles on, identification also runs in the web lab, which has no stderr channel; its
  lines are then simply not written.
- Edge attribution and tail join stay available: a turn with inferred-label words keeps
  its role for attention but withdraws nothing: it never overrides, cancels or withdraws
  any request, its own speaker's included, and its dismissal only stops playback.
- The optional `enrolled_follow_up_min_probability` (above 0, up to 1) lowers the
  conversation-mode follow-up bar for an engaged owner or trusted speaker.
