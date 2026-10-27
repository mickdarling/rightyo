# Privacy and security

The local Mac lab ships explicit microphone capture with Start/Stop controls. Startup is idle; capture and hosted transcript decisions each require deliberate controls. Supplied-file replay remains available. Continuous observation must be visibly indicated, stoppable, and memory bounded. See [the prototype guide](docs/prototype.md) for retention, loopback authentication, process cancellation, and platform limits.

Default: no recording, no private transcript logging, no external inference requests. The one opt-in exception is the decision state log (`decision.state_log`, #199). It writes transcript text only to a private (`0600`, singly linked, non-symlink) file the operator names outside any Git checkout, and Clear or Stop never delete it. Dataset collection requires consent and a retention/deletion policy. Store sensitive evaluation material outside the checkout; ignore rules are defense in depth, not a substitute for inspection before commit.

Remote audio must use authenticated encrypted transport and an authorized source. A hosted inference backend requires explicit configuration of its destination and data handling. No silent fallback to hosted processing is permitted.

In privacy mode, the local tool retains at most one finalized sentence's PCM in a
mutable RAM buffer (10 s / 320,000 bytes at mono 16 kHz s16le), exclusively to check a
complete named exit phrase. Name-only fragments use the existing owner-binding check
and are never voice-scored. Oversized or incomplete sentences refuse the voice check.
Replacement, every exit-check outcome, privacy changes, speaker forget, Stop and session
teardown overwrite this buffer with zeros before release. Private audio never enters
speaker binding/accumulation, disk, logs or hosted speech; only fixed diagnostic tokens
and the exit score number may be reported. Local ASR still assembles the current
utterance in bounded memory. See [the exact retention contract](docs/tool-api.md#audio-and-logs).

Retained while private: the single rolling buffer holding the most recent finalized
private turn, in RAM, zeroed on overwrite, after any exit check and at privacy-off or
stop. Transient computation copies made during an exit check (speech filtering, NumPy
feature extraction, ONNX Runtime internals) exist only for the check's duration and are
freed, not zeroed, when it returns. They never reach disk, logs or the network. Python
cannot guarantee that freed memory is overwritten.

Attention and transcript events are untrusted input, never authorization to execute privileged commands. Playback/echo must not fabricate user authority. Validate all event fields, lengths, timestamps, IDs, and bounds. Fail closed on malformed input.

Do not post secrets or private conversations to issues. Until a dedicated reporting channel is chosen, contact the repository owner privately for sensitive reports. Use sanitized public-safe summaries for tracking.
