# Privacy and security

The local Mac lab ships explicit microphone capture with Start/Stop controls. Startup is idle; capture and hosted transcript decisions each require deliberate controls. Supplied-file replay remains available. Continuous observation must be visibly indicated, stoppable, and memory bounded. See [the prototype guide](docs/prototype.md) for retention, loopback authentication, process cancellation, and platform limits.

Default: no recording, no private transcript logging, no external inference requests. The one opt-in exception is the decision state log (`decision.state_log`, #199). It writes transcript text only to a private (`0600`, singly linked, non-symlink) file the operator names outside any Git checkout, and Clear or Stop never delete it. Dataset collection requires consent and a retention/deletion policy. Store sensitive evaluation material outside the checkout; ignore rules are defense in depth, not a substitute for inspection before commit.

Remote audio must use authenticated encrypted transport and an authorized source. A hosted inference backend requires explicit configuration of its destination and data handling. No silent fallback to hosted processing is permitted.

In privacy mode, the local tool retains at most one finalized sentence's PCM in a
mutable RAM buffer (10 s / 320,000 bytes at mono 16 kHz s16le), exclusively to check a
named exit phrase or either fragment of a split exit. The name and phrase each require their own
owner verification, including when recognized together; unlabelled private processor turns use only their isolated voice.
Oversized or incomplete sentences refuse the voice check.
The matcher immediately zeros and releases the buffer when the turn is ineligible for
an exit check. For a candidate, verification wipes the buffer on every outcome.
Replacement, privacy changes, speaker forget, Stop and session teardown also wipe it.
Private audio never enters speaker binding/accumulation, disk, logs or hosted speech.
Only fixed diagnostic tokens and the exit score number may be reported. Local ASR
assembles the current utterance in bounded memory. See [the exact retention contract](docs/tool-api.md#audio-and-logs).
Local ASR recovery also has bounded queued windows. Privacy boundaries zero owned
mutable recovery PCM, drop intersecting held text/results, and cancel private retries;
an unaccountable in-flight decoder copy requires full speaker forget.

The local Smart Turn and owner-verification workers zero their mutable private PCM,
sample, feature and model output arrays. Each worker releases its request and serialized
reply before reading again. Python cannot guarantee overwriting freed immutable data
or temporary allocations inside NumPy and ONNX Runtime.

The local Silero VAD worker checks the pinned model bytes before creating its ONNX
session. Across consecutive blocks it keeps the 64 context samples and recurrent
state required by the model. The controller receives speech labels only. A model or
worker failure disables voice privacy exits for the session.

Attention and transcript events are untrusted input, never authorization to execute privileged commands. Playback/echo must not fabricate user authority. Validate all event fields, lengths, timestamps, IDs, and bounds. Fail closed on malformed input.

Do not post secrets or private conversations to issues. Until a dedicated reporting channel is chosen, contact the repository owner privately for sensitive reports. Use sanitized public-safe summaries for tracking.
