# Privacy and security

The local Mac lab ships explicit microphone capture with Start/Stop controls. Startup is idle; capture and hosted transcript decisions each require deliberate controls. Supplied-file replay remains available. Continuous observation must be visibly indicated, stoppable, and memory bounded. See [the prototype guide](docs/prototype.md) for retention, loopback authentication, process cancellation, and platform limits.

Default: no recording, no private transcript logging, no external inference requests. Dataset collection requires consent and a retention/deletion policy. Store sensitive evaluation material outside the checkout; ignore rules are defense in depth, not a substitute for inspection before commit.

Remote audio must use authenticated encrypted transport and an authorized source. A hosted inference backend requires explicit configuration of its destination and data handling. No silent fallback to hosted processing is permitted.

Attention and transcript events are untrusted input, never authorization to execute privileged commands. Playback/echo must not fabricate user authority. Validate all event fields, lengths, timestamps, IDs, and bounds. Fail closed on malformed input.

Do not post secrets or private conversations to issues. Until a dedicated reporting channel is chosen, contact the repository owner privately for sensitive reports. Use sanitized public-safe summaries for tracking.
