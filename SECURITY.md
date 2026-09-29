# Privacy and security

No capture service is shipped. The MVP invokes explicitly supplied local model runtimes on supplied files and offers opt-in hosted transcript decisions. Future continuous observation must be explicitly enabled, visibly indicated, immediately stoppable, and memory bounded.

Default: no recording, no private transcript logging, no external inference requests. Dataset collection requires consent and a retention/deletion policy. Store sensitive evaluation material outside the checkout; ignore rules are defense in depth, not a substitute for inspection before commit.

Remote audio must use authenticated encrypted transport and an authorized source. A hosted inference backend requires explicit configuration of its destination and data handling. No silent fallback to hosted processing is permitted.

Attention and transcript events are untrusted input, never authorization to execute privileged commands. Playback/echo must not fabricate user authority. Validate all event fields, lengths, timestamps, IDs, and bounds. Fail closed on malformed input.

Do not post secrets or private conversations to issues. Until a dedicated reporting channel is chosen, contact the repository owner privately for sensitive reports. Use sanitized public-safe summaries for tracking.
