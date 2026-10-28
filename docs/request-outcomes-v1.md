# Host request outcomes v1 (#262)

This provider-neutral contract is paired with Hailing Station #1002. It is an additive
trusted control-FD extension; it does not authorize capture, application actions or
provider control. Only ordinary `listen` advertises `"request_outcomes":{"version":1}`
on `session started`; coaching and legacy children do not. Hosts must negotiate that
exact version before sending it. Unsupported children remain explicitly unavailable.

A control is one UTF-8 JSON object followed by newline, at most 512 bytes excluding the
newline. Duplicate keys, extra fields, unknown versions/states and malformed IDs refuse.
Session and utterance identifiers use the existing 1–96 ASCII-scalar identifier grammar.
The request ID is exactly `session_id:utterance_id`, from a request this producer emitted.

```json
{"request_outcome":{"version":1,"session_id":"authored-session","request_id":"authored-session:turn-1","state":"accepted"}}
```

States are `accepted`, `failed`, `timed_out`, `cancelled`, `reply_started`, `reply_ended`.
An issued request begins pending; a real scoped reply requires prior `accepted`.
Only `reply_ended` may include optional boolean `asked`. A terminal failure/cancellation
may end pending, accepted or playing work. `reply_ended` ends reply reporting; terminal
outcomes cannot be reopened. An identical duplicate is acknowledged without extending
windows. Backwards, unknown, foreign, retired and expired identities emit no ack.

After applying a valid transition, the producer emits a normal sequenced event with
`type: request_outcome`, `version: 1`, and the exact `request_id` and `state`. Pipe write
admission is not this acknowledgement. A host must correlate the exact child session,
request and expected state before calling a transition accepted. A producer backlog
failure has the existing terminal semantics; no missing ack is treated as success.

Terminal failures clear only matching pending acknowledgement, open-request and
engagement state, and cannot revive dormant engagement. If that request held the
engagement, a `conversation` event returns to `ambient`, reason `request_ended`.
A newer request's engagement or acknowledgement is preserved. Privacy entry, speaker
forget, explicit withdrawal/override and session retirement revoke old identities.
The existing transcript retention cutoff also retires their control authority. The bounded map retains at most
128 request identities; an evicted identity has no authority. No transcript is added.

On the first valid lifecycle control, old unscoped semantic reply/asked/hold authority
is retired. Known pre-opt-in physical spans can still close at their actual end without
closing a competing scoped reply or creating new reply authority.

After accepting scoped controls, unscoped playback reports still record echo spans,
but cannot answer, extend or revive a conversation. An acknowledgement clip is not an
answer. Negotiating hosts must use scoped reply states for actual answer playback;
a late original answer to a terminated request cannot affect newer state. Legacy hosts
that never send `accepted` keep their previous behavior.

This PR implements the RightyO side only. Hailing Station must supply negotiation,
exact request/generation matching, acknowledgement deadlines, supervision and routed
notices in its own reviewed PR. It must not translate an unconfirmed provider interrupt
into `timed_out`, replay the old request, or claim playback occurred. Contract fixtures
and tests are authored; there is no live-device or speech-quality proof here.

Run `PYTHONPATH=src:tests python3 -m unittest tests/test_request_outcomes.py` and the
normal `scripts/verify.py`. The control-FD test uses a real pipe with authored records.
