"""Bounded host playback evidence on the written-sample clock."""

from collections import deque
from dataclasses import asdict, dataclass

MAX_WINDOWS = 8
MAX_SPANS = 512
OPEN_MS = 120_000
LATE_MS = 5_000
HISTORY_MS = 60_000 + LATE_MS


@dataclass(frozen=True)
class SelfVoiceConfig:
    enabled: bool = False
    transcripts: str = "suppress"
    echo_only_min_fraction: float = 0.8

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("invalid self_voice section")
        config = cls(**value)
        if (
            type(config.enabled) is not bool
            or config.transcripts not in ("suppress", "publish")
            or type(config.echo_only_min_fraction) not in (float, int)
            or not 0.5 <= config.echo_only_min_fraction <= 1
        ):
            raise ValueError("invalid self_voice setting")
        return config

    def to_config(self):
        defaults = asdict(SelfVoiceConfig())
        changed = {k: v for k, v in asdict(self).items() if v != defaults[k]}
        return {"self_voice": changed} if changed else {}


@dataclass
class Window:
    start: int
    end: int | None
    expires: int
    tail: int

    def region_end(self, received):
        return (min(received, self.expires) if self.end is None else self.end) + self.tail


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    reply: str
    kind: str


class SelfVoice:
    """One session. All accesses are serialized by the controller."""

    def __init__(self, config):
        self.config = config
        self.windows = {}
        self.spans = []
        self.received = 0
        self.self_count = 0
        self.self_starts = deque(maxlen=MAX_SPANS)

    def advance(self, received):
        self.received = received
        for window in self.windows.values():
            if window.end is None and received >= window.expires:
                window.end = window.expires
        cutoff = received - HISTORY_MS
        self.spans = [span for span in self.spans if span.end >= cutoff]
        while self.self_starts and self.self_starts[0] < cutoff:
            self.self_starts.popleft()

    def reply_window(self, reply, start, end, tail, received):
        self.advance(received)
        window = self.windows.get(reply)
        if window is not None:
            if (
                window.end is not None
                or start != window.start
                or (end is None and tail != window.tail)
            ):
                return
            if end is None:
                window.expires = max(received, start) + OPEN_MS
            window.end, window.tail = end, tail
            if end is not None:
                # A measured tail can retract part of the nominal region.
                self.spans = [
                    Span(s.start, min(s.end, end + tail), reply, s.kind) if s.reply == reply else s
                    for s in self.spans
                    if s.reply != reply or s.start < end + tail
                ]
            return
        if len(self.windows) == MAX_WINDOWS:
            oldest = next(iter(self.windows))
            del self.windows[oldest]
            self.spans = [s for s in self.spans if s.reply != oldest]
        self.windows[reply] = Window(start, end, start + OPEN_MS, tail)

    def self_echo(self, reply, start, end, kind, received, *, partial_end=False):
        self.advance(received)
        if end < received - LATE_MS:
            return
        window = self.windows.get(reply)
        if window is None or not (window.start <= start and end <= window.region_end(received)):
            return
        if kind == "mixed" and partial_end:
            # Include the last touched millisecond, bounded by the reply region.
            end = min(end + 1, window.region_end(received))
        if start >= end:
            return
        kept = []
        for span in self.spans:
            if (
                span.reply == reply
                and span.kind == kind
                and start <= span.end
                and span.start <= end
            ):
                start, end = min(start, span.start), max(end, span.end)
            else:
                kept.append(span)
        kept.append(Span(start, end, reply, kind))
        self.spans = sorted(kept, key=lambda span: span.start)[-MAX_SPANS:]

    def coverage(self, start, end):
        """Union of echo-only coverage, with any mixed overlap taking precedence."""
        covered = 0
        cursor = start
        mixed = False
        for span in self.spans:
            left, right = max(start, span.start), min(end, span.end)
            if left >= right:
                continue
            if span.kind == "mixed":
                mixed = True
            else:
                covered += max(0, right - max(cursor, left))
                cursor = max(cursor, right)
        return mixed, covered

    def overlaps_window(self, start, end):
        return any(
            w.start < end and start < w.region_end(self.received) for w in self.windows.values()
        )

    def classify(self, start, end, received):
        self.advance(received)
        if end <= start:
            return "outside", 0.0
        mixed, covered = self.coverage(start, end)
        fraction = covered / (end - start)
        if not any(
            w.start <= start and end <= w.region_end(received) for w in self.windows.values()
        ):
            return "outside", fraction
        if mixed:
            return "mixed", fraction
        if fraction < self.config.echo_only_min_fraction:
            return "partial", fraction
        self.self_count += 1
        self.self_starts.append(start)
        return "echo_only", fraction

    def excludes_answer(self, at):
        mixed, covered = self.coverage(at, at + 1)
        return at in self.self_starts or (bool(covered) and not mixed)
