"""Bounded host playback evidence on the written-sample clock.

Self classification may only remove evidence. Loss, uncertainty, rounding,
eviction or error must push a turn toward the person path, never toward self.
Mixed evidence is never discarded while echo-only evidence over the same time remains.
"""

from collections import deque
from dataclasses import asdict, dataclass

MAX_WINDOWS = 8
MAX_SPANS = 512
OPEN_MS = 120_000
LATE_MS = 5_000
AHEAD_MS = 35_000
MAX_COUNT = 2**31 - 1
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
    from_sample: int
    eligible_from: int

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
        self.mixed_lost_until = 0
        self.empty_closes = deque(maxlen=MAX_WINDOWS)
        self.kept = {"echo_only": 0, "mixed": 0}
        self.dropped = {"echo_only": 0, "mixed": 0}
        self.windows_dropped = 0
        self.published = {}
        self.self_count = 0
        self.self_starts = deque(maxlen=MAX_SPANS)

    def tag_summary(self):
        return (
            f"tags_echo={self.kept['echo_only']}/{self.dropped['echo_only']}"
            f" tags_mixed={self.kept['mixed']}/{self.dropped['mixed']}"
            f" windows_dropped={self.windows_dropped}"
        )

    def _discard(self, removed):
        if not removed:
            return
        mixed = [span for span in removed if span.kind == "mixed"]
        for span in mixed:
            self.mixed_lost_until = max(self.mixed_lost_until, span.end)
        self.spans = [
            span
            for span in self.spans
            if span not in removed
            and not (
                span.kind == "echo_only"
                and any(span.start < m.end and m.start < span.end for m in mixed)
            )
        ]

    def advance(self, received):
        cutoff = received - HISTORY_MS
        self._discard({span for span in self.spans if span.end < cutoff})
        while self.self_starts and self.self_starts[0] < cutoff:
            self.self_starts.popleft()

    def _remove_window(self, reply):
        del self.windows[reply]
        self._discard({s for s in self.spans if s.reply == reply and s.kind == "echo_only"})

    def reply_window(self, reply, start, end, tail, received, *, from_sample=None):
        self.advance(received)
        if from_sample is None:
            from_sample = start * 16
        if end is not None and end > received + AHEAD_MS:
            self.windows_dropped = min(MAX_COUNT, self.windows_dropped + 1)
            return
        if reply in self.empty_closes:
            return
        window = self.windows.get(reply)
        if window is not None and (
            window.end is not None
            or from_sample != window.from_sample
            or (end is None and tail != window.tail)
        ):
            return
        if end is not None and end + tail <= start:
            # Remember the explicit close without retaining an empty region.
            self.empty_closes.append(reply)
            if window is not None:
                self._remove_window(reply)
            self.windows_dropped = min(MAX_COUNT, self.windows_dropped + 1)
            return
        if window is not None:
            if end is None:
                if received >= window.expires:
                    window.eligible_from = max(window.start, received)
                window.expires = received + OPEN_MS
            else:
                window.end, window.tail = end, tail
                # A measured close retracts positive evidence only.
                self.spans = [
                    Span(s.start, min(s.end, end + tail), reply, s.kind)
                    if s.reply == reply and s.kind == "echo_only"
                    else s
                    for s in self.spans
                    if s.reply != reply or s.kind == "mixed" or s.start < end + tail
                ]
            return
        if len(self.windows) == MAX_WINDOWS:
            self._remove_window(next(iter(self.windows)))
        self.windows[reply] = Window(
            start, end, received + OPEN_MS, tail, from_sample, eligible_from=start
        )

    def self_echo(self, reply, start, end, kind, received):
        merged = self._insert(reply, start, end, kind, received)
        self.advance(received)
        self._discard(set(self.spans[:-MAX_SPANS]))
        counts = self.kept if merged is not None and merged in self.spans else self.dropped
        counts[kind] = min(MAX_COUNT, counts[kind] + 1)

    def _insert(self, reply, start, end, kind, received):
        if kind != "mixed":
            window = self.windows.get(reply)
            if window is None:
                return None
            # The host may have written samples the processor has yet to consume.
            limit = (window.expires if window.end is None else window.end) + window.tail
            if not (
                window.eligible_from <= start < end <= limit
                and received - LATE_MS <= end <= received + AHEAD_MS
            ):
                return None
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
        merged = Span(start, end, reply, kind)
        kept.append(merged)
        self.spans = sorted(kept, key=lambda span: span.start)
        return merged

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

    def overlaps_window(self, start, end, received):
        return any(w.start < end and start < w.region_end(received) for w in self.windows.values())

    def contains(self, start, end, received):
        """Read-only window eligibility at turn acceptance."""
        return start < end and any(
            w.eligible_from <= start and end <= w.region_end(received)
            for w in self.windows.values()
        )

    def classify(self, start, end, received):
        if end <= start:
            return "outside", 0.0
        mixed, covered = self.coverage(start, end)
        fraction = covered / (end - start)
        if start < self.mixed_lost_until:
            return "partial", fraction
        if not self.contains(start, end, received):
            return "outside", fraction
        if mixed:
            return "mixed", fraction
        if fraction < self.config.echo_only_min_fraction:
            return "partial", fraction
        return "echo_only", fraction

    def record_self(self, start):
        self.self_count = min(MAX_COUNT, self.self_count + 1)
        self.self_starts.append(start)

    def excludes_answer(self, at):
        return at in self.self_starts
