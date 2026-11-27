"""Offline CPU cost of #334. Run with the provisioned model's Python, no enrollment needed.

PYTHONPATH=src python3 scripts/measure_unlabelled_voiceprint.py --model /path/to/model.onnx
"""

import argparse
import math
import sys
import time
from array import array
from pathlib import Path

from rightyo.enroll import normalize
from rightyo.live_speaker_id import _isolated_owner_score
from rightyo.speaker_id import SAMPLE_RATE, _cpu_session, _embedding, model_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="existing local WeSpeaker ResNet34-LM ONNX")
    parser.add_argument("--threads", type=int, choices=range(1, 9), default=4)
    args = parser.parse_args()
    if args.model is None or not args.model.is_file():
        print("SKIP (no local model supplied)")
        return
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        print("SKIP (numpy and onnxruntime are required in the model interpreter)")
        return
    model_identity(args.model)
    session = _cpu_session(ort, str(args.model), args.threads)
    name = session.get_inputs()[0].name

    def embed(pcm):
        with _embedding(np, session, name, pcm, wipe_input=True) as vector:
            return vector.copy()

    # Twelve consecutive triplets of equally long turns. Only the third incurs
    # diagnostic cost. Each half-second step from 0.5 to 6 s occurs once.
    tone = array("h", (int(6000 * math.sin(2 * math.pi * i / 80)) for i in range(80)))
    if sys.byteorder == "big":
        tone.byteswap()
    signals = [bytearray(tone) * (size * SAMPLE_RATE // (2 * len(tone))) for size in range(1, 13)]
    audio_seconds = 3 * sum(len(pcm) for pcm in signals) / (2 * SAMPLE_RATE)
    # Warm the actual model and derive a compatible synthetic reference before timing.
    owners = [normalize(embed(memoryview(bytearray(signals[0]))))]
    timings = []
    cpu = time.process_time()
    for pcm in signals:
        start = time.perf_counter()
        _, ms, reason = _isolated_owner_score(pcm, embed, owners, 0, lambda: False)
        if reason is not None:
            raise RuntimeError("Synthetic turn could not be scored")
        timings.append((ms, time.perf_counter() - start))
    cpu = time.process_time() - cpu
    elapsed = sum(seconds for _, seconds in timings)
    for ms, seconds in timings:
        print(f"speech_ms={ms} embedding_ms={seconds * 1000:.2f}")
    print(
        f"turns=36 unlabelled=12 mean_ms_per_unlabelled={1000 * elapsed / 12:.2f} "
        f"mean_ms_per_turn={1000 * elapsed / 36:.2f} "
        f"cpu_share_one_core={100 * cpu / audio_seconds:.3f}% "
        f"(CPU seconds / {audio_seconds:g} seconds of synthetic stream; IPC excluded)"
    )


if __name__ == "__main__":
    main()
