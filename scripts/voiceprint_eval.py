"""Offline voiceprint (speaker-embedding) spike harness for #137, synthetic voices only.

Two explicit steps, both writing only to a directory you name outside any Git checkout:

    generate   render authored text with built-in macOS `say` voices (no microphone, no
               real person's voice, no download) into 16 kHz mono WAV plus a manifest.
    evaluate   enroll each synthetic speaker from ~30 s, score held-out clips of 0.5/1/2/4 s
               (clean, and with synthetic reverb plus pink noise) against every voiceprint,
               and write aggregate metrics only: EER, thresholds, latency, RSS.

Embeddings and audio stay in memory or in the named scratch directory; metrics files hold
no audio, no embeddings, no text and no local paths. The harness never provisions models:
pass explicitly downloaded files. Backends import their runtime lazily, so run each under
an interpreter that has it:

    speakerkit   coremltools + numpy; the SpeakerKit `speaker_embedder/pyannote-v3/W8A16`
                 directory (SpeakerEmbedderPreprocessor + SpeakerEmbedder .mlmodelc).
    wespeaker    onnxruntime + numpy; a WeSpeaker VoxCeleb `.onnx` (ResNet34-LM, CAM++, ...).
    speechbrain  torch + speechbrain; a local copy of `speechbrain/spkrec-ecapa-voxceleb`.

Synthetic TTS voices are far easier to separate than real voices: these numbers bound
mechanics (short-turn behaviour, latency, memory), not real-world accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
import wave
from pathlib import Path

SAMPLE_RATE = 16000
ENROLL_SECONDS = 30.0
DURATIONS = (0.5, 1.0, 2.0, 4.0)
TEST_RATES = (None, 150, 220)  # words per minute; None = the voice's default rate

# Built-in macOS voices. Enrolled speakers, then unenrolled impostors (open-set checks).
# "Reed (English (UK))" is the same voice family as the enrolled US Reed: a near-twin.
ENROLLED = (
    "Samantha",
    "Daniel",
    "Karen",
    "Moira",
    "Rishi",
    "Tessa",
    "Fred",
    "Kathy",
    "Ralph",
    "Reed (English (US))",
)
TWIN = "Reed (English (UK))"
IMPOSTORS = (TWIN, "Shelley (English (US))", "Flo (English (US))", "Albert")

ENROLL_TEXT = (
    "Good morning. I am setting up the kitchen assistant so that it knows my voice. "
    "Yesterday we cooked a large pot of vegetable soup and froze half of it for next week. "
    "The weather turned cold in the afternoon, so I closed the windows and lit the stove. "
    "Please remind me to call the plumber about the slow drain in the upstairs bathroom. "
    "On Saturdays I usually walk to the market, buy fresh bread, apples and a little cheese, "
    "and then read the newspaper on the porch while the coffee is still warm. "
    "My favourite season is autumn, when the leaves change colour along the river path."
)
TEST_TEXT = (
    "Could you turn the living room lights down a little? "
    "I think the train leaves at a quarter past six, but check the timetable to be sure. "
    "Remind me tomorrow to water the tomatoes and move the bicycle out of the garage. "
    "What was the name of that restaurant we liked near the harbour last summer? "
    "Set a timer for twelve minutes, and tell me when the oven has finished preheating. "
    "No, not that playlist. Play something quieter, maybe piano or soft jazz. "
    "The meeting moved to Thursday afternoon, so add it to the shared calendar please. "
    "How far is it to the airport if we leave before the morning traffic starts? "
    "Honestly, I would rather stay home tonight and finish the puzzle on the table. "
    "Ask whether the hardware store still has the blue paint we ordered last month. "
    "Okay, that sounds fine. Let us talk about it again after dinner."
)


# ----------------------------------------------------------------------------- audio io


def _inside_git_checkout(path: Path) -> bool:
    for parent in (path, *path.parents):
        if (parent / ".git").exists():
            return True
    return False


def _read_wav(path: Path):
    import numpy as np

    with wave.open(str(path), "rb") as handle:
        if (handle.getnchannels(), handle.getsampwidth(), handle.getframerate()) != (
            1,
            2,
            SAMPLE_RATE,
        ):
            raise SystemExit("expected 16 kHz mono PCM16 WAV")
        data = handle.readframes(handle.getnframes())
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def _speech_mask(np, x, frame=160, floor_db=-35.0):
    """Per-10 ms speech flags from frame energy relative to the loudest frame."""
    n = len(x) // frame
    energy = (x[: n * frame].reshape(n, frame) ** 2).mean(axis=1) + 1e-12
    db = 10 * np.log10(energy)
    return db > db.max() + floor_db


def _trim(np, x, max_gap_s=0.25):
    """Trim leading/trailing silence and shorten internal pauses to at most max_gap_s."""
    mask = _speech_mask(np, x)
    keep = np.zeros(len(mask), dtype=bool)
    gap_frames = int(max_gap_s * 100)
    run = 0
    started = False
    for i, voiced in enumerate(mask):
        if voiced:
            started = True
            run = 0
            keep[i] = True
        elif started:
            run += 1
            keep[i] = run <= gap_frames
    last = np.nonzero(mask)[0]
    if len(last):
        keep[last[-1] + 1 :] = False
    frames = x[: len(mask) * 160].reshape(len(mask), 160)
    return frames[keep].reshape(-1)


# ----------------------------------------------------------------------------- generate


def generate(out: Path) -> None:
    out = out.resolve()
    if _inside_git_checkout(out):
        raise SystemExit("refusing to write audio inside a Git checkout")
    if sys.platform != "darwin":
        raise SystemExit("generation uses the built-in macOS `say` command")
    out.mkdir(parents=True, exist_ok=True)
    os.chmod(out, 0o700)
    manifest = {"sample_rate": SAMPLE_RATE, "speakers": []}
    for index, voice in enumerate(ENROLLED + IMPOSTORS):
        role = "enrolled" if voice in ENROLLED else "impostor"
        files = {}
        jobs = [("enroll", None, ENROLL_TEXT)] if role == "enrolled" else []
        jobs += [(f"test-{rate or 'default'}", rate, TEST_TEXT) for rate in TEST_RATES]
        for name, rate, text in jobs:
            target = out / f"spk{index:02d}-{name}.wav"
            command = ["say", "-v", voice, "-o", str(target)]
            command += ["--file-format=WAVE", f"--data-format=LEI16@{SAMPLE_RATE}"]
            if rate:
                command += ["-r", str(rate)]
            subprocess.run(command + ["--", text], check=True, timeout=120)
            files[name] = {
                "file": target.name,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        manifest["speakers"].append(
            {"id": f"spk{index:02d}", "voice": voice, "role": role, "files": files}
        )
        print(f"rendered {voice} ({role})", flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


# ----------------------------------------------------------------------------- clips


def _pink_noise(np, rng, n):
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n)
    freqs[0] = freqs[1]
    noise = np.fft.irfft(spectrum / np.sqrt(freqs), n)
    return noise / (noise.std() + 1e-12)


def _room(np, rng, x, snr_db=10.0, rt60=0.4):
    """Synthetic reverb (exponentially decaying noise tail) plus pink noise at snr_db."""
    length = int(rt60 * SAMPLE_RATE)
    t = np.arange(length) / SAMPLE_RATE
    ir = rng.standard_normal(length) * np.exp(-6.9 * t / rt60) * 0.3
    ir[0] = 1.0
    size = 1 << int(np.ceil(np.log2(len(x) + length)))
    wet = np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(ir, size), size)[: len(x)]
    wet *= np.sqrt((x**2).mean() / ((wet**2).mean() + 1e-12))
    noise = _pink_noise(np, rng, len(x))
    noise *= np.sqrt((wet**2).mean() / 10 ** (snr_db / 10))
    mixed = wet + noise
    return (mixed / max(1.0, np.abs(mixed).max())).astype(np.float32)


def _clips(np, rng, audio, seconds, count, min_speech=0.7):
    """Non-overlapping windows from trimmed held-out speech, mostly voiced."""
    length = int(seconds * SAMPLE_RATE)
    mask = _speech_mask(np, audio)
    starts = []
    candidates = rng.permutation(max(1, (len(audio) - length) // 160))
    taken = np.zeros(len(audio), dtype=bool)
    for frame in candidates:
        start = int(frame) * 160
        if taken[start : start + length].any():
            continue
        window = mask[frame : frame + length // 160]
        if len(window) and window.mean() >= min_speech:
            starts.append(start)
            taken[start : start + length] = True
        if len(starts) == count:
            break
    return [audio[s : s + length] for s in sorted(starts)]


def build_dataset(np, data: Path, clips_per_duration: int, seed: int):
    manifest = json.loads((data / "manifest.json").read_text())
    rng = np.random.default_rng(seed)
    enroll, tests = {}, []
    for speaker in manifest["speakers"]:
        files = speaker["files"]
        for entry in files.values():
            if hashlib.sha256((data / entry["file"]).read_bytes()).hexdigest() != entry["sha256"]:
                raise SystemExit("input file hash mismatch")
        if speaker["role"] == "enrolled":
            x = _trim(np, _read_wav(data / files["enroll"]["file"]))
            enroll[speaker["id"]] = x[: int(ENROLL_SECONDS * SAMPLE_RATE)]
        held_out = np.concatenate(
            [_trim(np, _read_wav(data / v["file"])) for k, v in files.items() if k != "enroll"]
        )
        for seconds in DURATIONS:
            for clip in _clips(np, rng, held_out, seconds, clips_per_duration):
                for condition in ("clean", "room"):
                    audio = clip if condition == "clean" else _room(np, rng, clip)
                    tests.append(
                        {
                            "speaker": speaker["id"],
                            "role": speaker["role"],
                            "seconds": seconds,
                            "condition": condition,
                            "audio": audio,
                        }
                    )
    names = {s["id"]: s["voice"] for s in manifest["speakers"]}
    return enroll, tests, names


# ----------------------------------------------------------------------------- backends


def _kaldi_fbank(np, x, bins=80):
    """Kaldi-compatible fbank (25/10 ms, Hamming, pre-emphasis 0.97, no dither), with CMN.

    Matches the WeSpeaker reference front end: int16-scaled samples, snip_edges, log mel
    energies with low_freq 20 Hz and high_freq at Nyquist, then per-utterance mean removal.
    """
    x = x.astype(np.float64) * 32768.0
    frame_len, shift, n_fft = 400, 160, 512
    if len(x) < frame_len:
        x = np.pad(x, (0, frame_len - len(x)))
    count = 1 + (len(x) - frame_len) // shift
    index = np.arange(frame_len)[None, :] + shift * np.arange(count)[:, None]
    frames = x[index]
    frames = frames - frames.mean(axis=1, keepdims=True)
    frames[:, 1:] -= 0.97 * frames[:, :-1].copy()
    frames[:, 0] -= 0.97 * frames[:, 0]
    window = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(frame_len) / (frame_len - 1))
    power = np.abs(np.fft.rfft(frames * window, n=n_fft)) ** 2

    def mel(f):
        return 1127.0 * np.log(1.0 + np.asarray(f) / 700.0)

    low, high = mel(20.0), mel(SAMPLE_RATE / 2)
    centers = np.linspace(low, high, bins + 2)
    fft_mel = mel(np.arange(n_fft // 2) * SAMPLE_RATE / n_fft)
    banks = np.zeros((n_fft // 2 + 1, bins))
    for b in range(bins):
        left, center, right = centers[b : b + 3]
        up = (fft_mel - left) / (center - left)
        down = (right - fft_mel) / (right - center)
        banks[: n_fft // 2, b] = np.maximum(0.0, np.minimum(up, down))
    feats = np.log(np.maximum(power @ banks, np.finfo(np.float32).eps))
    return (feats - feats.mean(axis=0)).astype(np.float32)


class WeSpeaker:
    def __init__(self, model: Path, threads: int):
        import numpy as np
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.np = np
        self.session = ort.InferenceSession(
            str(model), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input = self.session.get_inputs()[0].name
        self.size_bytes = model.stat().st_size

    def embed(self, x):
        feats = _kaldi_fbank(self.np, x)[None]
        return self.session.run(None, {self.input: feats})[0][0]


class SpeakerKit:
    """Drive the SpeakerKit pyannote-v3 embedder directly with one whole-clip mask.

    The Core ML pair takes a fixed 30 s window (480,000 samples) and 64 speaker masks over
    1,767 segmentation frames. A short clip is tiled to fill the window (the model applies
    feature normalisation over the whole window, so zero padding would distort short
    clips) and slot 0's mask covers every frame; the other 63 slots are zero.
    """

    WINDOW = 480000
    FRAMES = 1767

    def __init__(self, model_dir: Path, compute_units: str, pad: str):
        import coremltools as ct
        import numpy as np

        units = {
            "all": ct.ComputeUnit.ALL,
            "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
            "cpu_only": ct.ComputeUnit.CPU_ONLY,
        }[compute_units]
        self.np = np
        self.pad = pad
        self.pre = ct.models.CompiledMLModel(
            str(model_dir / "SpeakerEmbedderPreprocessor.mlmodelc"), compute_units=units
        )
        self.model = ct.models.CompiledMLModel(
            str(model_dir / "SpeakerEmbedder.mlmodelc"), compute_units=units
        )
        self.size_bytes = sum(p.stat().st_size for p in model_dir.rglob("*") if p.is_file())

    def embed(self, x):
        np = self.np
        n = min(len(x), self.WINDOW)
        if self.pad == "tile":
            window = np.resize(x[:n], self.WINDOW)
            active = self.FRAMES
        else:
            window = np.zeros(self.WINDOW, dtype=np.float32)
            window[:n] = x[:n]
            active = max(1, int(round(self.FRAMES * n / self.WINDOW)))
        masks = np.zeros((1, 64, self.FRAMES), dtype=np.float16)
        masks[0, 0, :active] = 1.0
        feats = self.pre.predict({"waveforms": window[None].astype(np.float16)})
        feats = feats["preprocessor_output_1"].astype(np.float16)
        out = self.model.predict({"preprocessor_output_1": feats, "speaker_masks": masks})
        return out["speaker_embeddings"][0, 0].astype(np.float32)


class SpeechBrainEcapa:
    def __init__(self, model_dir: Path, threads: int):
        os.environ["HF_HUB_OFFLINE"] = "1"  # local files only; never fetch
        import numpy as np
        import torch
        from speechbrain.inference.speaker import EncoderClassifier

        torch.set_num_threads(threads)
        self.np, self.torch = np, torch
        self.model = EncoderClassifier.from_hparams(
            source=str(model_dir),
            savedir=str(model_dir),
            overrides={"pretrained_path": str(model_dir)},
            run_opts={"device": "cpu"},
        )
        self.model.eval()
        used = ("embedding_model.ckpt", "mean_var_norm_emb.ckpt")
        self.size_bytes = sum((model_dir / name).stat().st_size for name in used)

    def embed(self, x):
        with self.torch.inference_mode():
            out = self.model.encode_batch(self.torch.from_numpy(x[None].copy()))
        return out[0, 0].numpy().astype(self.np.float32)


# ----------------------------------------------------------------------------- metrics


def _normalize(np, v):
    return v / (np.linalg.norm(v) + 1e-12)


def _eer(np, target, nontarget):
    scores = np.concatenate([target, nontarget])
    labels = np.concatenate([np.ones(len(target)), np.zeros(len(nontarget))])
    order = np.argsort(scores)
    scores, labels = scores[order], labels[order]
    # Threshold just above each score: reject everything at or below it.
    frr = np.cumsum(labels) / max(1, len(target))
    far = 1 - np.cumsum(1 - labels) / max(1, len(nontarget))
    i = int(np.argmin(np.abs(frr - far)))
    return float((frr[i] + far[i]) / 2), float(scores[i])


def _threshold_at_far(np, nontarget, far):
    return float(np.quantile(nontarget, 1 - far, method="higher"))


def _summary(np, values):
    values = np.asarray(values)
    return {
        "mean": round(float(values.mean()), 4),
        "std": round(float(values.std()), 4),
        "p05": round(float(np.quantile(values, 0.05)), 4),
        "p95": round(float(np.quantile(values, 0.95)), 4),
        "min": round(float(values.min()), 4),
        "max": round(float(values.max()), 4),
    }


def _load_backend(args):
    start = time.perf_counter()
    if args.backend == "wespeaker":
        backend = WeSpeaker(args.model, args.threads)
    elif args.backend == "speakerkit":
        backend = SpeakerKit(args.model, args.compute_units, args.pad)
    else:
        backend = SpeechBrainEcapa(args.model, args.threads)
    return backend, time.perf_counter() - start


def _maxrss_mib():
    scale = 1 if sys.platform == "darwin" else 1024  # ru_maxrss: bytes on macOS, KiB on Linux
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale / 2**20, 1)


def footprint(args) -> dict:
    """Fresh-process model cost: load time, peak RSS, warm per-clip latency (one speaker)."""
    import numpy as np

    manifest = json.loads((args.data / "manifest.json").read_text())
    first = next(s for s in manifest["speakers"] if s["role"] == "enrolled")
    audio = _trim(np, _read_wav(args.data / first["files"]["enroll"]["file"]))
    rss_imports = _maxrss_mib()
    backend, load_seconds = _load_backend(args)
    rss_loaded = _maxrss_mib()
    timings = {}
    for seconds in (*DURATIONS, ENROLL_SECONDS):
        clip = audio[: int(seconds * SAMPLE_RATE)]
        for _ in range(3):
            backend.embed(clip)
        runs = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            backend.embed(clip)
            runs.append(time.perf_counter() - start)
        timings[str(seconds)] = round(statistics.median(runs) * 1000, 2)
    return {
        "backend": args.backend,
        "model_label": args.label,
        "compute_units": args.compute_units if args.backend == "speakerkit" else "cpu",
        "threads": args.threads,
        "model_bytes": backend.size_bytes,
        "load_seconds": round(load_seconds, 3),
        "rss_after_numpy_mib": rss_imports,
        "rss_after_load_mib": rss_loaded,
        "peak_rss_mib": _maxrss_mib(),
        "median_latency_ms": timings,
        "repeats": args.repeats,
    }


def evaluate(args) -> dict:
    import numpy as np

    rss_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    enroll_audio, tests, names = build_dataset(
        np, args.data.resolve(), args.clips_per_duration, args.seed
    )
    rss_data = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    backend, load_seconds = _load_backend(args)

    first = next(iter(enroll_audio.values()))
    for _ in range(3):
        backend.embed(first[: SAMPLE_RATE * 2])

    speakers = sorted(enroll_audio)
    voiceprints = np.stack([_normalize(np, backend.embed(enroll_audio[s])) for s in speakers])
    latency = {}
    for test in tests:
        start = time.perf_counter()
        vector = backend.embed(test["audio"])
        latency.setdefault(test["seconds"], []).append(time.perf_counter() - start)
        test["scores"] = voiceprints @ _normalize(np, vector)
    rss_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1 if sys.platform == "darwin" else 1024  # ru_maxrss: bytes on macOS, KiB on Linux

    results = []
    for condition in ("clean", "room"):
        for seconds in DURATIONS:
            group = [t for t in tests if t["condition"] == condition and t["seconds"] == seconds]
            enrolled = [t for t in group if t["role"] == "enrolled"]
            target, nontarget, correct = [], [], 0
            for t in enrolled:
                own = speakers.index(t["speaker"])
                target.append(t["scores"][own])
                nontarget.extend(np.delete(t["scores"], own))
                correct += int(np.argmax(t["scores"]) == own)
            target, nontarget = np.array(target), np.array(nontarget)
            # Open-set: best score of unenrolled voices, excluding the near-twin (reported apart).
            impostor = np.array(
                [
                    t["scores"].max()
                    for t in group
                    if t["role"] == "impostor" and names[t["speaker"]] != TWIN
                ]
            )
            eer, eer_threshold = _eer(np, target, nontarget)
            far1 = _threshold_at_far(np, nontarget, 0.01)
            results.append(
                {
                    "condition": condition,
                    "seconds": seconds,
                    "target_trials": int(len(target)),
                    "nontarget_trials": int(len(nontarget)),
                    "target": _summary(np, target),
                    "nontarget": _summary(np, nontarget),
                    "eer": round(eer, 4),
                    "eer_threshold": round(eer_threshold, 4),
                    "far1_threshold": round(far1, 4),
                    "frr_at_far1": round(float((target < far1).mean()), 4),
                    "top1_accuracy": round(correct / max(1, len(enrolled)), 4),
                    "unenrolled_best_score": _summary(np, impostor),
                    "_target": target,
                    "_nontarget": nontarget,
                    "_impostor": impostor,
                }
            )

    # Single operating thresholds evaluated across every duration/condition.
    operating = {}
    for value in args.thresholds:
        operating[str(value)] = {
            f"{r['condition']}-{r['seconds']}": {
                "frr": round(float((r["_target"] < value).mean()), 4),
                "far": round(float((r["_nontarget"] >= value).mean()), 4),
                "unenrolled_accept": round(float((r["_impostor"] >= value).mean()), 4),
            }
            for r in results
        }
    for r in results:
        for key in ("_target", "_nontarget", "_impostor"):
            del r[key]

    twin = speakers.index(next(k for k, v in names.items() if v == "Reed (English (US))"))
    twin_scores = [
        float(t["scores"][twin])
        for t in tests
        if names[t["speaker"]] == TWIN and t["condition"] == "clean"
    ]
    enroll_sim = voiceprints @ voiceprints.T
    off_diag = enroll_sim[~np.eye(len(speakers), dtype=bool)]

    return {
        "backend": args.backend,
        "model_label": args.label,
        "model_bytes": backend.size_bytes,
        "compute_units": args.compute_units if args.backend == "speakerkit" else "cpu",
        "speakerkit_pad": args.pad if args.backend == "speakerkit" else None,
        "threads": args.threads,
        "embedding_dim": int(voiceprints.shape[1]),
        "enrolled_speakers": len(speakers),
        "enroll_seconds": ENROLL_SECONDS,
        "clips_per_duration": args.clips_per_duration,
        "seed": args.seed,
        "load_seconds": round(load_seconds, 3),
        "latency_ms": {
            str(k): {
                "median": round(statistics.median(v) * 1000, 2),
                "p95": round(float(np.quantile(v, 0.95)) * 1000, 2),
            }
            for k, v in sorted(latency.items())
        },
        "peak_rss_mib": round(rss_peak * scale / 2**20, 1),
        "rss_before_model_mib": round(rss_data * scale / 2**20, 1),
        "rss_at_start_mib": round(rss_start * scale / 2**20, 1),
        "results": results,
        "operating_thresholds": operating,
        "near_twin_reed_uk_vs_reed_us_clean": _summary(np, twin_scores),
        "voiceprint_cross_similarity": _summary(np, off_diag),
        "host": {
            "machine": platform.machine(),
            "system": platform.platform(),
            "python": platform.python_version(),
        },
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="render synthetic macOS voices")
    gen.add_argument("--out", type=Path, required=True)
    ev = sub.add_parser("evaluate", help="score one embedding model, write metrics JSON")
    fp = sub.add_parser("footprint", help="fresh-process load time, RSS and latency")
    for command in (ev, fp):
        command.add_argument("--data", type=Path, required=True)
        command.add_argument(
            "--backend", choices=("speakerkit", "wespeaker", "speechbrain"), required=True
        )
        command.add_argument("--model", type=Path, required=True)
        command.add_argument("--label", default="")
        command.add_argument(
            "--compute-units", choices=("all", "cpu_and_ne", "cpu_only"), default="cpu_and_ne"
        )
        command.add_argument("--pad", choices=("tile", "zero"), default="tile")
        command.add_argument("--threads", type=int, default=4)
    ev.add_argument("--clips-per-duration", type=int, default=24)
    ev.add_argument("--seed", type=int, default=137)
    ev.add_argument("--thresholds", type=float, nargs="*", default=[0.3, 0.4, 0.5, 0.6])
    ev.add_argument("--output", type=Path, required=True)
    fp.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args(argv)
    if args.command == "generate":
        generate(args.out)
        return 0
    if args.command == "footprint":
        print(json.dumps(footprint(args), indent=2))
        return 0
    output = args.output.resolve()
    if output.exists():
        raise SystemExit("refusing to overwrite an existing metrics file")
    if _inside_git_checkout(output.parent):
        raise SystemExit("write metrics outside the Git checkout")
    report = evaluate(args)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
