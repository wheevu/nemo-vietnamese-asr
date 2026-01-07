from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tempfile
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:  # pragma: no cover
    import torch as torch_types  # noqa: F401

try:
    import torch  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore

from src.inference import transcribe_audio
from src.model_utils import apply_quantization, get_default_device, load_nemo_model


def _require_torch():
    """Import guard for optional PyTorch dependency.

    Why: benchmarking runs in GPU environments (e.g., Colab). When torch isn't
    installed, we want a clear error message.

    Returns:
        The imported `torch` module.

    Raises:
        ModuleNotFoundError: If torch is not installed.
    """
    if torch is None:  # pragma: no cover
        raise ModuleNotFoundError(
            "PyTorch is required to run benchmark.py. "
            "Install it (e.g., in Colab it is preinstalled), or run: pip install torch"
        )
    return torch


@dataclass(frozen=True)
class ManifestSample:
    audio_filepath: str
    text: str


def read_manifest_samples(manifest_path: str, limit: int) -> List[ManifestSample]:
    """Read samples from a NeMo-style JSONL manifest.

    Args:
        manifest_path: Path to a JSONL manifest (one JSON per line).
        limit: Maximum number of samples to return.

    Returns:
        A list of `ManifestSample` objects (may be shorter than `limit`).
    """

    samples: List[ManifestSample] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if "audio_filepath" not in obj or "text" not in obj:
                continue
            samples.append(
                ManifestSample(audio_filepath=str(obj["audio_filepath"]), text=str(obj["text"]))
            )
            if len(samples) >= limit:
                break
    return samples


def normalize_text(text: str) -> str:
    """Normalize text for WER computation without damaging Vietnamese text.

    Why: WER comparisons are very sensitive to whitespace/case noise. We do a
    gentle normalization that keeps diacritics and most symbols.

    Args:
        text: Input string.

    Returns:
        Lowercased, whitespace-normalized string.
    """

    return " ".join((text or "").strip().lower().split())


def compute_wer_percent(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """Compute word error rate (WER) as a percentage.

    Args:
        references: Ground-truth reference transcripts.
        hypotheses: Model predicted transcripts.

    Returns:
        WER as a percentage (0–100+).
    """

    import jiwer

    refs = [normalize_text(r) for r in references]
    hyps = [normalize_text(h) for h in hypotheses]
    return float(jiwer.wer(refs, hyps) * 100.0)


def serialized_state_dict_size_mb(model) -> float:
    """Approximate model size by serializing state_dict to an in-memory buffer.

    Note: This is the serialized size, not necessarily the true runtime footprint
    (especially after quantization).
    """

    buf = io.BytesIO()
    cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    _torch = _require_torch()
    _torch.save(cpu_state, buf)
    return buf.getbuffer().nbytes / (1024 * 1024)


def cuda_sync_if_needed(device: "torch_types.device") -> None:
    """Synchronize CUDA if the device is GPU.

    Why: accurate latency measurements on GPU require synchronizing before/after
    timing to avoid async kernel overlap.

    Args:
        device: Torch device in use.
    """
    _torch = _require_torch()
    if device.type == "cuda":
        _torch.cuda.synchronize()


def maybe_remap_audio_path(path: str, audio_root: Optional[str]) -> str:
    """If `path` doesn't exist, try mapping by basename into `audio_root`.

    This helps when manifests were created on macOS (absolute paths) but are
    evaluated on Colab (different filesystem layout).
    """

    if os.path.exists(path) or not audio_root:
        return path

    candidate = os.path.join(audio_root, os.path.basename(path))
    return candidate


# Maximum audio duration (seconds) to avoid Conformer O(N²) OOM on T4.
MAX_AUDIO_DURATION_S = 30.0
TARGET_SAMPLE_RATE = 16000


def prepare_audio_for_benchmark(path: str) -> str:
    """Prepare audio for NeMo benchmark: mono, 16kHz, max 30s.

    Conformer attention is O(N²) in sequence length. Long audio files cause OOM
    on T4 GPUs. This function:
      1. Converts stereo to mono
      2. Resamples to 16kHz if needed
      3. Crops to MAX_AUDIO_DURATION_S seconds

    Returns path to a cached temp WAV file (or original if already compliant).
    """

    try:
        import librosa  # type: ignore
    except ModuleNotFoundError:
        # Fallback if librosa not available: use soundfile (no resampling)
        import soundfile as sf  # type: ignore
        import numpy as np  # type: ignore

        info = sf.info(path)
        sr = info.samplerate
        channels = getattr(info, "channels", 1)
        duration = info.duration

        # If already short mono 16kHz, return as-is
        if channels == 1 and sr == TARGET_SAMPLE_RATE and duration <= MAX_AUDIO_DURATION_S:
            return path

        tmp_dir = os.path.join(tempfile.gettempdir(), "nemo_vietnamese_asr_bench")
        os.makedirs(tmp_dir, exist_ok=True)

        key_src = f"{os.path.abspath(path)}|{sr}|{channels}|{duration}|{MAX_AUDIO_DURATION_S}".encode("utf-8")
        key = hashlib.sha1(key_src).hexdigest()[:12]
        base = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(tmp_dir, f"{base}.prep.{key}.wav")
        if os.path.exists(out_path):
            return out_path

        max_frames = int(MAX_AUDIO_DURATION_S * sr)
        audio, _ = sf.read(path, dtype="float32", always_2d=True, frames=max_frames)
        mono = np.mean(audio, axis=1)
        sf.write(out_path, mono, sr)
        return out_path

    import numpy as np  # type: ignore
    import soundfile as sf  # type: ignore

    tmp_dir = os.path.join(tempfile.gettempdir(), "nemo_vietnamese_asr_bench")
    os.makedirs(tmp_dir, exist_ok=True)

    # Cache key
    info = sf.info(path)
    key_src = f"{os.path.abspath(path)}|{info.samplerate}|{info.channels}|{info.duration}|{MAX_AUDIO_DURATION_S}".encode("utf-8")
    key = hashlib.sha1(key_src).hexdigest()[:12]
    base = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(tmp_dir, f"{base}.prep.{key}.wav")
    if os.path.exists(out_path):
        return out_path

    # Load with librosa: handles resampling + mono conversion + duration limit
    y, sr = librosa.load(path, sr=TARGET_SAMPLE_RATE, mono=True, duration=MAX_AUDIO_DURATION_S)
    sf.write(out_path, y, sr)
    return out_path


def collect_existing_pairs(
    samples: Sequence[ManifestSample], audio_root: Optional[str]
) -> Tuple[List[str], List[str]]:
    """Collect valid audio paths and matching reference texts.

    Why: manifests often contain absolute paths from another machine (macOS vs
    Colab). We remap when needed and skip missing/unreadable files.

    Args:
        samples: Manifest samples to consider.
        audio_root: Optional folder to remap by basename when paths don't exist.

    Returns:
        Tuple `(audio_paths, references)` with aligned indices.
    """
    audio_paths: List[str] = []
    references: List[str] = []

    for s in samples:
        p = maybe_remap_audio_path(s.audio_filepath, audio_root)
        if not os.path.exists(p):
            continue
        try:
            p = prepare_audio_for_benchmark(p)
        except Exception:
            # If audio preparation fails, skip this sample.
            continue
        audio_paths.append(p)
        references.append(s.text)

    return audio_paths, references


def benchmark_one_precision(
    *,
    model_ref: str,
    manifest_path: str,
    precision: str,
    sample_limit: int,
    batch_size: int,
    warmup_runs: int,
    device: "torch_types.device",
    audio_root: Optional[str],
) -> Dict[str, object]:
    """Benchmark one precision mode (float32/float16/int8).

    Why: we want latency, memory footprint, and WER under different precision
    settings, using a consistent evaluation path.

    Args:
        model_ref: Local `.nemo` path or NGC model name.
        manifest_path: JSONL manifest path.
        precision: Precision mode (`float32`, `float16`, or `int8`).
        sample_limit: Maximum number of manifest entries to evaluate.
        batch_size: Batch size passed to NeMo transcribe.
        warmup_runs: Number of warmup batches to run before timing.
        device: Torch device to run on.
        audio_root: Optional root directory to remap audio paths.

    Returns:
        A dict of benchmark results (keys match README table columns).
    """
    _torch = _require_torch()
    # Reload model each time to avoid state leakage.
    try:
        model = load_nemo_model(model_ref, device=device)
        model = apply_quantization(model, precision)
    except Exception as e:
        return {
            "Precision": precision,
            "Avg Latency (ms/file)": None,
            "WER (%)": None,
            "VRAM/Size (MB)": None,
            "Samples": 0,
            "Notes": f"Model load/quantization failed: {type(e).__name__}: {e}",
        }

    samples = read_manifest_samples(manifest_path, limit=sample_limit)
    audio_files, references = collect_existing_pairs(samples, audio_root)

    model_size_mb = serialized_state_dict_size_mb(model)

    if not audio_files:
        return {
            "Precision": precision,
            "Avg Latency (ms/file)": None,
            "WER (%)": None,
            "VRAM/Size (MB)": round(model_size_mb, 2),
            "Samples": 0,
            "Notes": "No audio files found (paths may need remapping).",
        }

    # Warmup: do a few short runs to stabilize kernels/caches.
    warmup_batch = audio_files[: max(1, min(batch_size, len(audio_files)))]
    try:
        with _torch.no_grad():  # Use no_grad instead of inference_mode for Quanto compatibility
            for _ in range(max(0, warmup_runs)):
                _ = transcribe_audio(model, warmup_batch, batch_size=len(warmup_batch))
    except RuntimeError as e:
        # Quanto int8 models can fail with "Cannot set version_counter for inference tensor"
        return {
            "Precision": precision,
            "Avg Latency (ms/file)": None,
            "WER (%)": None,
            "VRAM/Size (MB)": round(model_size_mb, 2),
            "Samples": 0,
            "Notes": f"Inference failed (likely Quanto incompatibility): {type(e).__name__}",
        }

    vram_peak_mb: Optional[float] = None
    if device.type == "cuda":
        _torch = _require_torch()
        _torch.cuda.reset_peak_memory_stats()

    # Timed inference loop
    hypotheses: List[str] = []
    cuda_sync_if_needed(device)
    t0 = time.perf_counter()

    try:
        with _torch.no_grad():  # Use no_grad instead of inference_mode for Quanto compatibility
            for i in range(0, len(audio_files), batch_size):
                batch = audio_files[i : i + batch_size]
                hypotheses.extend(transcribe_audio(model, batch, batch_size=len(batch)))
    except RuntimeError as e:
        return {
            "Precision": precision,
            "Avg Latency (ms/file)": None,
            "WER (%)": None,
            "VRAM/Size (MB)": round(model_size_mb, 2),
            "Samples": 0,
            "Notes": f"Inference failed: {type(e).__name__}",
        }

    cuda_sync_if_needed(device)
    t1 = time.perf_counter()

    if device.type == "cuda":
        _torch = _require_torch()
        vram_peak_mb = _torch.cuda.max_memory_allocated() / (1024 * 1024)

    total_s = t1 - t0
    avg_latency_ms = (total_s / len(audio_files)) * 1000.0
    wer_percent = compute_wer_percent(references, hypotheses)

    vram_or_size_mb = vram_peak_mb if vram_peak_mb is not None else model_size_mb

    return {
        "Precision": precision,
        "Avg Latency (ms/file)": round(avg_latency_ms, 2),
        "WER (%)": round(wer_percent, 2),
        "VRAM/Size (MB)": round(vram_or_size_mb, 2),
        "Samples": len(audio_files),
    }


def results_to_markdown_table(rows: List[Dict[str, object]]) -> str:
    """Render benchmark rows as a Markdown table.

    Args:
        rows: List of result dicts.

    Returns:
        Markdown table string suitable for README.
    """
    cols = ["Precision", "Avg Latency (ms/file)", "WER (%)", "VRAM/Size (MB)", "Samples"]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join([header, sep] + body)


def main() -> int:
    """Script entry point.

    Returns:
        Exit code (0 for success).
    """
    _require_torch()
    parser = argparse.ArgumentParser(description="Benchmark NeMo ASR quantization precisions.")
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Local .nemo checkpoint path OR NGC pretrained name "
            "(e.g., stt_en_conformer_ctc_large)."
        ),
    )
    parser.add_argument(
        "--manifest",
        default="val_manifest.json",
        help="Path to a NeMo JSONL manifest (default: val_manifest.json).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of manifest entries to benchmark (default: 100).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size passed to NeMo transcribe (default: 1).",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=3,
        help="Warmup transcription runs before timing (default: 3).",
    )
    parser.add_argument(
        "--audio-root",
        default=None,
        help=(
            "Optional directory to remap audio paths by basename when manifest paths don't exist "
            "(useful for macOS->Colab path differences)."
        ),
    )
    args = parser.parse_args()

    device = get_default_device()
    precisions = ["float32", "float16", "int8"]

    results: List[Dict[str, object]] = []
    for p in precisions:
        results.append(
            benchmark_one_precision(
                model_ref=args.model,
                manifest_path=args.manifest,
                precision=p,
                sample_limit=max(1, args.samples),
                batch_size=max(1, args.batch_size),
                warmup_runs=max(0, args.warmup_runs),
                device=device,
                audio_root=args.audio_root,
            )
        )

    print("\n=== Benchmark Results (raw) ===")
    for r in results:
        print(r)

    print("\n=== Markdown Table (paste into README) ===")
    print(results_to_markdown_table(results))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
