from __future__ import annotations

import inspect
from typing import List, Sequence

try:
    import torch  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore


def _require_torch():
    """Import guard for optional PyTorch dependency.

    Why: data-prep parts of this repo can run without torch. We only require
    torch when doing transcription/benchmarking.

    Returns:
        The imported `torch` module.

    Raises:
        ModuleNotFoundError: If torch is not installed.
    """
    if torch is None:  # pragma: no cover
        raise ModuleNotFoundError(
            "PyTorch is required for ASR transcription. "
            "Install it (e.g., in Colab it is preinstalled), or run: pip install torch"
        )
    return torch


def transcribe_audio(model, audio_files: Sequence[str], batch_size: int = 1) -> List[str]:
    """Transcribe audio files using a NeMo ASR model.

    Why this wrapper exists: NeMo has changed the `transcribe()` signature across
    versions. We inspect the callable and dispatch in a compatible way.

    Args:
        model: A NeMo ASR model instance with a `transcribe` method.
        audio_files: Sequence of audio file paths.
        batch_size: Batch size passed to NeMo (default: 1).

    Returns:
        List of transcription strings, one per input audio file.
    """

    if not audio_files:
        return []

    model.eval()

    def _to_text(p) -> str:
        # Some NeMo versions return objects with `.text`.
        if hasattr(p, "text"):
            try:
                return str(getattr(p, "text"))
            except Exception:
                pass
        return str(p)

    _torch = _require_torch()
    # Note: We use no_grad instead of inference_mode because Quanto quantized
    # models can fail with "Cannot set version_counter for inference tensor"
    with _torch.no_grad():
        transcribe_fn = getattr(model, "transcribe")

        sig = None
        try:
            sig = inspect.signature(transcribe_fn)
        except (TypeError, ValueError):
            sig = None

        preds = None

        # NeMo has had multiple `transcribe()` signatures over time. Prefer
        # explicit dispatch based on the installed version.
        if sig is not None and "paths2audio_files" in sig.parameters:
            preds = transcribe_fn(paths2audio_files=list(audio_files), batch_size=batch_size)
        elif sig is not None and "audio" in sig.parameters:
            # Some variants use `audio=` for a list of filepaths.
            preds = transcribe_fn(audio=list(audio_files), batch_size=batch_size)
        else:
            try:
                preds = transcribe_fn(list(audio_files), batch_size=batch_size)
            except TypeError:
                preds = transcribe_fn(list(audio_files))

    return [_to_text(p) for p in (preds or [])]
