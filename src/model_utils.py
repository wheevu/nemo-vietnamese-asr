from __future__ import annotations

import os
import warnings
from typing import Optional

try:
    import torch  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore


def _require_torch():
    """Import guard for optional PyTorch dependency.

    Why: local ETL/harvesting can run without torch; model loading requires it.

    Returns:
        The imported `torch` module.

    Raises:
        ModuleNotFoundError: If torch is not installed.
    """
    if torch is None:  # pragma: no cover
        raise ModuleNotFoundError(
            "PyTorch is required for ASR model loading/benchmarking. "
            "Install it (e.g., in Colab it is preinstalled), or run: pip install torch"
        )
    return torch


def get_default_device() -> torch.device:
    """Choose a reasonable default torch device.

    Why: this repo frequently runs in Colab. If CUDA is available we prefer it,
    otherwise we fall back to CPU.

    Returns:
        A `torch.device` instance (`cuda` if available else `cpu`).
    """

    _torch = _require_torch()
    return _torch.device("cuda" if _torch.cuda.is_available() else "cpu")


def load_nemo_model(model_ref: str, device: Optional[torch.device] = None):
    """Load a NeMo ASR model from disk or from NGC.

    Args:
        model_ref: Either a local `.nemo` path (restore) or an NGC model name
            (download via `from_pretrained`).
        device: Torch device to move the model to. If None, uses
            `get_default_device()`.

    Returns:
        A NeMo ASR model in eval mode on the requested device.

    Raises:
        ModuleNotFoundError: If PyTorch is not installed.
    """

    _torch = _require_torch()
    device = device or get_default_device()

    # Import inside the function to keep local ETL usage lightweight.
    import nemo.collections.asr as nemo_asr

    if model_ref.endswith(".nemo") and os.path.exists(model_ref):
        model = nemo_asr.models.ASRModel.restore_from(restore_path=model_ref)
    else:
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_ref)

    model.eval()
    model = model.to(device)
    return model


def _quanto_quantize_in_place(model) -> bool:
    """Attempt to quantize a model in-place using Quanto.

    Why: INT8 can reduce memory footprint and improve latency, but quantization
    support varies across architectures and libraries. This helper is one step
    in a fallback chain.

    Args:
        model: A PyTorch model to quantize.

    Returns:
        True if quantization completed without raising.
    """

    # Import inside the function so Quanto is only required for int8 mode.
    from quanto import freeze, quantize  # type: ignore

    model.eval()

    # Different quanto versions / builds expose slightly different APIs.
    # We try a couple common call signatures.
    try:
        from quanto import qint8  # type: ignore

        try:
            quantize(model, weights=qint8)
        except TypeError:
            quantize(model, qint8)
    except Exception:
        # Fallback: let quanto choose defaults (if supported)
        quantize(model)

    freeze(model)
    return True


def _quanto_quantize_linear_layers_in_place(model) -> int:
    """Try a safer INT8 approach by quantizing only `nn.Linear` layers.

    Why: full-model quantization often fails on NeMo Conformer stacks. Linear-only
    quantization is a pragmatic compromise that sometimes works.

    Args:
        model: A PyTorch model to quantize in-place.

    Returns:
        Number of `nn.Linear` submodules successfully quantized.
    """

    import torch.nn as nn

    from quanto import freeze, quantize  # type: ignore

    # Try to get qint8 if present; if not, quantize() may still work.
    try:
        from quanto import qint8  # type: ignore

        q = qint8
    except Exception:
        q = None

    quantized_count = 0
    for _name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        try:
            if q is None:
                quantize(module)
            else:
                try:
                    quantize(module, weights=q)
                except TypeError:
                    quantize(module, q)
            quantized_count += 1
        except Exception:
            # Ignore individual failures; we only need partial success.
            continue

    if quantized_count > 0:
        freeze(model)

    return quantized_count


def apply_quantization(model, precision: str):
    """Apply a precision mode to a NeMo (PyTorch) model.

    Supported values:
    - `float32`: no-op
    - `float16`: cast weights to FP16 (`model.half()`), typically best on Colab T4
    - `int8`: attempt Quanto INT8 quantization with safe fallbacks

    Why the fallbacks: NeMo Conformer models are complex; some INT8 paths fail
    depending on Quanto/torch versions. We prefer returning a working model over
    hard-failing.

    Args:
        model: PyTorch model to modify (in-place for INT8).
        precision: One of `float32`, `float16`, `int8` (case-insensitive).

    Returns:
        The (possibly modified) model.

    Raises:
        ValueError: If `precision` is not a supported value.
    """

    precision = (precision or "float32").lower().strip()

    if precision == "float32":
        return model

    if precision == "float16":
        # Colab T4 supports FP16 well; BF16 generally isn't the best choice on T4.
        return model.half()

    if precision == "int8":
        try:
            _quanto_quantize_in_place(model)
            return model
        except Exception as e:
            warnings.warn(
                f"[apply_quantization] Full-model INT8 quantization failed: {type(e).__name__}: {e}. "
                "Attempting Linear-only quantization.",
                RuntimeWarning,
            )

        try:
            quantized_count = _quanto_quantize_linear_layers_in_place(model)
            if quantized_count > 0:
                warnings.warn(
                    f"[apply_quantization] Linear-only INT8 quantization succeeded for {quantized_count} layers.",
                    RuntimeWarning,
                )
                return model
        except Exception as e:
            warnings.warn(
                f"[apply_quantization] Linear-only INT8 quantization failed: {type(e).__name__}: {e}.",
                RuntimeWarning,
            )

        warnings.warn(
            "[apply_quantization] Falling back to float32 (unquantized model).",
            RuntimeWarning,
        )
        return model

    raise ValueError(
        f"Unsupported precision: {precision}. Expected one of: float32, float16, int8"
    )
