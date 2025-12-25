from __future__ import annotations

import os
import warnings
from typing import Optional

try:
    import torch  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore


def _require_torch():
    if torch is None:  # pragma: no cover
        raise ModuleNotFoundError(
            "PyTorch is required for ASR model loading/benchmarking. "
            "Install it (e.g., in Colab it is preinstalled), or run: pip install torch"
        )
    return torch


def get_default_device() -> torch.device:
    """Prefer CUDA when available (e.g., Colab T4), otherwise CPU."""

    _torch = _require_torch()
    return _torch.device("cuda" if _torch.cuda.is_available() else "cpu")


def load_nemo_model(model_ref: str, device: Optional[torch.device] = None):
    """Load a NeMo ASR model.

    `model_ref` can be either:
      - A local path to a `.nemo` file (loaded via `restore_from`)
      - An NGC pretrained model name (loaded via `from_pretrained`)

    The returned model is put into eval mode and moved to the requested device.
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

    Returns True if quantization appears to have succeeded.

    This is a helper so we can implement NeMo-safe fallbacks.
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
    """Try a safer INT8 approach: quantize only `nn.Linear` layers.

    NeMo Conformer models contain Conv + attention + normalization stacks.
    Some Quanto workflows fail when quantizing everything indiscriminately.

    This helper attempts to quantize only Linear modules in-place.
    Returns the number of Linear layers successfully quantized.
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
    """Apply precision/quantization to a NeMo (PyTorch) model.

    Supported values:
      - "float32": no-op
      - "float16": cast the model to FP16 via `model.half()` (best for Colab T4)
      - "int8": attempt Quanto INT8 quantization, with NeMo-safe fallbacks

    INT8 notes:
      1) We first try to quantize the full model.
      2) If that fails (common with complex NeMo Conformer stacks), we then try
         to quantize only `nn.Linear` layers.
      3) If that still fails, we log a warning and fall back to float32.

    The quantization and freeze steps are performed in-place; the model is
    returned for convenience.
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
