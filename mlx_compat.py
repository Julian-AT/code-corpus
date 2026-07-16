from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

LEGACY_GEMMA4_MLX_IDS = {
    "mlx-community/gemma-4-e2b-4bit",
    "mlx-community/gemma-4-e2b-5bit",
    "mlx-community/gemma-4-e4b-4bit",
    "mlx-community/gemma-4-e4b-5bit",
    "unsloth/gemma-4-e4b-it-ud-mlx-4bit",
}


def normalize_legacy_gemma4_weights(weights: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(weights)
    for key, value in weights.items():
        if (
            "subsample_conv_projection" in key
            and "conv.weight" in key
            and value.ndim == 4
            and value.shape[1:3] == (3, 3)
        ):
            normalized[key] = value.transpose(0, 3, 1, 2)
        elif "depthwise_conv1d.weight" in key and value.ndim == 3 and value.shape[-1] == 1:
            normalized[key] = value.transpose(0, 2, 1)
    return normalized


@contextmanager
def gemma4_checkpoint_compatibility(model_path: str) -> Iterator[None]:
    """Restore legacy Gemma 4 MLX tensor layouts while loading a checkpoint."""
    if model_path.casefold() not in LEGACY_GEMMA4_MLX_IDS:
        yield
        return

    from mlx_vlm.models.gemma4.gemma4 import Model

    original = Model.sanitize

    def sanitize(model: Any, weights: Mapping[str, Any]) -> dict[str, Any]:
        return original(model, normalize_legacy_gemma4_weights(weights))

    Model.sanitize = sanitize
    try:
        yield
    finally:
        Model.sanitize = original
