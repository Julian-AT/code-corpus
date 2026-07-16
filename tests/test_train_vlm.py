import numpy as np

from mlx_compat import LEGACY_GEMMA4_MLX_IDS, normalize_legacy_gemma4_weights


def test_legacy_gemma4_audio_weights_are_restored_to_source_layout() -> None:
    weights = {
        "audio_tower.subsample_conv_projection.layer0.conv.weight": np.zeros((128, 3, 3, 1)),
        "audio_tower.layers.0.lconv1d.depthwise_conv1d.weight": np.zeros((1024, 5, 1)),
        "language_model.model.layers.0.input_layernorm.weight": np.zeros((256,)),
    }

    normalized = normalize_legacy_gemma4_weights(weights)

    assert normalized["audio_tower.subsample_conv_projection.layer0.conv.weight"].shape == (
        128,
        1,
        3,
        3,
    )
    assert normalized["audio_tower.layers.0.lconv1d.depthwise_conv1d.weight"].shape == (1024, 1, 5)
    assert normalized["language_model.model.layers.0.input_layernorm.weight"].shape == (256,)


def test_pinned_e4b_instruction_checkpoint_uses_legacy_layout_compatibility() -> None:
    assert "unsloth/gemma-4-e4b-it-ud-mlx-4bit" in LEGACY_GEMMA4_MLX_IDS
