from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from deploy_ollama import (
    adapter_files,
    adapter_pairs,
    codex_command,
    converter_command,
    expected_tool_call,
    export_peft_adapter,
    peft_adapter_config,
    peft_tensor_name,
    render_modelfile,
    require_disk_space,
    runtime_metrics,
)


def test_modelfile_uses_matching_ollama_base_and_gguf_adapter(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter.gguf"

    rendered = render_modelfile(
        "gemma4:e4b",
        adapter,
        context_window=32_768,
        system_prompt="You are Julian's assistant.",
    )

    assert rendered == (
        f"FROM gemma4:e4b\nADAPTER {adapter.resolve()}\n"
        "PARAMETER temperature 0\nPARAMETER num_ctx 32768\n"
        'SYSTEM """You are Julian\'s assistant."""\n'
    )


def test_modelfile_rejects_system_prompt_delimiter(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="triple double quotes"):
        render_modelfile(
            "gemma4:e4b",
            tmp_path / "adapter.gguf",
            context_window=32_768,
            system_prompt='invalid """ prompt',
        )


def test_adapter_validation_requires_weights_and_config(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapters.safetensors").write_bytes(b"weights")

    with pytest.raises(ValueError, match="adapter_config.json"):
        adapter_files(adapter)


def test_peft_tensor_name_transposes_mlx_lora_factors() -> None:
    a_name, transpose_a = peft_tensor_name("language_model.model.layers.0.self_attn.q_proj.lora_a")
    b_name, transpose_b = peft_tensor_name("language_model.model.layers.0.self_attn.q_proj.lora_b")

    assert a_name == (
        "base_model.model.language_model.model.layers.0.self_attn.q_proj.lora_A.weight"
    )
    assert b_name == (
        "base_model.model.language_model.model.layers.0.self_attn.q_proj.lora_B.weight"
    )
    assert transpose_a
    assert transpose_b


def test_adapter_pairs_reject_incomplete_lora_modules() -> None:
    with pytest.raises(ValueError, match="missing lora_b"):
        adapter_pairs(["model.layers.0.self_attn.q_proj.lora_a"])


def test_peft_adapter_config_preserves_mlx_scale() -> None:
    config = peft_adapter_config(
        {
            "lora_parameters": {
                "rank": 8,
                "scale": 2.0,
                "dropout": 0.0,
                "keys": [
                    "language_model.model.layers.0.self_attn.q_proj",
                    "language_model.model.layers.0.mlp.down_proj",
                ],
            }
        },
        "google/gemma-4-E4B-it",
    )

    assert config["r"] == 8
    assert config["lora_alpha"] == 16.0
    assert config["target_modules"] == ["down_proj", "q_proj"]
    assert config["base_model_name_or_path"] == "google/gemma-4-E4B-it"


def test_export_peft_adapter_transposes_pairs_and_drops_frozen_tensors(tmp_path: Path) -> None:
    source = tmp_path / "mlx"
    source.mkdir()
    prefix = "language_model.model.layers.0.self_attn.q_proj"
    a = np.arange(6, dtype=np.float32).reshape(3, 2)
    b = np.arange(8, dtype=np.float32).reshape(2, 4)
    save_file(
        {
            f"{prefix}.lora_a": a,
            f"{prefix}.lora_b": b,
            "audio_tower.layers.0.input_min": np.zeros((), dtype=np.float32),
        },
        source / "adapters.safetensors",
    )
    (source / "adapter_config.json").write_text(
        f'{{"lora_parameters":{{"rank":2,"scale":4,"dropout":0,"keys":["{prefix}"]}}}}',
        encoding="utf-8",
    )

    result = export_peft_adapter(source, tmp_path / "peft", "google/gemma-4-E4B-it")
    exported = load_file(tmp_path / "peft" / "adapter_model.safetensors")

    np.testing.assert_array_equal(exported[f"base_model.model.{prefix}.lora_A.weight"], a.T)
    np.testing.assert_array_equal(exported[f"base_model.model.{prefix}.lora_B.weight"], b.T)
    assert len(exported) == 2
    assert result["tensor_pairs"] == 1


def test_converter_command_is_pinned_to_local_llama_cpp(tmp_path: Path) -> None:
    command = converter_command(
        python=Path("/venv/bin/python"),
        llama_cpp=tmp_path / "llama.cpp",
        peft_adapter=tmp_path / "peft",
        base_config=tmp_path / "base",
        output=tmp_path / "adapter.gguf",
    )

    assert command == [
        "/venv/bin/python",
        str(tmp_path / "llama.cpp" / "convert_lora_to_gguf.py"),
        "--base",
        str(tmp_path / "base"),
        "--outfile",
        str(tmp_path / "adapter.gguf"),
        "--outtype",
        "f16",
        str(tmp_path / "peft"),
    ]


def test_codex_command_overrides_unknown_local_model_context(tmp_path: Path) -> None:
    command = codex_command("gemma4-e4b-personal", tmp_path, context_window=32_768)

    assert "model_context_window=32768" in command
    assert "model_auto_compact_token_limit=24576" in command
    assert command[-1].startswith("Fix the failing test")


def test_runtime_metrics_use_ollama_nanosecond_durations() -> None:
    metrics = runtime_metrics(
        {"eval_count": 40, "eval_duration": 2_000_000_000, "total_duration": 3_000_000_000}
    )

    assert metrics == {"eval_count": 40, "decode_tps": 20.0, "ttft_seconds": 1.0}


def test_expected_tool_call_requires_the_named_function() -> None:
    response = {
        "message": {"tool_calls": [{"function": {"name": "read_project_status", "arguments": {}}}]}
    }

    assert expected_tool_call(response, "read_project_status")
    assert not expected_tool_call(response, "write_file")


def test_disk_preflight_includes_operational_reserve(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="required before Ollama import"):
        require_disk_space(tmp_path, minimum_free_gb=1_000_000, additional_required_gb=1)
