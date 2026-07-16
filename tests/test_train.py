from dataclasses import replace
from pathlib import Path

from train import (
    ModelSpec,
    SweepRun,
    TrainingProfile,
    base_command,
    prepare_training_data,
    runnable_state,
    split_completion,
)


def profile() -> TrainingProfile:
    return TrainingProfile(
        batch_size=1,
        iterations=1,
        max_seq_length=1024,
        learning_rate=1e-5,
        gradient_checkpointing=True,
        gradient_accumulation_steps=1,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.0,
        prompt_fraction=0.5,
        validation_batches=10,
        steps_per_report=10,
        steps_per_eval=100,
        save_every=1,
        seed=42,
    )


def test_placeholder_is_pending_and_never_runnable(tmp_path: Path) -> None:
    item = SweepRun(
        run_id="e2b-raw",
        model=ModelSpec("e2b", "Gemma", "<FILL_MODEL_ID>", "fill it"),
        variant="raw-max",
        dataset_path=tmp_path / "dataset",
        run_path=tmp_path / "run",
        adapter_path=tmp_path / "run/adapters",
        profile=profile(),
    )

    state, reason = runnable_state(item)

    assert state == "pending"
    assert "FILL_MODEL_ID" in str(reason)


def test_command_is_native_mlx_vlm_lora_with_safe_profile(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "train.jsonl").write_text('{"text":"code"}\n', encoding="utf-8")
    item = SweepRun(
        run_id="model-raw",
        model=ModelSpec("model", "Code", "mlx-community/example-4bit", "verified"),
        variant="raw-max",
        dataset_path=dataset,
        run_path=tmp_path / "run",
        adapter_path=tmp_path / "run/adapters",
        profile=profile(),
    )

    prepare_training_data(item)
    command = base_command(item)
    state, reason = runnable_state(item)

    assert state == "ready"
    assert reason is None
    assert command[:7] == [
        "uv",
        "run",
        "--extra",
        "train",
        "python",
        "train_vlm.py",
        "--model-path",
    ]
    assert "--grad-checkpoint" in command
    assert "--max-seq-length" in command
    assert "--train-on-completions" in command
    assert "--iters" in command
    assert "bitsandbytes" not in " ".join(command)


def test_q5_model_is_runnable_when_metadata_matches_model_id(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "train.jsonl").write_text('{"text":"code"}\n', encoding="utf-8")
    item = SweepRun(
        run_id="model-q5-raw",
        model=ModelSpec(
            "model-q5",
            "Code",
            "mlx-community/example-5bit",
            "verified",
            quantization_bits=5,
        ),
        variant="raw-max",
        dataset_path=dataset,
        run_path=tmp_path / "run",
        adapter_path=tmp_path / "run/adapters",
        profile=profile(),
    )

    prepare_training_data(item)

    assert runnable_state(item) == ("ready", None)


def test_training_staging_omits_empty_optional_validation_split(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "train.jsonl").write_text('{"text":"code"}\n', encoding="utf-8")
    (dataset / "valid.jsonl").write_text("", encoding="utf-8")
    item = SweepRun(
        run_id="model-raw",
        model=ModelSpec("model", "Code", "mlx-community/example-4bit", "verified"),
        variant="raw-max",
        dataset_path=dataset,
        run_path=tmp_path / "run",
        adapter_path=tmp_path / "run/adapters",
        profile=profile(),
    )

    prepare_training_data(item)

    staged = (item.training_data_path / "train.jsonl").read_text(encoding="utf-8")

    assert '"question"' in staged
    assert '"answer"' in staged
    assert not (item.training_data_path / "valid.jsonl").exists()


def test_completion_split_preserves_the_original_text() -> None:
    text = "def add(a, b):\n    total = a + b\n    return total\n"

    parts = split_completion(text, 0.5)

    assert parts is not None
    assert "".join(parts) == text
    assert parts[0].endswith("\n")


def test_production_training_merges_train_and_valid_rows(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "train.jsonl").write_text(
        '{"text":"def train():\\n    return 1\\n"}\n', encoding="utf-8"
    )
    (dataset / "valid.jsonl").write_text(
        '{"text":"def valid():\\n    return 2\\n"}\n', encoding="utf-8"
    )
    item = SweepRun(
        run_id="model-production",
        model=ModelSpec("model", "Code", "mlx-community/example-4bit", "verified"),
        variant="raw-max",
        dataset_path=dataset,
        run_path=tmp_path / "run",
        adapter_path=tmp_path / "run/adapters",
        profile=profile(),
        mode="production",
    )

    prepare_training_data(item)

    rows = (item.training_data_path / "train.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2
    assert not (item.training_data_path / "valid.jsonl").exists()


def test_epoch_profile_uses_epochs_instead_of_iterations(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "train.jsonl").write_text('{"text":"code"}\n', encoding="utf-8")
    epoch_profile = replace(profile(), iterations=None, epochs=1)
    item = SweepRun(
        run_id="model-selection",
        model=ModelSpec("model", "Code", "mlx-community/example-4bit", "verified"),
        variant="raw-max",
        dataset_path=dataset,
        run_path=tmp_path / "run",
        adapter_path=tmp_path / "run/adapters",
        profile=epoch_profile,
    )

    prepare_training_data(item)
    command = base_command(item)

    assert command[command.index("--epochs") + 1] == "1"
    assert "--iters" not in command
