from pathlib import Path

from evaluate import parse_training_log, report_markdown, write_charts


def test_report_keeps_missing_run_results_pending() -> None:
    results = {
        "generated_at": "2026-01-01T00:00:00+00:00",
        "evaluations": [
            {
                "run_id": "model-raw",
                "model": {"name": "model"},
                "variant": "raw-max",
                "status": "pending",
                "reason": "Training has not completed.",
            }
        ],
    }

    report = report_markdown(results, [], None)

    assert "pending" in report.casefold()
    assert "## Limitations" in report
    assert "Base TPS" in report
    assert "0.00" not in report


def test_local_charts_use_completed_e4b_measurements(tmp_path: Path) -> None:
    evaluations = [
        {
            "run_id": "gemma-4-e4b-it-raw-max-selection",
            "model": {"name": "gemma-4-e4b-it", "family": "Gemma 4 E4B"},
            "variant": "raw-max",
            "status": "completed",
            "base": {
                "generation_tps": 30.0,
                "perplexity": 4.0,
                "peak_memory_gb": 10.0,
            },
            "tuned": {
                "generation_tps": 29.0,
                "perplexity": 3.0,
                "peak_memory_gb": 10.5,
            },
        }
    ]

    training = [
        {
            "mode": "smoke",
            "status": "completed",
            "metrics": [
                {
                    "iteration": 1,
                    "train_loss": 1.2,
                    "tokens_per_second": 90.0,
                    "peak_memory_gb": 9.0,
                }
            ],
        }
    ]
    deployments = [
        {
            "tag": "gemma4-e4b-personal:smoke",
            "decode_tps": 80.0,
            "minimum_decode_tps": 20.0,
            "ttft_seconds": 0.3,
            "maximum_ttft_seconds": 5.0,
        }
    ]

    write_charts(tmp_path, evaluations, [], 72, training, deployments)

    for filename in (
        "tps-base-vs-tuned.png",
        "perplexity-by-model.png",
        "local-runtime-profile.png",
        "training-telemetry.png",
        "offline-deployment-headroom.png",
    ):
        assert (tmp_path / "charts" / filename).read_bytes().startswith(b"\x89PNG")
        assert (tmp_path / "charts" / filename.replace(".png", ".svg")).is_file()

    assert not (tmp_path / "charts/edge-throughput-q4-vs-q5.png").exists()


def test_training_log_parser_extracts_latest_report_for_each_iteration() -> None:
    log = (
        "Iter 10: Train loss \x1b[92m1.25000000\x1b[0m, Learning Rate 1.000e-05, "
        "It/sec 2.500, Tokens/sec 80.000, Trained Tokens 1000, Peak mem 9.250 GB\n"
        "Iter 10: Train loss 1.10000000, Learning Rate 9.000e-06, It/sec 3.000, "
        "Tokens/sec 90.000, Trained Tokens 1100, Peak mem 9.500 GB\n"
        "Iter 20: Val loss 1.050, Val took 2.300s\n"
        "Iter 20: Train loss 0.95000000, Learning Rate 8.000e-06, It/sec 3.200, "
        "Tokens/sec 95.000, Trained Tokens 2200, Peak mem 9.750 GB\n"
    )

    rows = parse_training_log(log)

    assert rows == [
        {
            "iteration": 10,
            "train_loss": 1.1,
            "learning_rate": 9e-6,
            "iterations_per_second": 3.0,
            "tokens_per_second": 90.0,
            "trained_tokens": 1100,
            "peak_memory_gb": 9.5,
        },
        {
            "iteration": 20,
            "train_loss": 0.95,
            "learning_rate": 8e-6,
            "iterations_per_second": 3.2,
            "tokens_per_second": 95.0,
            "trained_tokens": 2200,
            "peak_memory_gb": 9.75,
            "validation_loss": 1.05,
        },
    ]
