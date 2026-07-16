import json
import subprocess
import sys
from pathlib import Path

import yaml


def make_repository(root: Path, name: str, filename: str, body: str) -> None:
    repository = root / name
    repository.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    (repository / filename).write_text(body, encoding="utf-8")
    subprocess.run(["git", "add", filename], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Small Path",
            "-c",
            "user.email=small@example.com",
            "commit",
            "-qm",
            "initial",
        ],
        cwd=repository,
        check=True,
    )


def run_stage(project: Path, *arguments: str) -> None:
    subprocess.run(
        [sys.executable, str(project / arguments[0]), *arguments[1:]],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    )


def test_small_path_builds_dataset_prepares_smoke_and_writes_report(tmp_path: Path) -> None:
    project = Path(__file__).parents[1]
    config = yaml.safe_load((project / "config.yaml").read_text(encoding="utf-8"))
    config["paths"] = {
        "repos": str(tmp_path / "repos"),
        "stats": str(tmp_path / "stats"),
        "datasets": str(tmp_path / "datasets"),
        "runs": str(tmp_path / "runs"),
        "report": str(tmp_path / "report"),
    }
    config["training"]["models"][0]["model_id"] = "<FILL_MODEL_ID>"
    config["benchmarks"]["data"] = str(project / "benchmarks/public-benchmarks.json")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    make_repository(
        tmp_path / "repos",
        "alpha",
        "alpha.py",
        "def alpha(value):\n    return value + 1\n",
    )
    make_repository(
        tmp_path / "repos",
        "beta",
        "beta.ts",
        "export const beta = (value: string) => value.toUpperCase();\n",
    )

    run_stage(project, "stats.py", "--config", str(config_path), "--emails", "small@example.com")
    run_stage(
        project,
        "build_dataset.py",
        "--config",
        str(config_path),
        "--variant",
        "raw-max",
        "--max-rows",
        "20",
    )
    run_stage(project, "train.py", "--config", str(config_path))
    run_stage(project, "train.py", "--config", str(config_path), "--smoke", "--execute")
    run_stage(project, "evaluate.py", "--config", str(config_path), "--report-only")

    dataset_stats = json.loads(
        (tmp_path / "datasets/raw-max/statistics.json").read_text(encoding="utf-8")
    )
    run_state = json.loads(next((tmp_path / "runs").glob("*/run.json")).read_text(encoding="utf-8"))
    report_results = json.loads((tmp_path / "report/results.json").read_text(encoding="utf-8"))

    assert dataset_stats["self_test"]["status"] == "passed"
    assert dataset_stats["rows"] == {"total": 2, "train": 1, "valid": 1}
    assert run_state["status"] == "pending"
    assert report_results["evaluations"][0]["status"] == "pending"
    assert (tmp_path / "report/REPORT.md").is_file()
    assert (tmp_path / "report/charts/upstream-gemma4-benchmarks.png").is_file()
    assert (tmp_path / "report/charts/upstream-livecodebench-v6.svg").is_file()
    assert not (tmp_path / "report/charts/edge-throughput-q4-vs-q5.png").exists()
