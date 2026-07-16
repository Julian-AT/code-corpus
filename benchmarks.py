from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from corpuslib import final_summary

REQUIRED_OBSERVATION_FIELDS = {
    "model",
    "family",
    "parameter_class",
    "variant",
    "benchmark",
    "version",
    "score",
    "unit",
    "source_id",
}
GEMMA_MODEL_ORDER = [
    "Gemma 4 31B",
    "Gemma 4 26B A4B",
    "Gemma 4 12B Unified",
    "Gemma 4 E4B",
    "Gemma 4 E2B",
    "Gemma 3 27B (no think)",
]
GEMMA_PERCENT_BENCHMARKS = [
    "MMLU Pro",
    "AIME 2026",
    "LiveCodeBench",
    "GPQA Diamond",
    "Tau2",
    "MMMLU",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render sourced public benchmark charts.")
    parser.add_argument("--data", default="benchmarks/public-benchmarks.json")
    parser.add_argument("--output-dir", default="report/charts")
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("benchmark manifest root must be an object")
    sources = payload.get("sources")
    observations = payload.get("observations")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("benchmark manifest needs non-empty sources")
    if not isinstance(observations, list) or not observations:
        raise ValueError("benchmark manifest needs non-empty observations")
    seen: set[tuple[str, str, str, str]] = set()
    for index, observation in enumerate(observations):
        if not isinstance(observation, dict):
            raise ValueError(f"observation {index} must be an object")
        missing = REQUIRED_OBSERVATION_FIELDS - set(observation)
        if missing:
            raise ValueError(f"observation {index} is missing {', '.join(sorted(missing))}")
        if observation["source_id"] not in sources:
            raise ValueError(f"observation {index} references an unknown source")
        if not isinstance(observation["score"], (int, float)):
            raise ValueError(f"observation {index} score must be numeric")
        key = (
            str(observation["model"]),
            str(observation["benchmark"]),
            str(observation["version"]),
            str(observation["source_id"]),
        )
        if key in seen:
            raise ValueError(f"duplicate benchmark observation: {key}")
        seen.add(key)
    return payload


def new_figure(width: float, height: float) -> tuple[Figure, Any]:
    figure = Figure(figsize=(width, height), layout="constrained")
    FigureCanvasAgg(figure)
    return figure, figure.subplots()


def save_figure(figure: Figure, output_root: Path, stem: str, dpi: int) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_root / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    figure.savefig(output_root / f"{stem}.svg", bbox_inches="tight")


def source_name(manifest: Mapping[str, Any], source_id: str) -> str:
    source = manifest["sources"][source_id]
    return str(source["organization"])


def score_lookup(
    observations: Sequence[Mapping[str, Any]], source_id: str
) -> dict[tuple[str, str], float]:
    return {
        (str(item["model"]), str(item["benchmark"])): float(item["score"])
        for item in observations
        if item["source_id"] == source_id
    }


def gemma_heatmap(manifest: Mapping[str, Any], output_root: Path, dpi: int) -> None:
    observations = manifest["observations"]
    scores = score_lookup(observations, "google-gemma4-card")
    matrix = [
        [scores[(model, benchmark)] for benchmark in GEMMA_PERCENT_BENCHMARKS]
        for model in GEMMA_MODEL_ORDER
    ]
    figure, axis = new_figure(11.5, 5.7)
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".1f",
        cmap="crest",
        vmin=0,
        vmax=100,
        linewidths=0.8,
        cbar_kws={"label": "reported score (%)"},
        xticklabels=GEMMA_PERCENT_BENCHMARKS,
        yticklabels=GEMMA_MODEL_ORDER,
        ax=axis,
    )
    axis.set_title("Gemma upstream benchmark profile", loc="left", weight="bold")
    axis.set_xlabel("Google-reported instruction-tuned results; settings follow the model card")
    axis.set_ylabel("")
    axis.tick_params(axis="x", rotation=24)
    save_figure(figure, output_root, "upstream-gemma4-benchmarks", dpi)


def codeforces_chart(manifest: Mapping[str, Any], output_root: Path, dpi: int) -> None:
    observations = [
        item
        for item in manifest["observations"]
        if item["source_id"] == "google-gemma4-card" and item["benchmark"] == "Codeforces"
    ]
    by_model = {str(item["model"]): float(item["score"]) for item in observations}
    figure, axis = new_figure(9.5, 4.8)
    sns.barplot(
        x=[by_model[model] for model in GEMMA_MODEL_ORDER],
        y=GEMMA_MODEL_ORDER,
        hue=[
            "Selected E4B" if model == "Gemma 4 E4B" else "Reference"
            for model in GEMMA_MODEL_ORDER
        ],
        hue_order=["Selected E4B", "Reference"],
        palette={"Selected E4B": "#2A9D8F", "Reference": "#A8B0B9"},
        errorbar=None,
        dodge=False,
        ax=axis,
    )
    for container in axis.containers:
        axis.bar_label(container, fmt="%.0f", padding=4)
    axis.set_title("Gemma Codeforces rating", loc="left", weight="bold")
    axis.set_xlabel("Google-reported Elo (higher is better)")
    axis.set_ylabel("")
    axis.legend(title="", frameon=False, loc="lower right")
    save_figure(figure, output_root, "upstream-gemma4-codeforces", dpi)


def cross_family_chart(manifest: Mapping[str, Any], output_root: Path, dpi: int) -> None:
    observations = [
        item
        for item in manifest["observations"]
        if item["benchmark"] == "LiveCodeBench"
        and item["version"] == "v6"
        and (
            item["source_id"] == "qwen35-9b-card"
            or (item["source_id"] == "google-gemma4-card" and item["model"] == "Gemma 4 E4B")
        )
    ]
    ordered = sorted(observations, key=lambda item: float(item["score"]), reverse=True)
    reporters = [source_name(manifest, str(item["source_id"])) for item in ordered]
    palette = {"Google DeepMind": "#2A9D8F", "Qwen": "#5B6EE1"}
    figure, axis = new_figure(10.5, 5.8)
    sns.scatterplot(
        x=[float(item["score"]) for item in ordered],
        y=[str(item["model"]) for item in ordered],
        hue=reporters,
        style=reporters,
        palette=palette,
        s=140,
        ax=axis,
    )
    for item in ordered:
        axis.text(
            float(item["score"]) + 0.8,
            str(item["model"]),
            f"{float(item['score']):.1f}",
            va="center",
        )
    axis.set_xlim(45, 88)
    axis.set_title("Published LiveCodeBench v6 results", loc="left", weight="bold")
    axis.set_xlabel("reported score (%) — compare with source/evaluator caveat")
    axis.set_ylabel("")
    axis.legend(title="Reported by", frameon=False, loc="lower right")
    save_figure(figure, output_root, "upstream-livecodebench-v6", dpi)


def write_public_charts(manifest: Mapping[str, Any], output_root: Path, dpi: int) -> None:
    sns.set_theme(
        context="paper",
        style="ticks",
        palette="colorblind",
        font_scale=1.15,
        rc={"axes.titlepad": 12, "figure.facecolor": "white", "axes.facecolor": "white"},
    )
    gemma_heatmap(manifest, output_root, dpi)
    codeforces_chart(manifest, output_root, dpi)
    cross_family_chart(manifest, output_root, dpi)


def main() -> int:
    args = parse_args()
    data_path = Path(args.data).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    if args.dpi <= 0:
        raise SystemExit("--dpi must be greater than zero")
    manifest = load_manifest(data_path)
    write_public_charts(manifest, output_root, args.dpi)
    final_summary(
        "Public benchmarks",
        [
            ("Observations", len(manifest["observations"])),
            ("Sources", len(manifest["sources"])),
            ("Charts", output_root),
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
