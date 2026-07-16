import json
from pathlib import Path

import pytest

from benchmarks import load_manifest, write_public_charts


def test_public_benchmark_manifest_is_source_traceable() -> None:
    root = Path(__file__).parents[1]
    manifest = load_manifest(root / "benchmarks/public-benchmarks.json")

    assert len(manifest["observations"]) == 48
    assert {item["source_id"] for item in manifest["observations"]} == {
        "google-gemma4-card",
        "qwen35-9b-card",
    }
    assert all(source["url"].startswith("https://") for source in manifest["sources"].values())


def test_duplicate_benchmark_observations_are_rejected(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    payload = json.loads(
        (root / "benchmarks/public-benchmarks.json").read_text(encoding="utf-8")
    )
    payload["observations"].append(dict(payload["observations"][0]))
    path = tmp_path / "benchmarks.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate benchmark observation"):
        load_manifest(path)


def test_public_charts_export_png_and_svg(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    manifest = load_manifest(root / "benchmarks/public-benchmarks.json")

    write_public_charts(manifest, tmp_path, 72)

    for stem in (
        "upstream-gemma4-benchmarks",
        "upstream-gemma4-codeforces",
        "upstream-livecodebench-v6",
    ):
        assert (tmp_path / f"{stem}.png").read_bytes().startswith(b"\x89PNG")
        assert b"<svg" in (tmp_path / f"{stem}.svg").read_bytes()[:500]
