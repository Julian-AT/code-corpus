from pathlib import Path

from build_dataset import BuildSettings, QualitySettings, collect_rows, write_artifacts
from corpuslib import FilterRules, load_config
from publish_dataset import (
    prepare_upload,
    retained_private_repositories,
    validate_upload,
)
from tests.test_build_dataset import make_repository


def generated_dataset(tmp_path: Path) -> Path:
    repository_root = tmp_path / "repos"
    make_repository(repository_root, "alpha", "alpha.py", "def alpha():\n    return 1\n")
    make_repository(repository_root, "beta", "beta.ts", "export const beta = 2;\n")
    config, _ = load_config(Path(__file__).parents[1] / "config.yaml")
    settings = BuildSettings(
        variant="raw-max",
        repository_root=repository_root,
        output_root=tmp_path / "datasets",
        max_rows=20,
        chunk_tokens=32,
        overlap_tokens=4,
        near_threshold=0.85,
        minhash_permutations=32,
        shingle_tokens=3,
        validation_ratio=0.5,
        seed=42,
        emails=(),
        rules=FilterRules.from_config(config, max_file_bytes=1024 * 1024),
        quality=QualitySettings(True, 2, 30, 2, 1, 1, 2),
        dataset_license="other",
        dataset_languages=("code",),
    )
    rows, audit = collect_rows(settings)
    write_artifacts(rows, audit, settings)
    return settings.output_root / settings.variant


def test_prepare_upload_is_minimal_sanitized_and_loadable(tmp_path: Path) -> None:
    source = generated_dataset(tmp_path)
    destination = tmp_path / "upload"

    prepare_upload(source, destination, "owner/code-corpus")
    loaded = validate_upload(destination)

    files = {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file()
    }
    assert files == {
        "README.md",
        "dataset_infos.json",
        "statistics.json",
        "data/train-00000-of-00001.parquet",
        "data/valid-00000-of-00001.parquet",
    }
    assert sum(len(split) for split in loaded.values()) == 2
    statistics = (destination / "statistics.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in statistics
    assert 'datasets.load_dataset(\\"owner/code-corpus\\")' in statistics


def test_retained_private_repositories_only_reports_contributors() -> None:
    statistics = {"by_repo": {"private-used": 3, "public-used": 2}}
    manifest = {
        "repositories": [
            {"name": "private-used", "isPrivate": True},
            {"name": "private-unused", "isPrivate": True},
            {"name": "public-used", "isPrivate": False},
        ]
    }

    assert retained_private_repositories(statistics, manifest) == ["private-used"]
