from pathlib import Path

from build_dataset import (
    BuildSettings,
    Deduper,
    QualitySettings,
    chunk_source,
    collect_rows,
    detected_secret_kind,
    line_statistics,
    write_artifacts,
)
from corpuslib import FilterRules, load_config, repository_split
from datasets import load_dataset


def make_repository(root: Path, name: str, filename: str, body: str) -> Path:
    import subprocess

    repository = root / name
    repository.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    (repository / filename).write_text(body, encoding="utf-8")
    subprocess.run(["git", "add", filename], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Corpus Test",
            "-c",
            "user.email=corpus@example.com",
            "commit",
            "-qm",
            "initial",
        ],
        cwd=repository,
        check=True,
    )
    return repository


def test_chunk_source_overlaps_lexical_tokens() -> None:
    chunks = list(chunk_source("alpha beta gamma delta epsilon zeta", 4, 1))

    assert [count for _, count in chunks] == [4, 3]
    assert chunks[1][0].startswith("delta")


def test_line_statistics_counts_emitted_and_nonblank_lines() -> None:
    rows = [{"text": "one\n\ntwo\n"}, {"text": "three"}]

    assert line_statistics(rows) == {"total": 4, "nonblank": 3}


def test_repository_split_never_crosses_repositories() -> None:
    split = repository_split(["one", "two", "three"], 0.05, 42)

    assert set(split) == {"one", "two", "three"}
    assert list(split.values()).count("valid") == 1


def test_repository_split_honors_minimum_without_emptying_train() -> None:
    split = repository_split(
        [f"repo-{index}" for index in range(20)],
        validation_ratio=0.1,
        seed=42,
        minimum_validation=3,
    )

    assert list(split.values()).count("valid") == 3
    assert list(split.values()).count("train") == 17


def test_secret_detection_returns_only_the_secret_kind() -> None:
    token = "AKIA" + "A" * 16

    assert detected_secret_kind(f'AWS_ACCESS_KEY_ID="{token}"') == "aws_access_key"
    assert detected_secret_kind("API_KEY = 'replace-me'") is None


def test_exact_and_near_duplicates_are_rejected() -> None:
    deduper = Deduper(threshold=0.8, permutations=128, shingle_tokens=3, seed=42)
    original = " ".join(f"token_{index}" for index in range(100))
    near = original.replace("token_50", "changed")

    assert deduper.accept(original, "original") == (True, "kept")
    assert deduper.accept(original, "original") == (False, "exact_duplicate")
    assert deduper.accept(near, "near") == (False, "near_duplicate")


def test_raw_dataset_writes_and_loads_hugging_face_directory(tmp_path: Path) -> None:
    repository_root = tmp_path / "repos"
    make_repository(
        repository_root,
        "alpha",
        "alpha.py",
        "def alpha(value):\n    return value + 1\n",
    )
    make_repository(
        repository_root,
        "beta",
        "beta.ts",
        "export const beta = (value: string) => value.toUpperCase();\n",
    )
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
        validation_ratio=0.05,
        seed=42,
        emails=("corpus@example.com",),
        rules=FilterRules.from_config(config, max_file_bytes=1024 * 1024),
        quality=QualitySettings(
            drop_tests=True,
            min_history_commits=2,
            min_lifetime_days=30,
            min_score=2,
            history_weight=1,
            lifetime_weight=1,
            personal_weight=2,
        ),
        dataset_license="other",
        dataset_languages=("en",),
    )

    rows, audit = collect_rows(settings)
    statistics = write_artifacts(rows, audit, settings)
    loaded = load_dataset(str(settings.output_root / "raw-max"))

    assert statistics["self_test"]["status"] == "passed"
    assert len(loaded["train"]) + len(loaded["valid"]) == 2
    assert {row["repo"] for split in loaded.values() for row in split} == {"alpha", "beta"}


def test_quality_dataset_keeps_personal_code_without_leaking_single_repo(tmp_path: Path) -> None:
    repository_root = tmp_path / "repos"
    make_repository(
        repository_root,
        "alpha",
        "alpha.py",
        "def alpha(value):\n    return value + 1\n",
    )
    config, _ = load_config(Path(__file__).parents[1] / "config.yaml")
    settings = BuildSettings(
        variant="quality",
        repository_root=repository_root,
        output_root=tmp_path / "datasets",
        max_rows=20,
        chunk_tokens=32,
        overlap_tokens=4,
        near_threshold=0.85,
        minhash_permutations=32,
        shingle_tokens=3,
        validation_ratio=0.05,
        seed=42,
        emails=("corpus@example.com",),
        rules=FilterRules.from_config(config, max_file_bytes=1024 * 1024),
        quality=QualitySettings(
            drop_tests=True,
            min_history_commits=2,
            min_lifetime_days=30,
            min_score=2,
            history_weight=1,
            lifetime_weight=1,
            personal_weight=2,
        ),
        dataset_license="other",
        dataset_languages=("en",),
    )

    rows, audit = collect_rows(settings)
    statistics = write_artifacts(rows, audit, settings)
    loaded = load_dataset(str(settings.output_root / "quality"))

    assert statistics["rows"] == {"total": 1, "train": 1, "valid": 0}
    assert statistics["self_test"]["status"] == "passed"
    assert set(loaded) == {"train"}
