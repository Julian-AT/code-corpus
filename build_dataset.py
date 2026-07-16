from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import subprocess
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from datasketch import MinHash, MinHashLSH

from corpuslib import (
    LOG,
    FilterRules,
    bounded_rows,
    config_value,
    configure_logging,
    configured_emails,
    content_hash,
    content_tokens,
    discover_repositories,
    final_summary,
    gitignored_paths,
    is_test_path,
    load_config,
    positive_int,
    repository_split,
    resolved_path,
    utc_now,
    write_json,
    write_text,
)
from datasets import Dataset, DatasetDict, Features, Value, load_dataset

FEATURES = Features(
    {
        "text": Value("string"),
        "repo": Value("string"),
        "path": Value("string"),
        "language": Value("string"),
        "sha": Value("string"),
        "chunk_index": Value("int32"),
        "n_tokens": Value("int32"),
    }
)


@dataclass(frozen=True)
class QualitySettings:
    drop_tests: bool
    min_history_commits: int
    min_lifetime_days: int
    min_score: int
    history_weight: int
    lifetime_weight: int
    personal_weight: int


@dataclass(frozen=True)
class BuildSettings:
    variant: str
    repository_root: Path
    output_root: Path
    max_rows: int
    chunk_tokens: int
    overlap_tokens: int
    near_threshold: float
    minhash_permutations: int
    shingle_tokens: int
    validation_ratio: float
    seed: int
    emails: tuple[str, ...]
    rules: FilterRules
    quality: QualitySettings
    dataset_license: str
    dataset_languages: tuple[str, ...]
    min_validation_repos: int = 1
    hub_repo_id: str = "JulianAT/personal-codex-model"
    dataset_license_holder: str = "JulianAT"
    dataset_license_year: int = 2026


@dataclass(frozen=True)
class History:
    commits: int
    lifetime_days: int
    personally_authored: bool


SECRET_PATTERNS = (
    (
        "private_key",
        re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----"),
    ),
    (
        "github_token",
        re.compile(r"(?<![A-Za-z0-9_])(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{40,})"),
    ),
    ("aws_access_key", re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])")),
    ("slack_token", re.compile(r"(?<![A-Za-z0-9])xox[baprs]-[A-Za-z0-9-]{20,}")),
    ("google_api_key", re.compile(r"(?<![A-Za-z0-9_-])AIza[A-Za-z0-9_-]{35}")),
    (
        "openai_key",
        re.compile(r"(?<![A-Za-z0-9_-])sk-(?:proj-)?[A-Za-z0-9_-]{32,}"),
    ),
    (
        "jwt",
        re.compile(
            r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}"
        ),
    ),
)


def detected_secret_kind(text: str) -> str | None:
    for name, pattern in SECRET_PATTERNS:
        if pattern.search(text):
            return name
    return None


class Deduper:
    def __init__(self, threshold: float, permutations: int, shingle_tokens: int, seed: int) -> None:
        self.exact_hashes: set[str] = set()
        self.lsh = MinHashLSH(threshold=threshold, num_perm=permutations)
        self.permutations = permutations
        self.shingle_tokens = shingle_tokens
        self.seed = seed
        self.inserted = 0

    def accept(self, text: str, sha: str) -> tuple[bool, str]:
        if sha in self.exact_hashes:
            return False, "exact_duplicate"
        signature = self._signature(text)
        if self.lsh.query(signature):
            return False, "near_duplicate"
        self.exact_hashes.add(sha)
        self.lsh.insert(str(self.inserted), signature)
        self.inserted += 1
        return True, "kept"

    def _signature(self, text: str) -> MinHash:
        tokens = content_tokens(text)
        width = self.shingle_tokens
        if len(tokens) >= width:
            shingles = {
                "\u241f".join(tokens[index : index + width])
                for index in range(len(tokens) - width + 1)
            }
        else:
            shingles = set(tokens) or {content_hash(text)}
        signature = MinHash(num_perm=self.permutations, seed=self.seed)
        for shingle in sorted(shingles):
            signature.update(shingle.encode("utf-8"))
        return signature


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def ratio(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 0 and less than 1")
    return parsed


def threshold(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError("must be greater than 0 and at most 1")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build leakage-safe Hugging Face code datasets.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--variant", choices=["all", "raw-max", "quality"])
    parser.add_argument("--repo-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-rows", type=bounded_rows)
    parser.add_argument("--max-file-bytes", type=positive_int)
    parser.add_argument("--chunk-tokens", type=positive_int)
    parser.add_argument("--chunk-overlap-tokens", type=nonnegative_int)
    parser.add_argument("--near-dedup-threshold", type=threshold)
    parser.add_argument("--minhash-permutations", type=positive_int)
    parser.add_argument("--shingle-tokens", type=positive_int)
    parser.add_argument("--validation-ratio", type=ratio)
    parser.add_argument("--extensions", help="Comma-separated extension allowlist, such as .py,.ts")
    parser.add_argument("--emails", action="append", help="Author email or comma-separated aliases")
    parser.add_argument("--drop-tests", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--quality-min-history-commits", type=nonnegative_int)
    parser.add_argument("--quality-min-lifetime-days", type=nonnegative_int)
    parser.add_argument("--quality-min-score", type=nonnegative_int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def choose(cli_value: Any, config: Mapping[str, Any], key: str, default: Any) -> Any:
    return cli_value if cli_value is not None else config_value(config, key, default)


def settings_for(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    root: Path,
    variant: str,
) -> BuildSettings:
    extensions = None
    if args.extensions:
        extensions = [
            item.strip().casefold() for item in args.extensions.split(",") if item.strip()
        ]
    max_file_bytes = int(choose(args.max_file_bytes, config, "dataset.max_file_bytes", 1_048_576))
    if max_file_bytes <= 0:
        raise SystemExit("dataset.max_file_bytes must be greater than zero")
    rules = FilterRules.from_config(
        config,
        max_file_bytes=max_file_bytes,
        extensions=extensions,
    )
    quality_config = config_value(config, "dataset.quality", {})
    quality = QualitySettings(
        drop_tests=bool(choose(args.drop_tests, config, "dataset.quality.drop_tests", True)),
        min_history_commits=int(
            choose(
                args.quality_min_history_commits, config, "dataset.quality.min_history_commits", 2
            )
        ),
        min_lifetime_days=int(
            choose(args.quality_min_lifetime_days, config, "dataset.quality.min_lifetime_days", 30)
        ),
        min_score=int(choose(args.quality_min_score, config, "dataset.quality.min_score", 2)),
        history_weight=int(quality_config.get("history_weight", 1)),
        lifetime_weight=int(quality_config.get("lifetime_weight", 1)),
        personal_weight=int(quality_config.get("personal_weight", 2)),
    )
    if any(
        value < 0
        for value in (
            quality.min_history_commits,
            quality.min_lifetime_days,
            quality.min_score,
            quality.history_weight,
            quality.lifetime_weight,
            quality.personal_weight,
        )
    ):
        raise SystemExit("dataset quality thresholds and weights cannot be negative")
    repository_root = resolved_path(
        root, args.repo_dir or config_value(config, "paths.repos", "repos")
    )
    output_root = resolved_path(
        root, args.output_dir or config_value(config, "paths.datasets", "datasets")
    )
    chunk_tokens = int(choose(args.chunk_tokens, config, "dataset.chunk_tokens", 2048))
    overlap_tokens = int(
        choose(args.chunk_overlap_tokens, config, "dataset.chunk_overlap_tokens", 128)
    )
    if chunk_tokens <= 0 or overlap_tokens < 0:
        raise SystemExit("chunk size must be positive and overlap cannot be negative")
    if overlap_tokens >= chunk_tokens:
        raise SystemExit("chunk overlap must be smaller than chunk size")
    max_rows = int(choose(args.max_rows, config, "dataset.max_rows", 100_000))
    if not 0 < max_rows <= 1_000_000:
        raise SystemExit("dataset.max_rows must be between 1 and 1,000,000")
    near_threshold = float(
        choose(args.near_dedup_threshold, config, "dataset.near_dedup_threshold", 0.85)
    )
    if not 0 < near_threshold <= 1:
        raise SystemExit("near-dedup threshold must be greater than 0 and at most 1")
    validation_ratio = float(
        choose(args.validation_ratio, config, "dataset.validation_ratio", 0.05)
    )
    if not 0 <= validation_ratio < 1:
        raise SystemExit("validation ratio must be at least 0 and less than 1")
    permutations = int(
        choose(args.minhash_permutations, config, "dataset.minhash_permutations", 128)
    )
    shingle_tokens = int(choose(args.shingle_tokens, config, "dataset.shingle_tokens", 5))
    if permutations <= 0 or shingle_tokens <= 0:
        raise SystemExit("MinHash permutations and shingle tokens must be greater than zero")
    return BuildSettings(
        variant=variant,
        repository_root=repository_root,
        output_root=output_root,
        max_rows=max_rows,
        chunk_tokens=chunk_tokens,
        overlap_tokens=overlap_tokens,
        near_threshold=near_threshold,
        minhash_permutations=permutations,
        shingle_tokens=shingle_tokens,
        validation_ratio=validation_ratio,
        seed=int(choose(args.seed, config, "project.seed", 42)),
        emails=tuple(configured_emails(config, args.emails)),
        rules=rules,
        quality=quality,
        dataset_license=str(config_value(config, "project.dataset_license", "other")),
        dataset_languages=tuple(config_value(config, "project.dataset_languages", ["en"])),
        min_validation_repos=int(config_value(config, "dataset.min_validation_repos", 1)),
        hub_repo_id=str(config_value(config, "hub.repo_id", "JulianAT/personal-codex-model")),
        dataset_license_holder=str(
            config_value(config, "project.dataset_license_holder", "JulianAT")
        ),
        dataset_license_year=int(config_value(config, "project.dataset_license_year", 2026)),
    )


def walk_repository(
    repository: Path, rules: FilterRules
) -> tuple[list[tuple[Path, Path, str]], Counter[str]]:
    candidates: list[tuple[Path, Path, str]] = []
    excluded: Counter[str] = Counter()
    for directory, directory_names, filenames in os.walk(repository, followlinks=False):
        root = Path(directory)
        kept_directories: list[str] = []
        for name in sorted(directory_names, key=str.casefold):
            folded = name.casefold()
            path = root / name
            if path.is_symlink():
                excluded["symlink_directory"] += 1
            elif folded in rules.excluded_directories:
                excluded["excluded_directory"] += 1
            elif folded in rules.vendored_generated_directories:
                excluded["vendored_or_generated"] += 1
            else:
                kept_directories.append(name)
        directory_names[:] = kept_directories
        for name in sorted(filenames, key=str.casefold):
            path = root / name
            if path.is_symlink():
                excluded["symlink"] += 1
                continue
            relative = path.relative_to(repository)
            try:
                size = path.stat().st_size
            except OSError:
                excluded["stat_error"] += 1
                continue
            decision = rules.decide(relative, size)
            if not decision.keep or decision.language is None:
                excluded[decision.reason] += 1
                continue
            candidates.append((path, relative, decision.language))
    ignored = gitignored_paths(repository, [relative for _, relative, _ in candidates])
    kept = []
    for path, relative, language in candidates:
        if relative.as_posix() in ignored:
            excluded["gitignored"] += 1
        else:
            kept.append((path, relative, language))
    return kept, excluded


def read_source(path: Path) -> tuple[str | None, str | None]:
    try:
        payload = path.read_bytes()
    except OSError:
        return None, "read_error"
    if not payload:
        return None, "empty"
    if b"\0" in payload[:8192]:
        return None, "binary"
    try:
        return payload.decode("utf-8"), None
    except UnicodeDecodeError:
        return None, "non_utf8"


def chunk_source(text: str, target_tokens: int, overlap_tokens: int) -> Iterable[tuple[str, int]]:
    matches = list(content_token_matches(text))
    if not matches:
        return
    start = 0
    while start < len(matches):
        end = min(start + target_tokens, len(matches))
        character_start = 0 if start == 0 else matches[start][0]
        character_end = len(text) if end == len(matches) else matches[end][0]
        chunk = text[character_start:character_end]
        if chunk.strip():
            yield chunk, end - start
        if end == len(matches):
            break
        start = end - overlap_tokens


def content_token_matches(text: str) -> Iterable[tuple[int, int]]:
    from corpuslib import TOKEN_PATTERN

    return ((match.start(), match.end()) for match in TOKEN_PATTERN.finditer(text))


def file_history(repository: Path, relative: Path, emails: set[str]) -> History:
    result = subprocess.run(
        [
            "git",
            "log",
            "--all",
            "--follow",
            "--format=%H%x09%aI%x09%ae",
            "--",
            relative.as_posix(),
        ],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        LOG.warning("History lookup failed for %s/%s", repository.name, relative)
        return History(0, 0, False)
    rows = [line.split("\t", 2) for line in result.stdout.splitlines() if line.strip()]
    rows = [row for row in rows if len(row) == 3]
    dates: list[datetime] = []
    personally_authored = False
    for _, raw_date, email in rows:
        try:
            dates.append(datetime.fromisoformat(raw_date))
        except ValueError:
            continue
        personally_authored = personally_authored or email.casefold() in emails
    lifetime = (max(dates) - min(dates)).days if len(dates) > 1 else 0
    return History(len(rows), lifetime, personally_authored)


def quality_score(history: History, settings: QualitySettings) -> int:
    score = 0
    if history.commits >= settings.min_history_commits:
        score += settings.history_weight
    if history.lifetime_days >= settings.min_lifetime_days:
        score += settings.lifetime_weight
    if history.personally_authored:
        score += settings.personal_weight
    return score


def collect_rows(settings: BuildSettings) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    repositories = discover_repositories(settings.repository_root)
    deduper = Deduper(
        settings.near_threshold,
        settings.minhash_permutations,
        settings.shingle_tokens,
        settings.seed,
    )
    rows: list[dict[str, Any]] = []
    filters: Counter[str] = Counter()
    dedup: Counter[str] = Counter()
    files_considered = 0
    files_selected = 0
    chunks_considered = 0
    cap_reached = False
    for repo_index, repository in enumerate(repositories, start=1):
        LOG.info("[%d/%d] scanning %s", repo_index, len(repositories), repository.name)
        files, repository_filters = walk_repository(repository, settings.rules)
        filters.update(repository_filters)
        for path, relative, language in files:
            files_considered += 1
            if settings.variant == "quality":
                if settings.quality.drop_tests and is_test_path(relative):
                    filters["quality_test_or_fixture"] += 1
                    continue
                history = file_history(repository, relative, set(settings.emails))
                if quality_score(history, settings.quality) < settings.quality.min_score:
                    filters["quality_score"] += 1
                    continue
            text, read_error = read_source(path)
            if read_error or text is None:
                filters[read_error or "read_error"] += 1
                continue
            secret_kind = detected_secret_kind(text)
            if secret_kind:
                filters[f"secret_{secret_kind}"] += 1
                continue
            files_selected += 1
            for chunk_index, (chunk, token_count) in enumerate(
                chunk_source(text, settings.chunk_tokens, settings.overlap_tokens)
            ):
                chunks_considered += 1
                sha = content_hash(chunk)
                accepted, reason = deduper.accept(chunk, sha)
                if not accepted:
                    dedup[reason] += 1
                    continue
                rows.append(
                    {
                        "text": chunk,
                        "repo": repository.name,
                        "path": relative.as_posix(),
                        "language": language,
                        "sha": sha,
                        "chunk_index": chunk_index,
                        "n_tokens": token_count,
                    }
                )
                if len(rows) % 1000 == 0:
                    LOG.info("kept %s rows", f"{len(rows):,}")
                if len(rows) >= settings.max_rows:
                    cap_reached = True
                    break
            if cap_reached:
                break
        if cap_reached:
            break
    audit = {
        "repositories_discovered": len(repositories),
        "repositories_with_rows": len({row["repo"] for row in rows}),
        "files_considered": files_considered,
        "files_selected": files_selected,
        "chunks_considered": chunks_considered,
        "filter_report": dict(sorted(filters.items())),
        "dedup_report": {
            "exact_duplicates_dropped": dedup["exact_duplicate"],
            "near_duplicates_dropped": dedup["near_duplicate"],
            "rows_kept": len(rows),
        },
        "max_rows": settings.max_rows,
        "cap_reached": cap_reached,
    }
    LOG.info(
        "dedup kept %s/%s chunks; exact drops %s; near drops %s",
        f"{len(rows):,}",
        f"{chunks_considered:,}",
        f"{dedup['exact_duplicate']:,}",
        f"{dedup['near_duplicate']:,}",
    )
    if filters:
        LOG.info(
            "file/filter drops: %s",
            ", ".join(f"{reason}={count:,}" for reason, count in sorted(filters.items())),
        )
    return rows, audit


def token_histogram(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    boundaries = [0, 256, 512, 1024, 1536, 2048, 3072, math.inf]
    counts: Counter[str] = Counter()
    for row in rows:
        tokens = int(row["n_tokens"])
        for lower, upper in zip(boundaries, boundaries[1:], strict=True):
            if lower <= tokens < upper:
                label = (
                    f"{int(lower)}-{int(upper) - 1}" if math.isfinite(upper) else f"{int(lower)}+"
                )
                counts[label] += 1
                break
    return dict(counts)


def line_statistics(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    total = 0
    nonblank = 0
    for row in rows:
        lines = str(row["text"]).splitlines()
        total += len(lines)
        nonblank += sum(bool(line.strip()) for line in lines)
    return {"total": total, "nonblank": nonblank}


def size_category(rows: int) -> str:
    if rows < 1_000:
        return "n<1K"
    if rows < 10_000:
        return "1K<n<10K"
    if rows < 100_000:
        return "10K<n<100K"
    if rows < 1_000_000:
        return "100K<n<1M"
    return "1M<n<10M"


def clean_known_outputs(output: Path) -> None:
    for directory in (output / "arrow", output / "data"):
        if directory.exists():
            shutil.rmtree(directory)
    for name in (
        "README.md",
        "LICENSE",
        "dataset_infos.json",
        "statistics.json",
        "train.jsonl",
        "valid.jsonl",
    ):
        path = output / name
        if path.exists():
            path.unlink()


def dataset_from_rows(rows: Sequence[dict[str, Any]]) -> Dataset:
    if rows:
        return Dataset.from_list(list(rows), features=FEATURES)
    return Dataset.from_dict({name: [] for name in FEATURES}, features=FEATURES)


def write_artifacts(
    rows: list[dict[str, Any]],
    audit: dict[str, Any],
    settings: BuildSettings,
) -> dict[str, Any]:
    output = settings.output_root / settings.variant
    output.mkdir(parents=True, exist_ok=True)
    clean_known_outputs(output)
    assignments = repository_split(
        (str(row["repo"]) for row in rows),
        settings.validation_ratio,
        settings.seed,
        settings.min_validation_repos,
    )
    train_rows = [row for row in rows if assignments[str(row["repo"])] == "train"]
    valid_rows = [row for row in rows if assignments[str(row["repo"])] == "valid"]
    dataset = DatasetDict(
        {
            "train": dataset_from_rows(train_rows),
            "valid": dataset_from_rows(valid_rows),
        }
    )
    dataset.save_to_disk(output / "arrow")
    data_directory = output / "data"
    data_directory.mkdir(parents=True, exist_ok=True)
    dataset["train"].to_parquet(data_directory / "train-00000-of-00001.parquet")
    dataset["valid"].to_parquet(data_directory / "valid-00000-of-00001.parquet")
    dataset["train"].to_json(output / "train.jsonl", orient="records", lines=True)
    dataset["valid"].to_json(output / "valid.jsonl", orient="records", lines=True)
    split_statistics = {
        name: {"num_examples": len(split), "num_bytes": int(split.data.nbytes)}
        for name, split in dataset.items()
    }
    loader_splits = {
        name: statistics
        for name, statistics in split_statistics.items()
        if statistics["num_examples"] > 0
    }
    parquet_bytes = sum(path.stat().st_size for path in data_directory.glob("*.parquet"))
    feature_card = [
        {"name": name, "dtype": str(feature.dtype)} for name, feature in FEATURES.items()
    ]
    metadata = {
        "pretty_name": "Personal Codex Model Training Corpus",
        "license": settings.dataset_license,
        "language": list(settings.dataset_languages),
        "annotations_creators": ["no-annotation"],
        "language_creators": ["found"],
        "source_datasets": ["original"],
        "size_categories": [size_category(len(rows))],
        "task_categories": ["text-generation"],
        "task_ids": ["language-modeling"],
        "tags": [
            "code",
            "text",
            "source-code",
            "code-completion",
            "code-generation",
            "causal-language-modeling",
            "continued-pretraining",
            "fine-tuning",
            "coding-assistant",
            "software-engineering",
            "repository-level",
            "multilingual-code",
            "deduplicated",
            "provenance-aware",
            "parquet",
            "datasets",
            "typescript",
            "python",
            "javascript",
        ],
        "configs": [
            {
                "config_name": "default",
                "data_files": [
                    {"split": name, "path": f"data/{name}-*.parquet"} for name in loader_splits
                ],
            }
        ],
        "dataset_info": {
            "features": feature_card,
            "splits": [{"name": name, **statistics} for name, statistics in loader_splits.items()],
        },
    }
    write_text(output / "README.md", dataset_card(settings, rows, audit, metadata))
    if settings.dataset_license.casefold() == "mit":
        write_text(output / "LICENSE", mit_license(settings))
    dataset_info = {
        "default": {
            "builder_name": "parquet",
            "dataset_name": settings.hub_repo_id.rsplit("/", 1)[-1],
            "config_name": "default",
            "description": "Repository-walked personal code chunks for completion training.",
            "license": settings.dataset_license,
            "features": FEATURES.to_dict(),
            "splits": {
                name: {"name": name, **statistics} for name, statistics in loader_splits.items()
            },
            "download_size": parquet_bytes,
            "dataset_size": sum(item["num_bytes"] for item in split_statistics.values()),
            "size_in_bytes": parquet_bytes
            + sum(item["num_bytes"] for item in split_statistics.values()),
        }
    }
    write_json(output / "dataset_infos.json", dataset_info)
    statistics = {
        "generated_at": utc_now(),
        "variant": settings.variant,
        "sources": {
            "repository_root": "local repository checkouts (not published)",
            "config": "config.yaml",
        },
        "parameters": {
            "max_rows": settings.max_rows,
            "max_file_bytes": settings.rules.max_file_bytes,
            "chunk_tokens": settings.chunk_tokens,
            "chunk_overlap_tokens": settings.overlap_tokens,
            "near_dedup_threshold": settings.near_threshold,
            "minhash_permutations": settings.minhash_permutations,
            "shingle_tokens": settings.shingle_tokens,
            "validation_ratio": settings.validation_ratio,
            "min_validation_repos": settings.min_validation_repos,
            "seed": settings.seed,
        },
        "rows": {"total": len(rows), "train": len(train_rows), "valid": len(valid_rows)},
        "tokens": {
            "total": sum(int(row["n_tokens"]) for row in rows),
            "histogram": token_histogram(rows),
        },
        "lines": line_statistics(rows),
        "utf8_bytes": sum(len(str(row["text"]).encode("utf-8")) for row in rows),
        "source_files": len({(str(row["repo"]), str(row["path"])) for row in rows}),
        "by_language": dict(sorted(Counter(str(row["language"]) for row in rows).items())),
        "by_repo": dict(sorted(Counter(str(row["repo"]) for row in rows).items())),
        "repo_split": dict(sorted(assignments.items())),
        **audit,
        "self_test": {"status": "pending"},
    }
    write_json(output / "statistics.json", statistics)
    if not rows:
        statistics["self_test"] = {
            "status": "skipped",
            "reason": (
                "No retained rows; Hugging Face cannot materialize a declared zero-row split."
            ),
        }
        write_json(output / "statistics.json", statistics)
        return statistics
    loaded = load_dataset(str(output))
    if set(loaded) != set(loader_splits):
        raise AssertionError(f"self-test loaded unexpected splits: {sorted(loaded)}")
    for name, split in loaded.items():
        if split.features != FEATURES:
            raise AssertionError(f"self-test loaded an unexpected feature schema for {name}")
        if len(split) != split_statistics[name]["num_examples"]:
            raise AssertionError(f"self-test loaded an unexpected row count for {name}")
    statistics["self_test"] = {
        "status": "passed",
        "loader": f'datasets.load_dataset("{output}")',
        "splits": {name: len(split) for name, split in loaded.items()},
    }
    write_json(output / "statistics.json", statistics)
    return statistics


def dataset_card(
    settings: BuildSettings,
    rows: Sequence[dict[str, Any]],
    audit: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> str:
    lines = line_statistics(rows)
    utf8_mib = sum(len(str(row["text"]).encode("utf-8")) for row in rows) / (1024 * 1024)
    source_files = len({(str(row["repo"]), str(row["path"])) for row in rows})
    lexical_tokens = sum(int(row["n_tokens"]) for row in rows)
    language_counts = Counter(str(row["language"]) for row in rows)
    language_table = "\n".join(
        f"| {language} | {count:,} | {count / len(rows):.1%} |"
        for language, count in language_counts.most_common()
    )
    split_counts = {
        str(item["name"]): int(item["num_examples"]) for item in metadata["dataset_info"]["splits"]
    }
    frontmatter = yaml.safe_dump(
        dict(metadata),
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    ).strip()
    quality_rules = ""
    if settings.variant == "quality":
        quality_rules = f"""
### Quality selection

The quality variant {"excludes" if settings.quality.drop_tests else "retains"} test and fixture
paths. Files are scored from commit count, repository lifetime, and configured author matches. The
minimum accepted score is {settings.quality.min_score}. Author email values remain local and are not
published.
"""
    if settings.dataset_license.casefold() == "mit":
        license_text = """The packaged dataset is released under the MIT License. The included
`LICENSE` file contains the complete terms. This dataset-level license does not supersede separate
licenses, notices, or obligations that may apply to code from contributing repositories. Users are
responsible for source-specific compliance when redistributing code, releasing trained models, or
using generated output."""
    else:
        license_text = """The dataset uses the Hugging Face `other` license classification because
no single dataset-wide license supersedes the licenses and obligations of contributing repositories.
Users are responsible for source-specific compliance before redistribution, commercial use, model
release, or generated-code reuse."""
    return f"""---
{frontmatter}
---

# Personal Codex Model Training Corpus

## Overview

Personal Codex Model Training Corpus is a provenance-aware, repository-level dataset for causal
language modeling, code completion, continued pretraining, and coding assistant adaptation. It is
built from source files present in local Git repository checkouts at a defined collection point.

The dataset prioritizes broad, authentic software-engineering coverage while retaining enough
metadata to audit every emitted chunk. It is not an instruction dataset, benchmark, or collection
of verified solutions. Each record represents source text as it existed in a repository checkout,
after filtering, chunking, secret screening, and global deduplication.

## Dataset profile

| Metric | Value |
| --- | ---: |
| Total examples | {len(rows):,} |
| Training examples | {split_counts.get("train", 0):,} |
| Validation examples | {split_counts.get("valid", 0):,} |
| Emitted lines | {lines["total"]:,} |
| Nonblank emitted lines | {lines["nonblank"]:,} |
| Approximate lexical tokens | {lexical_tokens:,} |
| UTF-8 source text | {utf8_mib:,.1f} MiB |
| Source files with retained chunks | {source_files:,} |
| Repositories with retained rows | {audit["repositories_with_rows"]:,} |

Line, byte, and token totals measure emitted training chunks. The configured chunk overlap can
repeat text at chunk boundaries. These figures describe training volume, not unique repository
lines of code or model-tokenizer counts.

## Language and format distribution

The `language` value is assigned from a controlled extension and exact-filename mapping. Markdown,
configuration, schema, and build-system files are retained because they are part of real software
engineering workflows and frequently contain executable examples or machine-consumed structure.

| Language or format | Examples | Share |
| --- | ---: | ---: |
{language_table}

## Intended uses

Appropriate uses include:

- continued pretraining or domain adaptation of causal language models
- code completion and repository-aware coding assistant experiments
- tokenizer, chunking, deduplication, and corpus composition research
- retrieval and provenance experiments using repository and path metadata
- controlled studies of personalization on repository-disjoint validation data

The dataset is not suitable as a correctness benchmark, a secure-code reference, a software
license classifier, or evidence of authorship and repository ownership.

## Load the dataset

Install a compatible version of `datasets`, then load the full corpus:

```python
from datasets import load_dataset

dataset = load_dataset("{settings.hub_repo_id}")
print(dataset)
print(dataset["train"].features)
```

Stream examples without downloading the complete dataset:

```python
from datasets import load_dataset

stream = load_dataset("{settings.hub_repo_id}", split="train", streaming=True)
first_example = next(iter(stream))
```

## Schema

| Field | Type | Description |
| --- | --- | --- |
| `text` | string | Source-code or repository-text chunk used as the modeling target. |
| `repo` | string | Source repository name at collection time. |
| `path` | string | Repository-relative source path. |
| `language` | string | Language or format inferred from the configured mapping. |
| `sha` | string | SHA-256 digest of the emitted `text`. |
| `chunk_index` | int32 | Zero-based chunk position within the source file. |
| `n_tokens` | int32 | Tokenizer-independent lexical token estimate. |

## Dataset construction

The builder applies the following deterministic pipeline:

1. Discover configured Git repository checkouts.
2. Walk supported source, documentation, schema, configuration, and build files.
3. Exclude ignored, sensitive, generated, vendored, binary, oversized, and unsupported content.
4. Decode retained files as UTF-8 and reject unreadable or empty payloads.
5. Reject complete files containing high-confidence credential signatures.
6. Chunk source text to approximately {settings.chunk_tokens:,} lexical tokens with an overlap of
   {settings.overlap_tokens:,} lexical tokens.
7. Remove exact duplicate chunks by SHA-256.
8. Remove near-duplicate chunks with MinHash LSH.
9. Assign repositories, rather than individual rows, to deterministic train and validation splits.

This repository-level split prevents a source repository from appearing in both splits. It reduces
direct leakage from repeated project structure and repository-specific conventions.
{quality_rules}
## Deduplication and quality controls

Near-duplicate detection uses `datasketch.MinHashLSH` with
{settings.minhash_permutations} permutations, token {settings.shingle_tokens}-grams, and a Jaccard
threshold of {settings.near_threshold}. The current build retained
{audit["dedup_report"]["rows_kept"]:,} chunks after dropping
{audit["dedup_report"]["exact_duplicates_dropped"]:,} exact duplicates and
{audit["dedup_report"]["near_duplicates_dropped"]:,} near duplicates.

The file walk excludes Git metadata, ignored paths, dependency and environment directories, build
outputs, vendored and generated directories, lockfiles, minified files, symlinks, binary or
non-UTF-8 payloads, files above {settings.rules.max_file_bytes:,} bytes, and unsupported formats.
Credential screening covers high-confidence private-key, platform-token, cloud-key, API-key, and
JWT patterns.

These controls reduce common leakage and duplication risks. They do not constitute a formal proof
that every row is safe, original, correct, or free of sensitive information.

## Provenance, privacy, and licensing

Every record retains repository, path, language, chunk position, and content-hash metadata. This
supports traceability inside the published corpus without publishing local checkout locations or
builder credentials.

Some contributing repositories were private at collection time. Public publication was explicitly
enabled by the dataset maintainer. Users should still treat repository names, paths, comments, and
source text as potentially identifying information.

{license_text}

## Limitations

- Source is collected from working-tree snapshots, not from deleted Git history.
- Repository contents can include incomplete, insecure, outdated, experimental, or generated-like
  code that survives the configured filters.
- Extension-based language labels do not perform parser-level language verification.
- Lexical token estimates are not equivalent to tokens from a production model tokenizer.
- MinHash is approximate and can retain related text or remove independently written similar text.
- Chunk overlap increases emitted volume and can repeat boundary lines.
- The dataset contains no correctness, security, quality, preference, or authorship labels.
- Repository-disjoint validation measures transfer across included repositories, not general coding
  ability across unrelated ecosystems.

## Reproducibility and audit artifacts

`statistics.json` records build parameters, split assignments, row and token counts, language
distribution, filter decisions, and deduplication totals. `dataset_infos.json` records the feature
schema and split sizes. The Parquet shards are the canonical Hub loader source.

The Hub publication intentionally omits local Arrow and JSONL copies because they duplicate the
Parquet payload. It also omits source checkouts, local filesystem paths, author-email configuration,
and training artifacts.

## Citation

```bibtex
@misc{{personal_codex_model_training_corpus,
  author       = {{JulianAT}},
  title        = {{Personal Codex Model Training Corpus}},
  year         = {{2026}},
  howpublished = {{Hugging Face Datasets}},
  url          = {{https://huggingface.co/datasets/{settings.hub_repo_id}}}
}}
```
"""


def mit_license(settings: BuildSettings) -> str:
    return f"""MIT License

Copyright (c) {settings.dataset_license_year} {settings.dataset_license_holder}

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def selected_variants(args: argparse.Namespace, config: Mapping[str, Any]) -> list[str]:
    if args.variant and args.variant != "all":
        return [args.variant]
    configured = [
        str(value) for value in config_value(config, "dataset.variants", ["raw-max", "quality"])
    ]
    if not configured:
        raise SystemExit("dataset.variants cannot be empty")
    if len(configured) != len(set(configured)):
        raise SystemExit("dataset.variants must be unique")
    invalid = sorted(set(configured) - {"raw-max", "quality"})
    if invalid:
        raise SystemExit(f"Unsupported configured variants: {', '.join(invalid)}")
    return configured


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)
    config, root = load_config(args.config)
    variants = selected_variants(args, config)
    summaries: list[tuple[str, Any]] = []
    failures = 0
    for variant in variants:
        LOG.info("building dataset variant %s", variant)
        settings = settings_for(args, config, root, variant)
        try:
            rows, audit = collect_rows(settings)
            statistics = write_artifacts(rows, audit, settings)
        except Exception:
            LOG.exception("dataset variant %s failed", variant)
            failures += 1
            summaries.append((variant, "failed"))
            continue
        summaries.append(
            (
                variant,
                f"{statistics['rows']['total']:,} rows "
                f"({statistics['rows']['train']:,} train / {statistics['rows']['valid']:,} valid)",
            )
        )
    final_summary("Dataset build", summaries + [("Failures", failures)])
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
