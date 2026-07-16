from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

LOG = logging.getLogger("code-corpus")
TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+(?:\.\d+)?|[^\w\s]", re.UNICODE)


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).expanduser().resolve()
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise SystemExit(f"Config file not found: {config_path}") from error
    if not isinstance(loaded, dict):
        raise SystemExit(f"Config root must be a mapping: {config_path}")
    return loaded, config_path.parent


def config_value(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    value: Any = config
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def resolved_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    rendered = [[str(cell).replace("|", "\\|").replace("\n", " ") for cell in row] for row in rows]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rendered)
    return "\n".join(lines)


def print_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    materialized = [[str(cell) for cell in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in materialized:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in materialized:
        print("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)))


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-.").lower()
    return cleaned or "unnamed"


def discover_repositories(repo_root: Path, max_repos: int | None = None) -> list[Path]:
    if not repo_root.exists():
        return []
    repositories = [
        path
        for path in sorted(repo_root.iterdir(), key=lambda item: item.name.casefold())
        if path.is_dir() and ((path / ".git").is_dir() or (path / ".git").is_file())
    ]
    return repositories if max_repos is None else repositories[:max_repos]


def configured_emails(
    config: Mapping[str, Any], cli_emails: Sequence[str] | None = None
) -> list[str]:
    candidates: list[str] = []
    for value in config_value(config, "stats.emails", []) or []:
        candidates.extend(str(value).split(","))
    for value in cli_emails or []:
        candidates.extend(value.split(","))
    try:
        result = subprocess.run(
            ["git", "config", "--get", "user.email"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            candidates.append(result.stdout.strip())
    except FileNotFoundError:
        pass
    return sorted({email.strip().casefold() for email in candidates if email.strip()})


def content_tokens(text: str) -> list[str]:
    return [match.group(0) for match in TOKEN_PATTERN.finditer(text)]


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def repository_split(
    repositories: Iterable[str],
    validation_ratio: float,
    seed: int,
    minimum_validation: int = 1,
) -> dict[str, str]:
    names = sorted(set(repositories))
    if len(names) < 2 or validation_ratio <= 0:
        return {name: "train" for name in names}
    validation_count = max(minimum_validation, round(len(names) * validation_ratio))
    validation_count = min(validation_count, len(names) - 1)
    ordered = sorted(
        names,
        key=lambda name: hashlib.sha256(f"{seed}:{name}".encode()).digest(),
    )
    validation = set(ordered[:validation_count])
    return {name: "valid" if name in validation else "train" for name in names}


@dataclass(frozen=True)
class FilterDecision:
    keep: bool
    reason: str
    language: str | None = None


@dataclass(frozen=True)
class FilterRules:
    extensions: Mapping[str, str]
    filenames: Mapping[str, str]
    excluded_directories: frozenset[str]
    vendored_generated_directories: frozenset[str]
    excluded_filenames: frozenset[str]
    generated_globs: tuple[str, ...]
    sensitive_globs: tuple[str, ...]
    max_file_bytes: int

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        max_file_bytes: int | None = None,
        extensions: Sequence[str] | None = None,
    ) -> FilterRules:
        configured = config_value(config, "dataset.extensions", {})
        if not isinstance(configured, Mapping):
            raise SystemExit("dataset.extensions must be a mapping")
        normalized = {str(key).casefold(): str(value) for key, value in configured.items()}
        configured_filenames = config_value(config, "dataset.filenames", {})
        if not isinstance(configured_filenames, Mapping):
            raise SystemExit("dataset.filenames must be a mapping")
        normalized_filenames = {
            str(key).casefold(): str(value) for key, value in configured_filenames.items()
        }
        if extensions:
            selected = {
                extension.casefold() if extension.startswith(".") else f".{extension.casefold()}"
                for extension in extensions
            }
            unknown = selected - set(normalized)
            if unknown:
                raise SystemExit(f"Unknown configured extensions: {', '.join(sorted(unknown))}")
            normalized = {key: value for key, value in normalized.items() if key in selected}
            normalized_filenames = {}
        configured_bytes = int(config_value(config, "dataset.max_file_bytes", 1_048_576))
        resolved_bytes = max_file_bytes if max_file_bytes is not None else configured_bytes
        if resolved_bytes <= 0:
            raise SystemExit("dataset.max_file_bytes must be greater than zero")
        return cls(
            extensions=normalized,
            filenames=normalized_filenames,
            excluded_directories=frozenset(
                str(value).casefold()
                for value in config_value(config, "dataset.excluded_directories", [])
            ),
            vendored_generated_directories=frozenset(
                str(value).casefold()
                for value in config_value(config, "dataset.vendored_generated_directories", [])
            ),
            excluded_filenames=frozenset(
                str(value).casefold()
                for value in config_value(config, "dataset.excluded_filenames", [])
            ),
            generated_globs=tuple(
                str(value) for value in config_value(config, "dataset.generated_globs", [])
            ),
            sensitive_globs=tuple(
                str(value) for value in config_value(config, "dataset.sensitive_globs", [])
            ),
            max_file_bytes=resolved_bytes,
        )

    def decide(self, relative_path: Path, size: int | None = None) -> FilterDecision:
        parts = [part.casefold() for part in relative_path.parts[:-1]]
        if any(part in self.excluded_directories for part in parts):
            return FilterDecision(False, "excluded_directory")
        if any(part in self.vendored_generated_directories for part in parts):
            return FilterDecision(False, "vendored_or_generated")
        name = relative_path.name.casefold()
        folded_path = Path(relative_path.as_posix().casefold())
        if any(folded_path.match(pattern.casefold()) for pattern in self.sensitive_globs):
            return FilterDecision(False, "sensitive_path")
        if name in self.excluded_filenames:
            return FilterDecision(False, "lockfile")
        if ".min." in name:
            return FilterDecision(False, "minified")
        if any(folded_path.match(pattern.casefold()) for pattern in self.generated_globs):
            return FilterDecision(False, "generated_file")
        if size is not None and size > self.max_file_bytes:
            return FilterDecision(False, "too_large")
        suffix = relative_path.suffix.casefold()
        language = self.filenames.get(name, self.extensions.get(suffix))
        if language is None:
            return FilterDecision(False, "extension")
        return FilterDecision(True, "kept", language)


def is_test_path(path: Path) -> bool:
    test_directories = {"test", "tests", "fixture", "fixtures", "__tests__", "__snapshots__"}
    if any(part.casefold() in test_directories for part in path.parts[:-1]):
        return True
    name = path.name.casefold()
    stem = path.stem.casefold()
    return (
        stem.startswith("test_") or stem.endswith("_test") or ".test." in name or ".spec." in name
    )


def gitignored_paths(repository: Path, relative_paths: Sequence[Path]) -> set[str]:
    if not relative_paths:
        return set()
    payload = b"\0".join(os.fsencode(path.as_posix()) for path in relative_paths) + b"\0"
    try:
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", "-z", "--stdin"],
            cwd=repository,
            input=payload,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        return set()
    if result.returncode not in {0, 1}:
        LOG.warning(
            "git check-ignore failed for %s: %s",
            repository.name,
            result.stderr.decode(errors="replace"),
        )
        return set()
    return {os.fsdecode(value) for value in result.stdout.split(b"\0") if value}


def final_summary(stage: str, rows: Sequence[tuple[str, Any]]) -> None:
    print(f"\n{stage} summary")
    print_table(["Metric", "Value"], rows)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("must be greater than zero")
    return parsed


def bounded_rows(value: str) -> int:
    parsed = positive_int(value)
    if parsed > 1_000_000:
        raise ValueError("must be at most 1,000,000")
    return parsed
