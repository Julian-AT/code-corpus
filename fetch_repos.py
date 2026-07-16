from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from corpuslib import (
    LOG,
    config_value,
    configure_logging,
    final_summary,
    load_config,
    positive_int,
    print_table,
    resolved_path,
    utc_now,
    write_json,
)

AUTH_COMMAND = "gh auth login --hostname github.com --web"


@dataclass(frozen=True)
class FetchSettings:
    repository_root: Path
    include_private: bool
    include_forks: bool
    include_archived: bool
    since_year: int
    limit: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clone repositories from the authenticated GitHub account."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--repo-dir")
    parser.add_argument("--include-private", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--include-forks", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--include-archived", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--since-year", type=int)
    parser.add_argument("--limit", type=positive_int)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def choose(cli_value: Any, config: Mapping[str, Any], key: str, default: Any) -> Any:
    return cli_value if cli_value is not None else config_value(config, key, default)


def make_settings(args: argparse.Namespace, config: Mapping[str, Any], root: Path) -> FetchSettings:
    configured_year = config_value(config, "fetch.since_year")
    years_ago = int(config_value(config, "fetch.years_ago", 4))
    since_year = args.since_year or configured_year or (datetime.now().year - years_ago)
    if int(since_year) < 1970 or int(since_year) > datetime.now().year:
        raise SystemExit(f"since year must be between 1970 and {datetime.now().year}")
    limit = int(choose(args.limit, config, "fetch.limit", 1000))
    if limit <= 0:
        raise SystemExit("fetch.limit must be greater than zero")
    return FetchSettings(
        repository_root=resolved_path(
            root,
            args.repo_dir or config_value(config, "paths.repos", "repos"),
        ),
        include_private=bool(choose(args.include_private, config, "fetch.include_private", False)),
        include_forks=bool(choose(args.include_forks, config, "fetch.include_forks", False)),
        include_archived=bool(
            choose(args.include_archived, config, "fetch.include_archived", False)
        ),
        since_year=int(since_year),
        limit=limit,
    )


def ensure_authenticated() -> str:
    if shutil.which("gh") is None:
        raise SystemExit(
            f"GitHub CLI 'gh' was not found. After installing it, run:\n{AUTH_COMMAND}"
        )
    status = subprocess.run(
        ["gh", "auth", "status", "--hostname", "github.com"],
        capture_output=True,
        text=True,
        check=False,
    )
    if status.returncode != 0:
        raise SystemExit(f"GitHub CLI is not authenticated. Run exactly:\n{AUTH_COMMAND}")
    user = subprocess.run(
        ["gh", "api", "user", "--jq", ".login"],
        capture_output=True,
        text=True,
        check=False,
    )
    if user.returncode != 0 or not user.stdout.strip():
        raise SystemExit(
            f"Could not resolve the authenticated GitHub user. Run exactly:\n{AUTH_COMMAND}"
        )
    return user.stdout.strip()


def list_repositories(owner: str, settings: FetchSettings) -> list[dict[str, Any]]:
    fields = ",".join(
        [
            "name",
            "nameWithOwner",
            "isPrivate",
            "isFork",
            "isArchived",
            "primaryLanguage",
            "diskUsage",
            "pushedAt",
            "url",
        ]
    )
    command = [
        "gh",
        "repo",
        "list",
        owner,
        "--limit",
        str(settings.limit),
        "--json",
        fields,
    ]
    if not settings.include_private:
        command.extend(["--visibility", "public"])
    if not settings.include_forks:
        command.append("--source")
    if not settings.include_archived:
        command.append("--no-archived")
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "unknown gh error"
        raise SystemExit(f"gh repo list failed: {message}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise SystemExit("gh repo list returned invalid JSON") from error
    if not isinstance(payload, list):
        raise SystemExit("gh repo list returned an unexpected payload")
    return [item for item in payload if isinstance(item, dict)]


def pushed_year(repository: Mapping[str, Any]) -> int | None:
    raw = repository.get("pushedAt")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).astimezone(UTC).year
    except ValueError:
        return None


def select_repositories(
    repositories: list[dict[str, Any]],
    settings: FetchSettings,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    selected: list[dict[str, Any]] = []
    excluded = {"private": 0, "fork": 0, "archived": 0, "older": 0}
    for repository in repositories:
        if bool(repository.get("isPrivate")) and not settings.include_private:
            excluded["private"] += 1
            continue
        if bool(repository.get("isFork")) and not settings.include_forks:
            excluded["fork"] += 1
            continue
        if bool(repository.get("isArchived")) and not settings.include_archived:
            excluded["archived"] += 1
            continue
        year = pushed_year(repository)
        if year is None or year < settings.since_year:
            excluded["older"] += 1
            continue
        selected.append(repository)
    return selected[: settings.limit], excluded


def language_name(repository: Mapping[str, Any]) -> str:
    language = repository.get("primaryLanguage")
    if isinstance(language, Mapping):
        return str(language.get("name") or "unknown")
    return "unknown"


def human_size(kibibytes: Any) -> str:
    try:
        value = float(kibibytes or 0)
    except (TypeError, ValueError):
        return "unknown"
    if value < 1024:
        return f"{value:.0f} KiB"
    if value < 1024 * 1024:
        return f"{value / 1024:.1f} MiB"
    return f"{value / 1024 / 1024:.1f} GiB"


def clone_repository(repository: Mapping[str, Any], destination: Path) -> tuple[str, str | None]:
    git_marker = destination / ".git"
    if git_marker.exists():
        shallow = subprocess.run(
            ["git", "rev-parse", "--is-shallow-repository"],
            cwd=destination,
            capture_output=True,
            text=True,
            check=False,
        )
        if shallow.returncode == 0 and shallow.stdout.strip() == "true":
            return "failed", "existing clone is shallow; run git fetch --unshallow before stats"
        return "existing", None
    if destination.exists() and any(destination.iterdir()):
        return "failed", "destination exists and is not a git repository"
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["gh", "repo", "clone", str(repository["nameWithOwner"]), str(destination)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return "cloned", None
    return "failed", result.stderr.strip() or result.stdout.strip() or "clone failed"


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)
    config, root = load_config(args.config)
    settings = make_settings(args, config, root)
    owner = ensure_authenticated()
    LOG.info("listing repositories for %s", owner)
    listed = list_repositories(owner, settings)
    selected, excluded = select_repositories(listed, settings)
    settings.repository_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for index, repository in enumerate(selected, start=1):
        name = str(repository["name"])
        LOG.info("[%d/%d] %s", index, len(selected), name)
        status, error = clone_repository(repository, settings.repository_root / name)
        records.append(
            {
                **repository,
                "primaryLanguageName": language_name(repository),
                "localPath": str(settings.repository_root / name),
                "status": status,
                "error": error,
            }
        )
        if error:
            LOG.error("%s: %s", name, error)
    summary = {
        "generated_at": utc_now(),
        "owner": owner,
        "settings": {
            "include_private": settings.include_private,
            "include_forks": settings.include_forks,
            "include_archived": settings.include_archived,
            "since_year": settings.since_year,
            "limit": settings.limit,
        },
        "listed": len(listed),
        "selected": len(selected),
        "excluded": excluded,
        "repositories": records,
    }
    write_json(settings.repository_root / "repos.json", summary)
    print()
    print_table(
        ["Name", "Private?", "Primary language", "Size", "Status"],
        [
            [
                record["name"],
                "yes" if record.get("isPrivate") else "no",
                record["primaryLanguageName"],
                human_size(record.get("diskUsage")),
                record["status"],
            ]
            for record in records
        ],
    )
    cloned = sum(record["status"] == "cloned" for record in records)
    existing = sum(record["status"] == "existing" for record in records)
    failed = sum(record["status"] == "failed" for record in records)
    final_summary(
        "Repository fetch",
        [
            ("Listed", len(listed)),
            ("Selected", len(selected)),
            ("Cloned", cloned),
            ("Already present", existing),
            ("Failed", failed),
            ("Metadata", settings.repository_root / "repos.json"),
        ],
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
