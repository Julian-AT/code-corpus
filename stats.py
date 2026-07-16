from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from corpuslib import (
    LOG,
    FilterRules,
    config_value,
    configure_logging,
    configured_emails,
    discover_repositories,
    final_summary,
    gitignored_paths,
    load_config,
    markdown_table,
    positive_int,
    resolved_path,
    utc_now,
    write_json,
    write_text,
)

COMMIT_MARKER = "__CODE_CORPUS_COMMIT__"


@dataclass(frozen=True)
class FileChange:
    path: Path
    language: str
    additions: int
    deletions: int


@dataclass(frozen=True)
class AuthoredCommit:
    sha: str
    authored_at: datetime | None
    changes: tuple[FileChange, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute personal contribution statistics from git history."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--repo-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--emails", action="append", help="Author email or comma-separated aliases")
    parser.add_argument("--max-repos", type=positive_int)
    parser.add_argument("--max-file-bytes", type=positive_int)
    parser.add_argument("--extensions", help="Comma-separated extension allowlist")
    parser.add_argument("--chart-dpi", type=positive_int)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def choose(cli_value: Any, config: Mapping[str, Any], key: str, default: Any) -> Any:
    return cli_value if cli_value is not None else config_value(config, key, default)


def empty_totals() -> dict[str, int]:
    return {"commits": 0, "additions": 0, "deletions": 0, "net": 0, "changed_lines": 0}


def add_lines(bucket: dict[str, int], additions: int, deletions: int) -> None:
    bucket["additions"] += additions
    bucket["deletions"] += deletions
    bucket["net"] += additions - deletions
    bucket["changed_lines"] += additions + deletions


def decode_git_path(value: str) -> Path:
    if value.startswith('"') and value.endswith('"'):
        try:
            decoded = json.loads(value)
            if isinstance(decoded, str):
                return Path(decoded)
        except json.JSONDecodeError:
            pass
    return Path(value)


def destination_path(value: str) -> Path:
    decoded = decode_git_path(value).as_posix()
    if " => " not in decoded:
        return Path(decoded)
    opening = decoded.find("{")
    closing = decoded.find("}", opening + 1)
    if opening >= 0 and closing > opening:
        renamed = decoded[opening + 1 : closing].rsplit(" => ", 1)[-1]
        return Path(decoded[:opening] + renamed + decoded[closing + 1 :])
    return Path(decoded.rsplit(" => ", 1)[-1])


def parse_repository_history(
    repository: Path,
    emails: set[str],
    rules: FilterRules,
) -> tuple[list[AuthoredCommit], Counter[str], str | None]:
    command = [
        "git",
        "log",
        "--all",
        "--date=iso-strict",
        f"--pretty=format:{COMMIT_MARKER}%x09%H%x09%aI%x09%ae",
        "--numstat",
        "--find-renames",
    ]
    process = subprocess.Popen(
        command,
        cwd=repository,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    commits: list[AuthoredCommit] = []
    current_sha = ""
    current_date: datetime | None = None
    current_matches = False
    current_changes: list[FileChange] = []
    filtered: Counter[str] = Counter()

    def finish_current() -> None:
        if current_sha and current_matches:
            commits.append(AuthoredCommit(current_sha, current_date, tuple(current_changes)))

    for raw_line in process.stdout:
        line = raw_line.rstrip("\n")
        if line.startswith(COMMIT_MARKER + "\t"):
            finish_current()
            current_changes = []
            fields = line.split("\t", 3)
            current_sha = fields[1] if len(fields) > 1 else ""
            current_date = None
            author_email = fields[3].casefold() if len(fields) > 3 else ""
            current_matches = author_email in emails
            if len(fields) > 2:
                try:
                    current_date = datetime.fromisoformat(fields[2])
                except ValueError:
                    filtered["invalid_commit_date"] += 1
            continue
        if not current_matches or not line.strip():
            continue
        fields = line.split("\t", 2)
        if len(fields) != 3:
            filtered["unparsed_numstat"] += 1
            continue
        if fields[0] == "-" or fields[1] == "-":
            filtered["binary"] += 1
            continue
        try:
            additions = int(fields[0])
            deletions = int(fields[1])
        except ValueError:
            filtered["unparsed_numstat"] += 1
            continue
        path = destination_path(fields[2])
        checkout_path = repository / path
        if checkout_path.is_symlink():
            filtered["symlink"] += 1
            continue
        try:
            current_size = checkout_path.stat().st_size if checkout_path.is_file() else None
        except OSError:
            current_size = None
        decision = rules.decide(path, current_size)
        if not decision.keep or decision.language is None:
            filtered[decision.reason] += 1
            continue
        current_changes.append(FileChange(path, decision.language, additions, deletions))
    finish_current()
    stderr = process.stderr.read() if process.stderr is not None else ""
    return_code = process.wait()
    if return_code != 0:
        return [], filtered, stderr.strip() or f"git log exited {return_code}"
    ignored = gitignored_paths(
        repository,
        sorted(
            {change.path for commit in commits for change in commit.changes},
            key=lambda path: path.as_posix(),
        ),
    )
    if ignored:
        retained: list[AuthoredCommit] = []
        for commit in commits:
            changes = tuple(
                change for change in commit.changes if change.path.as_posix() not in ignored
            )
            filtered["gitignored"] += len(commit.changes) - len(changes)
            retained.append(AuthoredCommit(commit.sha, commit.authored_at, changes))
        commits = retained
    return commits, filtered, None


def is_shallow_repository(repository: Path) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "--is-shallow-repository"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def accumulate_repository(name: str, commits: Sequence[AuthoredCommit]) -> dict[str, Any]:
    overall = empty_totals()
    by_language: dict[str, dict[str, int]] = {}
    by_year: dict[str, dict[str, int]] = {}
    by_month: dict[str, dict[str, int]] = {}
    for commit in commits:
        overall["commits"] += 1
        languages = {change.language for change in commit.changes}
        for language in languages:
            bucket = by_language.setdefault(language, empty_totals())
            bucket["commits"] += 1
        year = str(commit.authored_at.year) if commit.authored_at else None
        month = commit.authored_at.strftime("%Y-%m") if commit.authored_at else None
        if year:
            by_year.setdefault(year, empty_totals())["commits"] += 1
        if month:
            by_month.setdefault(month, empty_totals())["commits"] += 1
        for change in commit.changes:
            add_lines(overall, change.additions, change.deletions)
            add_lines(
                by_language.setdefault(change.language, empty_totals()),
                change.additions,
                change.deletions,
            )
            if year:
                add_lines(
                    by_year.setdefault(year, empty_totals()), change.additions, change.deletions
                )
            if month:
                add_lines(
                    by_month.setdefault(month, empty_totals()), change.additions, change.deletions
                )
    return {
        "repo": name,
        "overall": overall,
        "by_language": dict(sorted(by_language.items())),
        "by_year": dict(sorted(by_year.items())),
        "by_month": dict(sorted(by_month.items())),
    }


def merge_totals(target: dict[str, int], source: Mapping[str, Any]) -> None:
    for key in target:
        target[key] += int(source.get(key, 0))


def combine(repositories: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    overall = empty_totals()
    by_language: dict[str, dict[str, int]] = {}
    by_year: dict[str, dict[str, int]] = {}
    by_month: dict[str, dict[str, int]] = {}
    by_repo: dict[str, dict[str, Any]] = {}
    for repository in repositories:
        merge_totals(overall, repository["overall"])
        name = str(repository["repo"])
        by_repo[name] = {
            **repository["overall"],
            "by_language": repository["by_language"],
        }
        for language, totals in repository["by_language"].items():
            merge_totals(by_language.setdefault(language, empty_totals()), totals)
        for year, totals in repository["by_year"].items():
            merge_totals(by_year.setdefault(year, empty_totals()), totals)
        for month, totals in repository["by_month"].items():
            merge_totals(by_month.setdefault(month, empty_totals()), totals)
    return {
        "overall": overall,
        "by_repo": dict(sorted(by_repo.items())),
        "by_language": dict(sorted(by_language.items())),
        "over_time": {
            "year": dict(sorted(by_year.items())),
            "month": dict(sorted(by_month.items())),
        },
    }


def new_figure(width: float = 9, height: float = 4.8) -> tuple[Figure, Any]:
    figure = Figure(figsize=(width, height), layout="constrained")
    FigureCanvasAgg(figure)
    return figure, figure.subplots()


def no_data_chart(path: Path, title: str, dpi: int) -> None:
    figure, axis = new_figure()
    axis.set_title(title)
    axis.text(
        0.5,
        0.5,
        "No matching contribution data",
        ha="center",
        va="center",
        transform=axis.transAxes,
    )
    axis.set_axis_off()
    save_chart(figure, path, dpi)


def save_chart(figure: Figure, path: Path, dpi: int) -> None:
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    figure.savefig(path.with_suffix(".svg"), bbox_inches="tight")


def write_charts(stats: Mapping[str, Any], output: Path, dpi: int) -> None:
    sns.set_theme(style="whitegrid", palette="colorblind")
    chart_root = output / "charts"
    chart_root.mkdir(parents=True, exist_ok=True)
    months = list(stats["over_time"]["month"])
    if months:
        positions = list(range(len(months)))
        tick_step = max(1, math.ceil(len(months) / 24))
        tick_positions = positions[::tick_step]
        tick_labels = months[::tick_step]
        figure, axis = new_figure(min(18, max(9, len(months) * 0.25)))
        sns.lineplot(
            x=positions,
            y=[stats["over_time"]["month"][month]["commits"] for month in months],
            marker="o",
            ax=axis,
        )
        axis.set_title("Personal commits over time")
        axis.set_ylabel("commits")
        axis.set_xticks(tick_positions, tick_labels, rotation=45, ha="right")
        save_chart(figure, chart_root / "commits-over-time.png", dpi)

        figure, axis = new_figure(min(18, max(9, len(months) * 0.25)))
        additions = [stats["over_time"]["month"][month]["additions"] for month in months]
        deletions = [-stats["over_time"]["month"][month]["deletions"] for month in months]
        net = [stats["over_time"]["month"][month]["net"] for month in months]
        sns.lineplot(x=positions, y=additions, marker="o", label="added", ax=axis)
        sns.lineplot(
            x=positions,
            y=deletions,
            marker="o",
            label="removed (negative)",
            ax=axis,
        )
        sns.lineplot(x=positions, y=net, marker="o", label="net", ax=axis)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title("Personal lines changed over time")
        axis.set_ylabel("lines")
        axis.set_xticks(tick_positions, tick_labels, rotation=45, ha="right")
        axis.legend()
        save_chart(figure, chart_root / "loc-over-time.png", dpi)
    else:
        no_data_chart(chart_root / "commits-over-time.png", "Personal commits over time", dpi)
        no_data_chart(chart_root / "loc-over-time.png", "Personal lines changed over time", dpi)
    languages = sorted(
        stats["by_language"],
        key=lambda language: stats["by_language"][language]["changed_lines"],
        reverse=True,
    )
    if languages:
        figure, axis = new_figure(min(18, max(9, len(languages) * 0.7)))
        sns.barplot(
            x=languages,
            y=[stats["by_language"][language]["changed_lines"] for language in languages],
            ax=axis,
        )
        axis.set_title("Language distribution by changed lines")
        axis.set_ylabel("added + removed lines")
        axis.tick_params(axis="x", rotation=35)
        save_chart(figure, chart_root / "language-distribution.png", dpi)
    else:
        no_data_chart(chart_root / "language-distribution.png", "Language distribution", dpi)


def stats_markdown(stats: Mapping[str, Any]) -> str:
    overall = stats["overall"]
    repo_rows = [
        [
            name,
            f"{values['commits']:,}",
            f"{values['additions']:,}",
            f"{values['deletions']:,}",
            f"{values['net']:,}",
        ]
        for name, values in stats["by_repo"].items()
    ]
    language_rows = [
        [
            name,
            f"{values['commits']:,}",
            f"{values['additions']:,}",
            f"{values['deletions']:,}",
            f"{values['net']:,}",
            f"{values['changed_lines']:,}",
        ]
        for name, values in sorted(
            stats["by_language"].items(),
            key=lambda item: item[1]["changed_lines"],
            reverse=True,
        )
    ]
    yearly_rows = [
        [year, values["commits"], values["additions"], values["deletions"], values["net"]]
        for year, values in stats["over_time"]["year"].items()
    ]
    return f"""# Personal code statistics

Generated: {stats["generated_at"]}  
Canonical data: [`stats.json`](stats.json)

## Overall

{
        markdown_table(
            ["Commits", "Lines added", "Lines removed", "Net LOC", "Changed lines"],
            [
                [
                    f"{overall['commits']:,}",
                    f"{overall['additions']:,}",
                    f"{overall['deletions']:,}",
                    f"{overall['net']:,}",
                    f"{overall['changed_lines']:,}",
                ]
            ],
        )
    }

## By repository

{
        markdown_table(["Repository", "Commits", "Added", "Removed", "Net"], repo_rows)
        if repo_rows
        else "No matching commits."
    }

## By language

Language commit counts count a commit once for every retained language it touches.

{
        markdown_table(["Language", "Commits", "Added", "Removed", "Net", "Changed"], language_rows)
        if language_rows
        else "No retained language changes."
    }

## By year

{
        markdown_table(["Year", "Commits", "Added", "Removed", "Net"], yearly_rows)
        if yearly_rows
        else "No matching commits."
    }

## Charts

![Commits over time](charts/commits-over-time.png)

![LOC over time](charts/loc-over-time.png)

![Language distribution](charts/language-distribution.png)

All chart values come from `stats.json → over_time` or `stats.json → by_language`.

## Methodology and limits

Commits are matched case-insensitively against local `git config user.email`, configured aliases,
and `--emails` values. `git log --all --numstat --find-renames` supplies additions and removals
without treating a pure rename as delete-and-readd churn.
Binary changes, gitignored current paths, excluded/generated/vendored directories and patterns,
lockfiles, minified files, and unsupported extensions use the dataset builder's filter rules.
Git cannot apply today's ignore or file-size rules to a deleted historical path that no longer
exists, so such a path is filtered only by its historical name and extension. Net LOC is additions
minus removals; it is contribution churn, not the current checkout's physical line count.
"""


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)
    config, root = load_config(args.config)
    repository_root = resolved_path(
        root,
        args.repo_dir or config_value(config, "paths.repos", "repos"),
    )
    output = resolved_path(root, args.output_dir or config_value(config, "paths.stats", "stats"))
    max_repos_value = choose(args.max_repos, config, "stats.max_repos", None)
    max_repos = int(max_repos_value) if max_repos_value is not None else None
    if max_repos is not None and max_repos <= 0:
        raise SystemExit("stats.max_repos must be greater than zero when set")
    extensions = (
        [item.strip().casefold() for item in args.extensions.split(",") if item.strip()]
        if args.extensions
        else None
    )
    rules = FilterRules.from_config(
        config,
        max_file_bytes=args.max_file_bytes,
        extensions=extensions,
    )
    emails = configured_emails(config, args.emails)
    repositories = discover_repositories(repository_root, max_repos)
    output.mkdir(parents=True, exist_ok=True)
    per_repository: list[dict[str, Any]] = []
    filters: Counter[str] = Counter()
    failures: dict[str, str] = {}
    if not emails:
        LOG.warning("No author emails are configured; writing empty statistics.")
    else:
        for index, repository in enumerate(repositories, start=1):
            LOG.info("[%d/%d] reading history for %s", index, len(repositories), repository.name)
            if is_shallow_repository(repository):
                failures[repository.name] = "shallow clone cannot provide all-time statistics"
                LOG.error("%s is shallow; run git fetch --unshallow", repository.name)
                continue
            commits, repository_filters, error = parse_repository_history(
                repository, set(emails), rules
            )
            filters.update(repository_filters)
            if error:
                failures[repository.name] = error
                LOG.error("%s: %s", repository.name, error)
                continue
            per_repository.append(accumulate_repository(repository.name, commits))
    combined = combine(per_repository)
    stats = {
        "generated_at": utc_now(),
        "sources": {
            "repository_root": str(repository_root),
            "config": "config.yaml",
            "git_command": "git log --all --date=iso-strict --numstat --find-renames",
        },
        "matched_emails": emails,
        "repositories_discovered": len(repositories),
        "repositories_processed": len(per_repository),
        "failures": failures,
        "filter_report": dict(sorted(filters.items())),
        **combined,
    }
    write_json(output / "stats.json", stats)
    write_text(output / "STATS.md", stats_markdown(stats))
    dpi = int(choose(args.chart_dpi, config, "stats.chart_dpi", 160))
    if dpi <= 0:
        raise SystemExit("stats.chart_dpi must be greater than zero")
    write_charts(stats, output, dpi)
    final_summary(
        "Contribution statistics",
        [
            ("Matched emails", len(emails)),
            ("Repositories", len(per_repository)),
            ("Commits", stats["overall"]["commits"]),
            ("Lines added", stats["overall"]["additions"]),
            ("Lines removed", stats["overall"]["deletions"]),
            ("Failures", len(failures)),
            ("Markdown", output / "STATS.md"),
        ],
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
