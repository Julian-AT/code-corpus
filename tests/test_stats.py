from pathlib import Path

from corpuslib import FilterRules, load_config
from stats import accumulate_repository, destination_path, parse_repository_history, write_charts


def commit(repository: Path, message: str, email: str) -> None:
    import subprocess

    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Stats Test",
            "-c",
            f"user.email={email}",
            "commit",
            "-qm",
            message,
        ],
        cwd=repository,
        check=True,
    )


def test_git_history_counts_only_matching_author_and_retained_files(tmp_path: Path) -> None:
    import subprocess

    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    (repository / "main.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    (repository / "package-lock.json").write_text("{}\n", encoding="utf-8")
    commit(repository, "mine", "me@example.com")
    (repository / "other.py").write_text("OTHER = True\n", encoding="utf-8")
    commit(repository, "theirs", "other@example.com")
    config, _ = load_config(Path(__file__).parents[1] / "config.yaml")
    rules = FilterRules.from_config(config)

    commits, filtered, error = parse_repository_history(repository, {"me@example.com"}, rules)
    totals = accumulate_repository("repo", commits)

    assert error is None
    assert totals["overall"]["commits"] == 1
    assert totals["overall"]["additions"] == 2
    assert totals["by_language"]["Python"]["additions"] == 2
    assert filtered["lockfile"] == 1


def test_compact_git_rename_uses_destination_path() -> None:
    assert destination_path("src/{old.py => new.py}") == Path("src/new.py")
    assert destination_path("old.py => new.py") == Path("new.py")


def test_stats_charts_export_png_and_svg(tmp_path: Path) -> None:
    data = {
        "over_time": {
            "month": {
                "2026-01": {"commits": 1, "additions": 4, "deletions": 1, "net": 3}
            }
        },
        "by_language": {"Python": {"changed_lines": 5}},
    }

    write_charts(data, tmp_path, 72)

    assert (tmp_path / "charts/commits-over-time.png").is_file()
    assert b"<svg" in (tmp_path / "charts/commits-over-time.svg").read_bytes()[:500]
