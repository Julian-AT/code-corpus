from datetime import datetime
from pathlib import Path

from fetch_repos import FetchSettings, select_repositories


def test_repository_filters_are_applied_before_limit() -> None:
    year = datetime.now().year
    repositories = [
        {
            "name": "private",
            "isPrivate": True,
            "isFork": False,
            "isArchived": False,
            "pushedAt": f"{year}-01-01T00:00:00Z",
        },
        {
            "name": "fork",
            "isPrivate": False,
            "isFork": True,
            "isArchived": False,
            "pushedAt": f"{year}-01-01T00:00:00Z",
        },
        {
            "name": "old",
            "isPrivate": False,
            "isFork": False,
            "isArchived": False,
            "pushedAt": "2000-01-01T00:00:00Z",
        },
        {
            "name": "kept",
            "isPrivate": False,
            "isFork": False,
            "isArchived": False,
            "pushedAt": f"{year}-01-01T00:00:00Z",
        },
    ]
    settings = FetchSettings(Path("repos"), False, False, False, year - 1, 10)

    selected, excluded = select_repositories(repositories, settings)

    assert [item["name"] for item in selected] == ["kept"]
    assert excluded == {"private": 1, "fork": 1, "archived": 0, "older": 1}
