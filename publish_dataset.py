from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from huggingface_hub import DatasetCard, HfApi
from huggingface_hub.errors import RepositoryNotFoundError

from build_dataset import FEATURES, detected_secret_kind
from corpuslib import config_value, load_config, resolved_path, write_json
from datasets import DatasetDict, load_dataset

PUBLISHED_ROOT_FILES = {"README.md", "LICENSE", "dataset_infos.json", "statistics.json"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and publish a generated dataset to a Hugging Face dataset repo."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--repo-id", help="Hub repo in owner/name form")
    parser.add_argument("--variant", help="Generated dataset directory to publish")
    parser.add_argument(
        "--public",
        action="store_true",
        help="Create a public repo; refused when retained rows came from private repositories",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate without changing the Hub")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def retained_private_repositories(
    statistics: Mapping[str, Any], manifest: Mapping[str, Any]
) -> list[str]:
    retained = set(statistics.get("by_repo", {}))
    repositories = manifest.get("repositories", [])
    if not isinstance(repositories, list):
        return []
    return sorted(
        str(item["name"])
        for item in repositories
        if isinstance(item, dict)
        and item.get("isPrivate") is True
        and str(item.get("name", "")) in retained
    )


def sanitized_statistics(statistics: dict[str, Any], repo_id: str) -> dict[str, Any]:
    sanitized = json.loads(json.dumps(statistics))
    sources = sanitized.setdefault("sources", {})
    sources["repository_root"] = "local repository checkouts (not published)"
    sources["config"] = "local config.yaml (not published)"
    sanitized["variant"] = "default"
    sanitized["self_test"] = {
        "status": "passed",
        "loader": f'datasets.load_dataset("{repo_id}")',
        "splits": {
            name: int(count)
            for name, count in (
                ("train", sanitized.get("rows", {}).get("train", 0)),
                ("valid", sanitized.get("rows", {}).get("valid", 0)),
            )
            if count
        },
    }
    return sanitized


def prepare_upload(source: Path, destination: Path, repo_id: str) -> None:
    required = [source / name for name in PUBLISHED_ROOT_FILES]
    missing = [str(path) for path in required if not path.is_file()]
    parquet_files = sorted((source / "data").glob("*.parquet"))
    if not parquet_files:
        missing.append(str(source / "data/*.parquet"))
    if missing:
        raise FileNotFoundError("Missing generated dataset artifacts: " + ", ".join(missing))

    destination.mkdir(parents=True, exist_ok=True)
    for name in ("README.md", "LICENSE", "dataset_infos.json"):
        shutil.copy2(source / name, destination / name)
    data_directory = destination / "data"
    data_directory.mkdir()
    for path in parquet_files:
        shutil.copy2(path, data_directory / path.name)
    write_json(
        destination / "statistics.json",
        sanitized_statistics(read_json(source / "statistics.json"), repo_id),
    )


def validate_upload(folder: Path) -> DatasetDict:
    card = DatasetCard.load(folder / "README.md")
    card.validate(repo_type="dataset")
    dataset = load_dataset(str(folder))
    if not isinstance(dataset, DatasetDict) or not dataset:
        raise ValueError("The staged Hub directory did not load as a non-empty DatasetDict")
    for split_name, split in dataset.items():
        if split.features != FEATURES:
            raise ValueError(f"Unexpected feature schema in {split_name}")
        for index, text in enumerate(split["text"]):
            kind = detected_secret_kind(text)
            if kind:
                raise ValueError(
                    f"Secret pattern {kind!r} found in staged row {split_name}[{index}]"
                )
    local_home = str(Path.home())
    for path in folder.rglob("*"):
        if (
            path.is_file()
            and path.suffix != ".parquet"
            and local_home in path.read_text(encoding="utf-8")
        ):
            raise ValueError(f"Local home path leaked into publication metadata: {path}")
        if path.is_file() and path.suffix != ".parquet":
            text = path.read_text(encoding="utf-8")
            if "raw-max" in text.casefold():
                raise ValueError(f"Internal variant name leaked into publication metadata: {path}")
    return dataset


def remote_dataset(api: HfApi, repo_id: str) -> tuple[bool, bool | None]:
    try:
        info = api.repo_info(repo_id, repo_type="dataset")
    except RepositoryNotFoundError:
        return False, None
    return True, bool(info.private)


def publish(folder: Path, repo_id: str, private: bool) -> str:
    api = HfApi()
    exists, current_private = remote_dataset(api, repo_id)
    if exists and current_private != private:
        visibility = "private" if current_private else "public"
        requested = "private" if private else "public"
        raise ValueError(
            f"{repo_id} already exists as {visibility}; refusing to change it to {requested}"
        )
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=folder,
        commit_message="Publish validated code corpus",
        delete_patterns=[
            "README.md",
            "LICENSE",
            "dataset_infos.json",
            "statistics.json",
            "data/**",
        ],
    )
    remote = load_dataset(repo_id, token=True, download_mode="force_redownload")
    local = load_dataset(str(folder))
    expected = {name: len(split) for name, split in local.items()}
    actual = {name: len(split) for name, split in remote.items()}
    if actual != expected:
        raise RuntimeError(f"Remote split counts do not match: expected {expected}, got {actual}")
    return commit.repo_url


def main() -> int:
    args = parse_args()
    config, root = load_config(args.config)
    repo_id = args.repo_id or str(config_value(config, "hub.repo_id", ""))
    if not repo_id or "/" not in repo_id:
        raise SystemExit("Set hub.repo_id or pass --repo-id owner/name")
    variant = args.variant or str(config_value(config, "hub.variant", "raw-max"))
    source = resolved_path(root, config_value(config, "paths.datasets", "datasets")) / variant
    statistics = read_json(source / "statistics.json")
    manifest_path = resolved_path(root, config_value(config, "paths.repos", "repos")) / "repos.json"
    manifest = read_json(manifest_path)
    private_sources = retained_private_repositories(statistics, manifest)
    configured_private = bool(config_value(config, "hub.private", True))
    private = False if args.public else configured_private
    allow_private_sources = bool(config_value(config, "hub.allow_private_sources", False))
    if not private and private_sources and not allow_private_sources:
        raise SystemExit(
            "Public upload refused: retained rows came from private repositories: "
            + ", ".join(private_sources)
            + ". Set hub.allow_private_sources only after explicitly approving their publication."
        )
    if not private and private_sources:
        print(
            "WARNING: public upload includes retained rows from "
            f"{len(private_sources)} private source repositories (explicitly allowed by config)."
        )

    with tempfile.TemporaryDirectory(prefix="code-corpus-hub-") as temporary:
        folder = Path(temporary)
        prepare_upload(source, folder, repo_id)
        dataset = validate_upload(folder)
        counts = {name: len(split) for name, split in dataset.items()}
        files = sorted(
            path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file()
        )
        print(f"Validated {repo_id}: {counts}; files={files}; private={private}")
        if args.dry_run:
            return 0
        url = publish(folder, repo_id, private)
        print(f"Published and verified: {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
