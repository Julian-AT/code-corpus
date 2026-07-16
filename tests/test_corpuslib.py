from pathlib import Path

from corpuslib import FilterRules, is_test_path, load_config


def test_filter_rules_exclude_required_paths() -> None:
    config, _ = load_config(Path(__file__).parents[1] / "config.yaml")
    rules = FilterRules.from_config(config)

    assert rules.decide(Path("node_modules/lib.js"), 10).reason == "excluded_directory"
    assert rules.decide(Path("src/app.min.js"), 10).reason == "minified"
    assert rules.decide(Path("package-lock.json"), 10).reason == "lockfile"
    assert rules.decide(Path("config/credentials-prod.json"), 10).reason == "sensitive_path"
    assert rules.decide(Path("keys/service.key"), 10).reason == "sensitive_path"
    assert rules.decide(Path("src/blob.png"), 10).reason == "extension"
    assert rules.decide(Path("src/main.py"), 10).keep
    assert rules.decide(Path("README.md"), 10).language == "Markdown"
    assert rules.decide(Path("docker/Dockerfile"), 10).language == "Dockerfile"


def test_test_path_detection() -> None:
    assert is_test_path(Path("tests/test_parser.py"))
    assert is_test_path(Path("src/parser.spec.ts"))
    assert not is_test_path(Path("src/parser.ts"))
