from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from corpuslib import (
    config_value,
    final_summary,
    load_config,
    resolved_path,
    safe_name,
    utc_now,
    write_json,
    write_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package an MLX Gemma 4 adapter for Ollama and verify it through Codex."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--tag")
    parser.add_argument("--output-dir", default="deployment")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--skip-codex", action="store_true")
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    return parser.parse_args()


def adapter_files(adapter_path: Path) -> tuple[Path, Path]:
    weights = adapter_path / "adapters.safetensors"
    config = adapter_path / "adapter_config.json"
    missing = [path.name for path in (weights, config) if not path.is_file()]
    if missing:
        raise ValueError(f"adapter is missing: {', '.join(missing)}")
    return weights, config


def adapter_pairs(names: Iterable[str]) -> list[tuple[str, str]]:
    modules: dict[str, dict[str, str]] = {}
    for name in names:
        if name.endswith(".lora_a"):
            modules.setdefault(name.removesuffix(".lora_a"), {})["a"] = name
        elif name.endswith(".lora_b"):
            modules.setdefault(name.removesuffix(".lora_b"), {})["b"] = name
    if not modules:
        raise ValueError("adapter contains no LoRA tensor pairs")
    pairs: list[tuple[str, str]] = []
    for module, factors in sorted(modules.items()):
        if "a" not in factors:
            raise ValueError(f"{module} is missing lora_a")
        if "b" not in factors:
            raise ValueError(f"{module} is missing lora_b")
        pairs.append((factors["a"], factors["b"]))
    return pairs


def peft_tensor_name(name: str) -> tuple[str, bool]:
    if name.endswith(".lora_a"):
        suffix = ".lora_A.weight"
        module = name.removesuffix(".lora_a")
    elif name.endswith(".lora_b"):
        suffix = ".lora_B.weight"
        module = name.removesuffix(".lora_b")
    else:
        raise ValueError(f"not an MLX LoRA tensor: {name}")
    return f"base_model.model.{module}{suffix}", True


def peft_adapter_config(mlx_config: Mapping[str, Any], base_model_name: str) -> dict[str, Any]:
    parameters = mlx_config.get("lora_parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("adapter config is missing lora_parameters")
    rank = int(parameters.get("rank", 0) or 0)
    scale = float(parameters.get("scale", 0) or 0)
    if rank <= 0 or scale <= 0:
        raise ValueError("adapter rank and scale must be positive")
    keys = parameters.get("keys", [])
    if not isinstance(keys, list) or not all(isinstance(key, str) for key in keys):
        raise ValueError("adapter target keys must be a list of strings")
    return {
        "base_model_name_or_path": base_model_name,
        "bias": "none",
        "inference_mode": True,
        "lora_alpha": rank * scale,
        "lora_dropout": float(parameters.get("dropout", 0) or 0),
        "peft_type": "LORA",
        "r": rank,
        "target_modules": sorted({key.rsplit(".", 1)[-1] for key in keys}),
        "task_type": "CAUSAL_LM",
    }


def export_peft_adapter(
    adapter_path: Path,
    output_path: Path,
    base_model_name: str,
) -> dict[str, Any]:
    from safetensors import safe_open
    from safetensors.numpy import save_file

    weights_path, config_path = adapter_files(adapter_path)
    mlx_config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(mlx_config, Mapping):
        raise ValueError("adapter config must be a JSON object")
    tensors: dict[str, Any] = {}
    with safe_open(weights_path, framework="np") as source:
        pairs = adapter_pairs(source.keys())
        for a_name, b_name in pairs:
            a = source.get_tensor(a_name)
            b = source.get_tensor(b_name)
            if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
                raise ValueError(f"incompatible LoRA shapes for {a_name} and {b_name}")
            for name, tensor in ((a_name, a), (b_name, b)):
                target_name, transpose = peft_tensor_name(name)
                tensors[target_name] = tensor.T.copy() if transpose else tensor.copy()
    output_path.mkdir(parents=True, exist_ok=True)
    weights_output = output_path / "adapter_model.safetensors"
    save_file(tensors, weights_output, metadata={"format": "pt"})
    config = peft_adapter_config(mlx_config, base_model_name)
    write_json(output_path / "adapter_config.json", config)
    with weights_output.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {
        "source": str(adapter_path),
        "output": str(output_path),
        "tensor_pairs": len(tensors) // 2,
        "rank": config["r"],
        "lora_alpha": config["lora_alpha"],
        "sha256": digest,
    }


def converter_command(
    python: Path,
    llama_cpp: Path,
    peft_adapter: Path,
    base_config: Path,
    output: Path,
) -> list[str]:
    return [
        str(python),
        str(llama_cpp / "convert_lora_to_gguf.py"),
        "--base",
        str(base_config),
        "--outfile",
        str(output),
        "--outtype",
        "f16",
        str(peft_adapter),
    ]


def ensure_llama_cpp(path: Path, repository: str, revision: str) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        run_checked(["git", "init", "-q", str(path)])
        run_checked(["git", "-C", str(path), "remote", "add", "origin", repository])
    if not (path / ".git").is_dir():
        raise ValueError(f"llama.cpp path is not a git checkout: {path}")
    head = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    current = head.stdout.strip() if head.returncode == 0 else ""
    if current != revision:
        run_checked(["git", "-C", str(path), "fetch", "--depth", "1", "origin", revision])
        run_checked(["git", "-C", str(path), "checkout", "--detach", "FETCH_HEAD"])
    converter = path / "convert_lora_to_gguf.py"
    if not converter.is_file():
        raise ValueError(f"llama.cpp converter is missing: {converter}")
    return path


def render_modelfile(
    base_model: str,
    adapter_path: Path,
    context_window: int,
    system_prompt: str = "",
) -> str:
    if context_window <= 0:
        raise ValueError("context window must be positive")
    prompt = system_prompt.strip()
    if '"""' in prompt:
        raise ValueError("system prompt cannot contain triple double quotes")
    rendered = (
        f"FROM {base_model}\nADAPTER {adapter_path.resolve()}\n"
        f"PARAMETER temperature 0\nPARAMETER num_ctx {context_window}\n"
    )
    if prompt:
        rendered += f'SYSTEM """{prompt}"""\n'
    return rendered


def listed_models() -> set[str]:
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return set()
    return {line.split()[0] for line in result.stdout.splitlines()[1:] if line.split()}


def available_disk_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1_000_000_000


def require_disk_space(
    path: Path,
    minimum_free_gb: float,
    additional_required_gb: float,
) -> float:
    available = available_disk_gb(path)
    required = minimum_free_gb + additional_required_gb
    if available < required:
        raise RuntimeError(
            f"{available:.1f} GB free; {required:.1f} GB required before Ollama import"
        )
    return available


def run_checked(command: Sequence[str], cwd: Path | None = None) -> str:
    result = subprocess.run(list(command), cwd=cwd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "command failed"
        raise RuntimeError(f"{command[0]} exited {result.returncode}: {message}")
    return (result.stdout + result.stderr).strip()


def ollama_chat(host: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{host.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Ollama chat failed: HTTP {error.code}: {detail}") from error
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Ollama chat failed: {error}") from error
    if not isinstance(result, dict):
        raise RuntimeError("Ollama chat returned an unexpected payload")
    return result


def runtime_metrics(response: Mapping[str, Any]) -> dict[str, float | int | None]:
    eval_count = int(response.get("eval_count", 0) or 0)
    eval_duration = int(response.get("eval_duration", 0) or 0)
    total_duration = int(response.get("total_duration", 0) or 0)
    decode_tps = (
        eval_count / (eval_duration / 1_000_000_000)
        if eval_count > 0 and eval_duration > 0
        else None
    )
    ttft_seconds = (
        max(0, total_duration - eval_duration) / 1_000_000_000 if total_duration > 0 else None
    )
    return {
        "eval_count": eval_count,
        "decode_tps": round(decode_tps, 3) if decode_tps is not None else None,
        "ttft_seconds": round(ttft_seconds, 3) if ttft_seconds is not None else None,
    }


def expected_tool_call(response: Mapping[str, Any], name: str) -> bool:
    message = response.get("message")
    if not isinstance(message, Mapping):
        return False
    calls = message.get("tool_calls")
    if not isinstance(calls, list):
        return False
    for call in calls:
        if not isinstance(call, Mapping):
            continue
        function = call.get("function")
        if isinstance(function, Mapping) and function.get("name") == name:
            return True
    return False


def chat_and_tool_smoke(host: str, tag: str) -> dict[str, Any]:
    warmup = {
        "model": tag,
        "messages": [{"role": "user", "content": "Reply with exactly: ready"}],
        "stream": False,
        "think": False,
        "keep_alive": "10m",
    }
    ollama_chat(host, warmup)
    timed = ollama_chat(host, warmup)
    tool_name = "read_project_status"
    tool_response = ollama_chat(
        host,
        {
            "model": tag,
            "messages": [
                {
                    "role": "user",
                    "content": "Use the available tool to inspect the current project status.",
                }
            ],
            "stream": False,
            "think": False,
            "keep_alive": "10m",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": "Read the current local project status.",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        },
    )
    return {
        "runtime": runtime_metrics(timed),
        "tool_call": {"name": tool_name, "passed": expected_tool_call(tool_response, tool_name)},
    }


def codex_command(tag: str, root: Path, context_window: int) -> list[str]:
    if context_window <= 0:
        raise ValueError("Codex context window must be positive")
    compact_limit = context_window * 3 // 4
    return [
        "codex",
        "exec",
        "--oss",
        "--local-provider",
        "ollama",
        "--model",
        tag,
        "--sandbox",
        "workspace-write",
        "--ephemeral",
        "-c",
        'approval_policy="never"',
        "-c",
        f"model_context_window={context_window}",
        "-c",
        f"model_auto_compact_token_limit={compact_limit}",
        "-C",
        str(root),
        "Fix the failing test in calculator.py, run the test, and stop when it passes.",
    ]


def codex_smoke(tag: str, context_window: int = 32_768) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="gemma4-codex-smoke-") as directory:
        root = Path(directory)
        (root / "calculator.py").write_text(
            "def add(left: int, right: int) -> int:\n    return left - right\n",
            encoding="utf-8",
        )
        (root / "test_calculator.py").write_text(
            "from calculator import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n",
            encoding="utf-8",
        )
        run_checked(["git", "init", "-q"], cwd=root)
        started = time.monotonic()
        command = codex_command(tag, root, context_window)
        output = run_checked(command, cwd=root)
        verification = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        return {
            "passed": verification.returncode == 0,
            "duration_seconds": round(time.monotonic() - started, 3),
            "codex_output_tail": output[-2000:],
            "pytest_output": (verification.stdout + verification.stderr).strip(),
        }


def configured_training_model(config: Mapping[str, Any]) -> Mapping[str, Any]:
    models = config_value(config, "training.models", [])
    if not isinstance(models, list) or len(models) != 1 or not isinstance(models[0], Mapping):
        raise ValueError("deployment requires exactly one configured training model")
    return models[0]


def prepare_gguf_adapter(
    root: Path,
    adapter_path: Path,
    output_root: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    model = configured_training_model(config)
    model_id = str(model.get("model_id", ""))
    revision = str(model.get("revision", ""))
    upstream_model_id = str(model.get("upstream_model_id", ""))
    if not model_id or not revision or not upstream_model_id:
        raise ValueError("training model must define model_id, revision, and upstream_model_id")
    base_config = Path(snapshot_download(repo_id=model_id, revision=revision))
    peft_path = output_root / "peft-adapter"
    export = export_peft_adapter(adapter_path, peft_path, upstream_model_id)
    llama_cpp_path = resolved_path(
        root, str(config_value(config, "deployment.llama_cpp_path", ".cache/llama.cpp"))
    )
    llama_cpp = ensure_llama_cpp(
        llama_cpp_path,
        str(
            config_value(
                config,
                "deployment.llama_cpp_repository",
                "https://github.com/ggml-org/llama.cpp.git",
            )
        ),
        str(config_value(config, "deployment.llama_cpp_revision", "")),
    )
    output = output_root / "adapter.gguf"
    command = converter_command(Path(sys.executable), llama_cpp, peft_path, base_config, output)
    conversion = subprocess.run(command, capture_output=True, text=True, check=False)
    conversion_log = (conversion.stdout + conversion.stderr).strip()
    write_text(output_root / "conversion.log", conversion_log + "\n")
    if conversion.returncode != 0:
        raise RuntimeError(f"LoRA conversion failed: {conversion_log[-4000:]}")
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("LoRA conversion did not produce a GGUF adapter")
    return {
        "base_config": str(base_config),
        "llama_cpp_revision": run_checked(
            ["git", "-C", str(llama_cpp), "rev-parse", "HEAD"]
        ).strip(),
        "gguf_adapter": str(output),
        "gguf_size_bytes": output.stat().st_size,
        "peft_export": export,
    }


def execute_deployment(
    root: Path,
    modelfile: Path,
    adapter_path: Path,
    gguf_adapter: Path,
    base_model: str,
    tag: str,
    host: str,
    config: Mapping[str, Any],
    skip_codex: bool,
) -> dict[str, Any]:
    minimum_free = float(config_value(config, "deployment.minimum_free_disk_gb", 12))
    base_size = float(config_value(config, "deployment.ollama_base_size_gb", 0))
    adapter_size = gguf_adapter.stat().st_size / 1_000_000_000
    additional = adapter_size + (0.0 if base_model in listed_models() else base_size)
    available = require_disk_space(root, minimum_free, additional)
    if base_model not in listed_models():
        run_checked(["ollama", "pull", base_model])
    run_checked(["ollama", "create", tag, "-f", str(modelfile)])
    smoke = chat_and_tool_smoke(host, tag)
    runtime = smoke["runtime"]
    minimum_tps = float(config_value(config, "deployment.minimum_decode_tps", 20))
    maximum_ttft = float(config_value(config, "deployment.maximum_ttft_seconds", 5))
    performance_passed = (
        isinstance(runtime.get("decode_tps"), (int, float))
        and float(runtime["decode_tps"]) >= minimum_tps
        and isinstance(runtime.get("ttft_seconds"), (int, float))
        and float(runtime["ttft_seconds"]) <= maximum_ttft
    )
    codex_context = int(config_value(config, "deployment.codex_context_window", 32_768))
    codex = {"passed": None, "reason": "skipped"} if skip_codex else codex_smoke(tag, codex_context)
    return {
        "generated_at": utc_now(),
        "status": "passed"
        if smoke["tool_call"]["passed"] and performance_passed and codex.get("passed") is not False
        else "failed",
        "base_model": base_model,
        "tag": tag,
        "adapter_path": str(adapter_path),
        "gguf_adapter": str(gguf_adapter),
        "modelfile": str(modelfile),
        "available_disk_before_gb": round(available, 3),
        "performance_thresholds": {
            "minimum_decode_tps": minimum_tps,
            "maximum_ttft_seconds": maximum_ttft,
        },
        "smoke": smoke,
        "performance_passed": performance_passed,
        "codex": codex,
    }


def main() -> int:
    args = parse_args()
    config, root = load_config(args.config)
    adapter_path = Path(args.adapter_path).expanduser().resolve()
    adapter_files(adapter_path)
    base_model = str(config_value(config, "deployment.ollama_base", "gemma4:e4b"))
    default_tag = str(config_value(config, "deployment.ollama_model", "gemma4-e4b-personal"))
    tag = args.tag or default_tag
    output_root = resolved_path(root, args.output_dir)
    deployment_root = output_root / safe_name(tag)
    gguf_adapter = deployment_root / "adapter.gguf"
    modelfile = deployment_root / "Modelfile"
    context_window = int(config_value(config, "deployment.codex_context_window", 32_768))
    system_prompt = ""
    system_prompt_value = config_value(config, "deployment.system_prompt_path", None)
    if system_prompt_value:
        system_prompt_path = resolved_path(root, str(system_prompt_value))
        system_prompt = system_prompt_path.read_text(encoding="utf-8")
    write_text(
        modelfile,
        render_modelfile(base_model, gguf_adapter, context_window, system_prompt),
    )
    if not args.execute:
        final_summary(
            "Ollama deployment preparation",
            [
                ("Base", base_model),
                ("Tag", tag),
                ("Adapter", gguf_adapter),
                ("Modelfile", modelfile),
            ],
        )
        return 0
    package = prepare_gguf_adapter(root, adapter_path, deployment_root, config)
    result = execute_deployment(
        root,
        modelfile,
        adapter_path,
        gguf_adapter,
        base_model,
        tag,
        args.ollama_host,
        config,
        args.skip_codex,
    )
    result["package"] = package
    result_path = modelfile.parent / "verification.json"
    write_json(result_path, result)
    final_summary(
        "Ollama deployment verification",
        [
            ("Status", result["status"]),
            ("Tool call", result["smoke"]["tool_call"]["passed"]),
            ("Performance", result["performance_passed"]),
            ("Codex", result["codex"].get("passed")),
            ("Evidence", result_path),
        ],
    )
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
