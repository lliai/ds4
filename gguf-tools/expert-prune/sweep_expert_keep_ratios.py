#!/usr/bin/env python3
"""Sweep DS4 expert-mask keep ratios and smoke-test whether output is readable."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_DATASET = SCRIPT_DIR / "ds4-c4-prune.txt"
DEFAULT_PROMPT = (
    "Please answer in two short sentences: introduce yourself and say one "
    "thing you can help with."
)
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
WORDISH_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]")


@dataclass(frozen=True)
class RatioSpec:
    percent: int
    ratio: float

    @property
    def tag(self) -> str:
        return f"keep-{self.percent:03d}"

    @property
    def ratio_arg(self) -> str:
        return f"{self.ratio:.6f}".rstrip("0").rstrip(".")


@dataclass
class CommandResult:
    returncode: Optional[int]
    seconds: float
    timed_out: bool
    command: List[str]
    stdout_path: str
    stderr_path: str


@dataclass
class SweepResult:
    keep_percent: int
    keep_ratio: float
    gpu: str
    passed: bool
    reason: str
    keep_per_layer: Optional[int]
    mask_path: str
    ratio_dir: str
    prune_returncode: Optional[int]
    test_returncode: Optional[int]
    prune_seconds: float
    test_seconds: float
    prune_stdout: str
    prune_stderr: str
    test_stdout: str
    test_stderr: str
    output_preview: str


def resolve_repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def resolve_work_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    if len(path.parts) >= 2 and path.parts[0] == "gguf-tools" and path.parts[1] == "expert-prune":
        return (REPO_ROOT / path).resolve()
    return (SCRIPT_DIR / path).resolve()


def path_for_json(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_gpus(value: str) -> List[str]:
    gpus = [part.strip() for part in value.split(",") if part.strip()]
    if not gpus:
        raise SystemExit("--gpus must contain at least one CUDA device id")
    return gpus


def lock_tag(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def parse_ratios(args: argparse.Namespace) -> List[RatioSpec]:
    if args.ratios:
        specs: List[RatioSpec] = []
        seen = set()
        for item in args.ratios.split(","):
            item = item.strip()
            if not item:
                continue
            value = float(item)
            if value > 1.0:
                percent = int(round(value))
                ratio = percent / 100.0
            else:
                ratio = value
                percent = int(round(ratio * 100.0))
            if percent < 1 or percent > 100:
                raise SystemExit(f"invalid keep ratio/percent: {item}")
            if percent not in seen:
                seen.add(percent)
                specs.append(RatioSpec(percent=percent, ratio=ratio))
        if not specs:
            raise SystemExit("--ratios did not contain any valid ratios")
        return sorted(specs, key=lambda spec: spec.percent, reverse=True)

    if args.start_percent < args.stop_percent:
        raise SystemExit("--start-percent must be >= --stop-percent")
    if args.step_percent <= 0:
        raise SystemExit("--step-percent must be positive")

    specs = []
    percent = args.start_percent
    while percent >= args.stop_percent:
        specs.append(RatioSpec(percent=percent, ratio=percent / 100.0))
        percent -= args.step_percent
    return specs


def format_cmd(cmd: Sequence[str], env_prefix: Optional[Dict[str, str]] = None) -> str:
    parts = []
    if env_prefix:
        for key, value in env_prefix.items():
            parts.append(f"{key}={shlex.quote(value)}")
    parts.extend(shlex.quote(str(part)) for part in cmd)
    return " ".join(parts)


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text.replace("\0", ""))


def read_text(path: Path, limit: Optional[int] = None) -> str:
    if not path.exists():
        return ""
    data = path.read_text(encoding="utf-8", errors="replace")
    if limit is not None and len(data) > limit:
        return data[-limit:]
    return data


def single_line(text: str, max_len: int = 240) -> str:
    text = " ".join(strip_ansi(text).split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def longest_same_char_run(text: str) -> int:
    best = 0
    current = 0
    previous = None
    for ch in text:
        if ch == previous:
            current += 1
        else:
            previous = ch
            current = 1
        if current > best:
            best = current
    return best


def looks_like_normal_speech(text: str, min_chars: int, min_wordish_ratio: float) -> Tuple[bool, str]:
    text = strip_ansi(text).strip()
    if not text:
        return False, "empty output"

    compact = "".join(ch for ch in text if not ch.isspace())
    if len(compact) < min_chars:
        return False, f"too short ({len(compact)} < {min_chars} non-space chars)"

    printable = sum(1 for ch in text if ch.isprintable() or ch in "\n\r\t")
    if printable / max(len(text), 1) < 0.98:
        return False, "contains too many non-printable characters"

    if compact.count("\ufffd") / max(len(compact), 1) > 0.02:
        return False, "contains too many replacement characters"

    wordish = len(WORDISH_RE.findall(compact))
    if wordish / max(len(compact), 1) < min_wordish_ratio:
        return False, "too little alphanumeric/CJK content"

    if len(set(compact)) <= 5 and len(compact) >= max(min_chars, 20):
        return False, "too few unique characters"

    max_run = longest_same_char_run(compact)
    if max_run >= max(10, len(compact) // 3):
        return False, f"repeated character run is too long ({max_run})"

    lower = text.lower()
    if re.search(r"\b(nan|inf|-inf)\b", lower):
        return False, "contains numerical failure marker"

    for width in range(4, min(16, len(compact) // 2) + 1):
        unit = compact[:width]
        repeated = 0
        offset = 0
        while compact.startswith(unit, offset):
            repeated += width
            offset += width
        if repeated >= max(len(compact) * 0.7, min_chars * 2):
            return False, "starts with a repeated fragment"

    return True, "looks like readable speech"


async def run_command(
    cmd: Sequence[str],
    cwd: Path,
    env: Dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
    timeout: float,
    dry_run: bool,
) -> CommandResult:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    if dry_run:
        stdout_path.write_text(format_cmd(cmd) + "\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return CommandResult(
            returncode=0,
            seconds=0.0,
            timed_out=False,
            command=list(cmd),
            stdout_path=path_for_json(stdout_path),
            stderr_path=path_for_json(stderr_path),
        )

    timed_out = False
    with stdout_path.open("wb") as stdout_fp, stderr_path.open("wb") as stderr_fp:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(cwd),
            env=env,
            stdout=stdout_fp,
            stderr=stderr_fp,
        )
        try:
            if timeout > 0:
                await asyncio.wait_for(proc.wait(), timeout=timeout)
            else:
                await proc.wait()
        except asyncio.TimeoutError:
            timed_out = True
            proc.kill()
            await proc.wait()
    return CommandResult(
        returncode=proc.returncode,
        seconds=time.monotonic() - start,
        timed_out=timed_out,
        command=list(cmd),
        stdout_path=path_for_json(stdout_path),
        stderr_path=path_for_json(stderr_path),
    )


def load_keep_per_layer(mask_path: Path) -> Optional[int]:
    try:
        with mask_path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
    except (OSError, json.JSONDecodeError):
        return None
    value = data.get("keep_per_layer")
    if isinstance(value, int):
        return value
    layers = data.get("layers")
    if isinstance(layers, list) and layers:
        first = layers[0]
        if isinstance(first, dict) and isinstance(first.get("keep"), list):
            return len(first["keep"])
    return None


def build_c4_dataset(args: argparse.Namespace, dataset_path: Path) -> None:
    builder = SCRIPT_DIR / "build_c4_prune_dataset.py"
    cmd = [
        sys.executable,
        str(builder),
        "--out",
        str(dataset_path),
        "--source",
        args.c4_source,
        "--max-docs",
        str(args.c4_max_docs),
        "--max-chars-per-doc",
        str(args.c4_max_chars_per_doc),
        "--min-chars",
        str(args.c4_min_chars),
        "--skip",
        str(args.c4_skip),
    ]
    if args.c4_input:
        cmd.extend(["--input", args.c4_input])
    print(format_cmd(cmd), flush=True)
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


async def run_ratio(
    args: argparse.Namespace,
    spec: RatioSpec,
    gpu_queue: "asyncio.Queue[str]",
    results_jsonl: Path,
    write_lock: asyncio.Lock,
) -> SweepResult:
    gpu = await gpu_queue.get()
    try:
        ratio_dir = args.work_dir / spec.tag
        ratio_dir.mkdir(parents=True, exist_ok=True)
        mask_path = ratio_dir / "expert-mask.json"
        commands_path = ratio_dir / "commands.json"

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        lock_path = args.work_dir / "locks" / f"ds4-gpu-{lock_tag(gpu)}.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        env["DS4_LOCK_FILE"] = str(lock_path)
        env_prefix = {"CUDA_VISIBLE_DEVICES": gpu, "DS4_LOCK_FILE": str(lock_path)}

        common_prune_cmd = [
            str(args.binary),
            "--cuda",
            "-m",
            str(args.model),
            "--expert-prune-dataset",
            str(args.dataset),
            "--expert-prune-out",
            str(mask_path),
            "--expert-prune-keep-ratio",
            spec.ratio_arg,
            "--expert-prune-max-tokens",
            str(args.expert_prune_max_tokens),
            "--ctx",
            str(args.prune_ctx),
        ]
        if args.expert_prune_max_prompts > 0:
            common_prune_cmd.extend(["--expert-prune-max-prompts", str(args.expert_prune_max_prompts)])
        if args.power:
            common_prune_cmd.extend(["--power", str(args.power)])
        if args.quality:
            common_prune_cmd.append("--quality")

        test_cmd = [
            str(args.binary),
            "--cuda",
            "-m",
            str(args.model),
            "--expert-mask",
            str(mask_path),
            "--nothink",
            "-n",
            str(args.test_tokens),
            "--ctx",
            str(args.test_ctx),
            "-p",
            args.prompt,
        ]
        if args.seed is not None:
            test_cmd.extend(["--seed", str(args.seed)])
        if args.temperature is not None:
            test_cmd.extend(["--temp", str(args.temperature)])
        if args.power:
            test_cmd.extend(["--power", str(args.power)])
        if args.quality:
            test_cmd.append("--quality")

        command_payload = {
            "keep_percent": spec.percent,
            "keep_ratio": spec.ratio,
            "gpu": gpu,
            "prune": format_cmd(common_prune_cmd, env_prefix),
            "test": format_cmd(test_cmd, env_prefix),
        }
        commands_path.write_text(json.dumps(command_payload, indent=2) + "\n", encoding="utf-8")

        print(f"[{spec.tag}] gpu={gpu} prune -> {path_for_json(mask_path)}", flush=True)
        prune_stdout = ratio_dir / "prune.stdout.log"
        prune_stderr = ratio_dir / "prune.stderr.log"
        if args.resume and mask_path.exists() and not args.force:
            prune_result = CommandResult(
                returncode=0,
                seconds=0.0,
                timed_out=False,
                command=common_prune_cmd,
                stdout_path=path_for_json(prune_stdout),
                stderr_path=path_for_json(prune_stderr),
            )
        else:
            prune_result = await run_command(
                common_prune_cmd,
                cwd=REPO_ROOT,
                env=env,
                stdout_path=prune_stdout,
                stderr_path=prune_stderr,
                timeout=args.prune_timeout,
                dry_run=args.dry_run,
            )

        keep_per_layer = load_keep_per_layer(mask_path)
        passed = False
        reason = "not tested"
        test_result = CommandResult(
            returncode=None,
            seconds=0.0,
            timed_out=False,
            command=test_cmd,
            stdout_path=path_for_json(ratio_dir / "test.stdout.txt"),
            stderr_path=path_for_json(ratio_dir / "test.stderr.log"),
        )

        if prune_result.returncode != 0 or prune_result.timed_out:
            reason = "prune command timed out" if prune_result.timed_out else "prune command failed"
        elif not args.dry_run and keep_per_layer is None:
            reason = "mask JSON is missing keep_per_layer"
        else:
            print(f"[{spec.tag}] gpu={gpu} smoke test", flush=True)
            test_stdout = ratio_dir / "test.stdout.txt"
            test_stderr = ratio_dir / "test.stderr.log"
            test_result = await run_command(
                test_cmd,
                cwd=REPO_ROOT,
                env=env,
                stdout_path=test_stdout,
                stderr_path=test_stderr,
                timeout=args.test_timeout,
                dry_run=args.dry_run,
            )
            if test_result.returncode != 0 or test_result.timed_out:
                reason = "test command timed out" if test_result.timed_out else "test command failed"
            elif args.dry_run:
                passed = True
                reason = "dry run"
            else:
                output_text = read_text(test_stdout)
                passed, reason = looks_like_normal_speech(
                    output_text,
                    min_chars=args.min_output_chars,
                    min_wordish_ratio=args.min_wordish_ratio,
                )

        output_text = read_text(Path(test_result.stdout_path) if Path(test_result.stdout_path).is_absolute() else REPO_ROOT / test_result.stdout_path)
        result = SweepResult(
            keep_percent=spec.percent,
            keep_ratio=spec.ratio,
            gpu=gpu,
            passed=passed,
            reason=reason,
            keep_per_layer=keep_per_layer,
            mask_path=path_for_json(mask_path),
            ratio_dir=path_for_json(ratio_dir),
            prune_returncode=prune_result.returncode,
            test_returncode=test_result.returncode,
            prune_seconds=round(prune_result.seconds, 3),
            test_seconds=round(test_result.seconds, 3),
            prune_stdout=prune_result.stdout_path,
            prune_stderr=prune_result.stderr_path,
            test_stdout=test_result.stdout_path,
            test_stderr=test_result.stderr_path,
            output_preview=single_line(output_text),
        )
        (ratio_dir / "result.json").write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")

        async with write_lock:
            with results_jsonl.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

        status = "PASS" if result.passed else "FAIL"
        print(f"[{spec.tag}] {status}: {result.reason} ({result.output_preview})", flush=True)
        return result
    finally:
        gpu_queue.put_nowait(gpu)


def write_summary(results: Sequence[SweepResult], work_dir: Path) -> None:
    ordered = sorted(results, key=lambda item: item.keep_percent, reverse=True)
    csv_path = work_dir / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "keep_percent",
                "keep_ratio",
                "keep_per_layer",
                "passed",
                "reason",
                "gpu",
                "prune_seconds",
                "test_seconds",
                "mask_path",
                "test_stdout",
                "output_preview",
            ],
        )
        writer.writeheader()
        for row in ordered:
            writer.writerow({
                "keep_percent": row.keep_percent,
                "keep_ratio": row.keep_ratio,
                "keep_per_layer": row.keep_per_layer,
                "passed": row.passed,
                "reason": row.reason,
                "gpu": row.gpu,
                "prune_seconds": row.prune_seconds,
                "test_seconds": row.test_seconds,
                "mask_path": row.mask_path,
                "test_stdout": row.test_stdout,
                "output_preview": row.output_preview,
            })

    passed = sorted([item for item in results if item.passed], key=lambda item: item.keep_percent)
    best = passed[0] if passed else None
    md_lines = [
        "# DS4 Expert Keep Ratio Sweep",
        "",
        f"Work dir: `{path_for_json(work_dir)}`",
        "",
    ]
    if best:
        md_lines.append(
            f"Lowest keep ratio that passed the readable-output smoke test: "
            f"**{best.keep_percent}%** ({best.keep_per_layer} experts/layer)."
        )
    else:
        md_lines.append("No keep ratio passed the readable-output smoke test.")
    md_lines.extend([
        "",
        "This is a coarse smoke test. Check `test.stdout.txt` for any ratio near the cutoff.",
        "",
        "| keep | experts/layer | pass | reason | output preview |",
        "| ---: | ---: | :---: | --- | --- |",
    ])
    for row in ordered:
        preview = row.output_preview.replace("|", "\\|")
        md_lines.append(
            f"| {row.keep_percent}% | {row.keep_per_layer if row.keep_per_layer is not None else ''} "
            f"| {'yes' if row.passed else 'no'} | {row.reason} | {preview} |"
        )
    md_lines.append("")
    (work_dir / "summary.md").write_text("\n".join(md_lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate OMP expert masks for keep ratios 100%, 90%, ..., 10%, "
            "then run a short masked inference smoke test for each ratio."
        )
    )
    parser.add_argument("--binary", default="./ds4", help="Path to the ds4 CLI binary, relative to repo root by default.")
    parser.add_argument("--model", default="ds4flash.gguf", help="Model GGUF path, relative to repo root by default.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="C4 calibration text file.")
    parser.add_argument("--work-dir", help="Output directory. Default: gguf-tools/expert-prune/sweep-runs/TIMESTAMP")
    parser.add_argument("--gpus", default="0,1,2,3,5,7", help="Comma-separated CUDA device ids.")
    parser.add_argument("--ratios", help="Optional comma-separated keep ratios or percents, for example 1.0,0.875,0.75 or 100,90.")
    parser.add_argument("--start-percent", type=int, default=100)
    parser.add_argument("--stop-percent", type=int, default=10)
    parser.add_argument("--step-percent", type=int, default=10)
    parser.add_argument("--expert-prune-max-tokens", type=int, default=131072)
    parser.add_argument("--expert-prune-max-prompts", type=int, default=0)
    parser.add_argument("--prune-ctx", type=int, default=32768)
    parser.add_argument("--test-ctx", type=int, default=4096)
    parser.add_argument("--test-tokens", type=int, default=80)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--power", type=int, help="Optional ds4 --power value.")
    parser.add_argument("--quality", action="store_true", help="Pass --quality to ds4.")
    parser.add_argument("--min-output-chars", type=int, default=16)
    parser.add_argument("--min-wordish-ratio", type=float, default=0.45)
    parser.add_argument("--prune-timeout", type=float, default=0.0, help="Seconds; 0 disables timeout.")
    parser.add_argument("--test-timeout", type=float, default=1800.0, help="Seconds; 0 disables timeout.")
    parser.add_argument("--force", action="store_true", help="Regenerate masks even if they already exist.")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Do not reuse existing masks in --work-dir.")
    parser.add_argument("--dry-run", action="store_true", help="Write commands and summaries without running ds4.")
    parser.add_argument("--build-c4", action="store_true", help="Build --dataset first using build_c4_prune_dataset.py.")
    parser.add_argument("--c4-source", choices=("hf", "jsonl", "text"), default="hf")
    parser.add_argument("--c4-input", help="Input path for --c4-source jsonl/text.")
    parser.add_argument("--c4-max-docs", type=int, default=512)
    parser.add_argument("--c4-max-chars-per-doc", type=int, default=8192)
    parser.add_argument("--c4-min-chars", type=int, default=200)
    parser.add_argument("--c4-skip", type=int, default=0)
    parser.set_defaults(resume=True)
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> int:
    specs = parse_ratios(args)
    gpus = parse_gpus(args.gpus)
    gpu_queue: "asyncio.Queue[str]" = asyncio.Queue()
    for gpu in gpus:
        gpu_queue.put_nowait(gpu)

    results_jsonl = args.work_dir / "results.jsonl"
    if results_jsonl.exists():
        results_jsonl.unlink()
    args.work_dir.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(REPO_ROOT),
        "dataset": str(args.dataset),
        "model": str(args.model),
        "binary": str(args.binary),
        "gpus": gpus,
        "ratios": [asdict(spec) for spec in specs],
        "expert_prune_max_tokens": args.expert_prune_max_tokens,
        "expert_prune_max_prompts": args.expert_prune_max_prompts,
        "prune_ctx": args.prune_ctx,
        "test_ctx": args.test_ctx,
        "test_tokens": args.test_tokens,
        "prompt": args.prompt,
    }
    (args.work_dir / "run.json").write_text(json.dumps(run_meta, indent=2) + "\n", encoding="utf-8")

    write_lock = asyncio.Lock()
    tasks = [run_ratio(args, spec, gpu_queue, results_jsonl, write_lock) for spec in specs]
    results = await asyncio.gather(*tasks)
    write_summary(results, args.work_dir)
    print(f"summary: {path_for_json(args.work_dir / 'summary.md')}", flush=True)
    return 0


def main() -> int:
    args = parse_args()
    args.binary = resolve_repo_path(args.binary)
    args.model = resolve_repo_path(args.model)
    args.dataset = Path(args.dataset).expanduser()
    if not args.dataset.is_absolute():
        args.dataset = (REPO_ROOT / args.dataset).resolve()
    else:
        args.dataset = args.dataset.resolve()

    if args.work_dir:
        args.work_dir = resolve_work_path(args.work_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.work_dir = SCRIPT_DIR / "sweep-runs" / stamp

    if args.build_c4:
        build_c4_dataset(args, args.dataset)
    elif not args.dataset.exists():
        raise SystemExit(
            f"calibration dataset not found: {args.dataset}\n"
            f"Run with --build-c4, or create it with {SCRIPT_DIR / 'build_c4_prune_dataset.py'}."
        )

    if not args.dry_run:
        if not args.binary.exists():
            raise SystemExit(f"ds4 binary not found: {args.binary}")
        if not args.model.exists():
            raise SystemExit(f"model not found: {args.model}")

    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
