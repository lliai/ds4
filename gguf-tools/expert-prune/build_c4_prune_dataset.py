#!/usr/bin/env python3
"""Build a marker-separated C4 calibration file for DS4 expert pruning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Iterator


MARKER_PREFIX = "===== DS4_IMATRIX_PROMPT"


def normalize_doc(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\0", "")
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines).strip()
    while "\n\n\n\n" in text:
        text = text.replace("\n\n\n\n", "\n\n\n")
    return text


def clip_doc(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    split = max(cut.rfind("\n"), cut.rfind(" "), cut.rfind("\t"))
    if split > max_chars * 3 // 4:
        cut = cut[:split]
    return cut.rstrip()


def iter_hf_c4(args: argparse.Namespace) -> Iterator[str]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "build_c4_prune_dataset.py --source hf requires the `datasets` package. "
            "Install it in your Python environment or use --source jsonl/--source text."
        ) from exc

    kwargs = {"split": args.split, "streaming": args.streaming}
    if args.config:
        dataset = load_dataset(args.dataset, args.config, **kwargs)
    else:
        dataset = load_dataset(args.dataset, **kwargs)

    if args.shuffle_buffer > 0:
        if not hasattr(dataset, "shuffle"):
            raise SystemExit("Loaded dataset does not support shuffle(); rerun with --shuffle-buffer 0.")
        if args.streaming:
            dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
        else:
            dataset = dataset.shuffle(seed=args.seed)

    skipped = 0
    for row in dataset:
        if skipped < args.skip:
            skipped += 1
            continue
        value = row.get(args.text_field) if isinstance(row, dict) else None
        if isinstance(value, str):
            yield value


def iter_jsonl(args: argparse.Namespace) -> Iterator[str]:
    if not args.input:
        raise SystemExit("--source jsonl requires --input FILE")
    with Path(args.input).open("r", encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{args.input}:{line_no}: invalid JSONL: {exc}") from exc
            value = row.get(args.text_field) if isinstance(row, dict) else None
            if isinstance(value, str):
                yield value


def iter_text(args: argparse.Namespace) -> Iterator[str]:
    if not args.input:
        raise SystemExit("--source text requires --input FILE")
    block: list[str] = []
    with Path(args.input).open("r", encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                block.append(line.rstrip("\n"))
            elif block:
                yield "\n".join(block)
                block.clear()
    if block:
        yield "\n".join(block)


def iter_source(args: argparse.Namespace) -> Iterable[str]:
    if args.source == "hf":
        return iter_hf_c4(args)
    if args.source == "jsonl":
        return iter_jsonl(args)
    if args.source == "text":
        return iter_text(args)
    raise SystemExit(f"unknown source: {args.source}")


def write_dataset(args: argparse.Namespace) -> dict:
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest) if args.manifest else out_path.with_suffix(out_path.suffix + ".manifest.json")

    docs = 0
    chars = 0
    with out_path.open("w", encoding="utf-8") as fp:
        for raw in iter_source(args):
            text = clip_doc(normalize_doc(raw), args.max_chars_per_doc)
            if len(text) < args.min_chars:
                continue
            fp.write(f"{MARKER_PREFIX} {docs} source=c4 split={args.split} =====\n")
            fp.write(text)
            fp.write("\n\n")
            docs += 1
            chars += len(text)
            if args.max_docs > 0 and docs >= args.max_docs:
                break

    if docs == 0:
        raise SystemExit("no calibration documents were written; lower --min-chars or check the input source")

    manifest = {
        "format": "ds4-expert-prune-c4-dataset-v1",
        "output": str(out_path),
        "source": args.source,
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "text_field": args.text_field,
        "docs": docs,
        "chars": chars,
        "max_chars_per_doc": args.max_chars_per_doc,
        "min_chars": args.min_chars,
    }
    with manifest_path.open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2, sort_keys=True)
        fp.write("\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a DS4_IMATRIX_PROMPT separated C4 text file for --expert-prune-dataset."
    )
    parser.add_argument("--out", required=True, help="Output calibration text path.")
    parser.add_argument("--manifest", help="Optional manifest JSON path. Default: OUT.manifest.json")
    parser.add_argument("--source", choices=("hf", "jsonl", "text"), default="hf")
    parser.add_argument("--input", help="Input path for --source jsonl or --source text.")
    parser.add_argument("--dataset", default="allenai/c4", help="Hugging Face dataset name for --source hf.")
    parser.add_argument("--config", default="en", help="Hugging Face dataset config/name.")
    parser.add_argument("--split", default="train", help="Dataset split.")
    parser.add_argument("--text-field", default="text", help="Field containing document text.")
    parser.add_argument("--max-docs", type=int, default=512, help="Maximum documents to write. <=0 means unlimited.")
    parser.add_argument("--max-chars-per-doc", type=int, default=8192, help="Clip each document to this many chars.")
    parser.add_argument("--min-chars", type=int, default=200, help="Skip shorter documents.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--skip", type=int, default=0, help="Skip this many source rows before writing.")
    parser.add_argument("--shuffle-buffer", type=int, default=10000, help="Shuffle buffer for HF datasets. 0 disables.")
    parser.add_argument("--no-streaming", dest="streaming", action="store_false", help="Load HF dataset non-streaming.")
    parser.set_defaults(streaming=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = write_dataset(args)
    print(
        f"wrote {manifest['docs']} C4 calibration docs, "
        f"{manifest['chars']} chars -> {manifest['output']}"
    )


if __name__ == "__main__":
    main()
