# Copyright (c) 2025 Tylt LLC. All rights reserved.
# Licensed for research use only. Commercial use requires a license from Tylt LLC.
# Contact: hello@claimhawk.app | See LICENSE for terms.

"""Merge task datasets into a routing JSONL with adapter labels."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass
class SourceSpec:
    label: str
    jsonl: Path
    image_root: Path | None
    tasks: set[str]
    limit: int | None
    name: str | None = None


def parse_source(spec: str) -> SourceSpec:
    """Parse a single --source entry of the form key=val,key=val."""
    parts = {}
    for chunk in spec.split(","):
        if not chunk.strip():
            continue
        if "=" not in chunk:
            raise ValueError(f"Invalid source chunk (missing '='): {chunk}")
        key, value = chunk.split("=", 1)
        parts[key.strip()] = value.strip()

    if "label" not in parts or "jsonl" not in parts:
        raise ValueError(f"--source requires at least label and jsonl: {spec}")

    tasks = {t.strip() for t in parts.get("tasks", "").split(",") if t.strip()}
    limit = int(parts["limit"]) if "limit" in parts else None
    image_root = Path(parts["image_root"]).expanduser() if "image_root" in parts else None
    jsonl_path = Path(parts["jsonl"]).expanduser()
    name = parts.get("name")

    return SourceSpec(
        label=parts["label"],
        jsonl=jsonl_path,
        image_root=image_root,
        tasks=tasks,
        limit=limit,
        name=name,
    )


def load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Stream JSON objects from a JSONL file."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def resolve_image(image: str, jsonl_path: Path, image_root: Path | None) -> Path:
    """Resolve image path using provided root, jsonl parent, and fallback to grandparent if needed."""
    img_path = Path(image)
    if img_path.is_absolute():
        return img_path

    candidates: list[Path] = []
    if image_root:
        candidates.append(image_root)
    # Prefer JSONL parent
    candidates.append(jsonl_path.parent)
    # If the first part matches the JSONL parent name, also try the grandparent (calendar datasets)
    if img_path.parts and img_path.parts[0] == jsonl_path.parent.name:
        candidates.append(jsonl_path.parent.parent)

    for base in candidates:
        candidate = (base / img_path).resolve()
        if candidate.exists():
            return candidate

    # Fallback to first candidate even if missing
    if candidates:
        return (candidates[0] / img_path).resolve()
    return img_path.resolve()


def process_source(spec: SourceSpec) -> list[dict[str, Any]]:
    """Load rows from a source JSONL, apply filters, and add routing labels."""
    rows: list[dict[str, Any]] = []
    count = 0
    for record in load_jsonl(spec.jsonl):
        if spec.tasks:
            task_type = record.get("metadata", {}).get("task_type")
            if task_type not in spec.tasks:
                continue
        enriched = dict(record)
        enriched["adapter"] = spec.label
        enriched["source_dataset"] = spec.name or spec.jsonl.stem
        enriched["image_path"] = str(resolve_image(record.get("image", ""), spec.jsonl, spec.image_root))
        rows.append(enriched)
        count += 1
        if spec.limit is not None and count >= spec.limit:
            break
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge task datasets into routing JSONL.")
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Source spec: label=NAME,jsonl=PATH[,image_root=PATH][,tasks=comma,sep][,limit=N][,name=ALIAS]",
    )
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle merged rows.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for shuffling.")
    args = parser.parse_args(argv)

    sources = [parse_source(spec) for spec in args.source]

    merged: list[dict[str, Any]] = []
    for spec in sources:
        rows = process_source(spec)
        merged.extend(rows)
        print(
            f"[source:{spec.label}] kept {len(rows)} rows "
            f"from {spec.jsonl} "
            f"(tasks filter: {','.join(sorted(spec.tasks)) if spec.tasks else 'none'})"
        )

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(merged)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(merged)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
