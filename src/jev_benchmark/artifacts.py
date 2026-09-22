from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def create_run_directory(task: str, output_dir: Path | None) -> tuple[Path, str]:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = output_dir or Path("results") / task / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path, run_id


def write_metadata(
    output_dir: Path,
    *,
    task: str,
    dataset: Path,
    providers: list[str],
    repetitions: int,
    run_id: str,
) -> Path:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    metadata = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "task": task,
        "dataset": str(dataset),
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "providers": providers,
        "repetitions": repetitions,
        "git_commit": commit,
    }
    path = output_dir / "metadata.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path
