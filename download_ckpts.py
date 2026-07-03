#!/usr/bin/env python3
"""Download the DR-LfD DMG checkpoints from the Hugging Face Hub.

The release weights live in the model repo ``ConvexBuchi/dr-lfd-ckpt`` with the
same layout as this repo's ``ckpt/`` directory::

    dmg_threading/{policy.ckpt, equi.pth, biop.pth}
    dmg_assembly/{policy.ckpt, equi.pth, biop.pth}

Running this script mirrors that layout into ``<repo>/ckpt`` so the launch
configs resolve without any manual copying. Re-runs are cheap: the Hub client
verifies each file's hash and only fetches what is missing or changed.

Usage::

    python download_ckpts.py                 # both tasks (~2.9 GB)
    python download_ckpts.py --task threading # threading only (~1.1 GB)
    python download_ckpts.py --dest /some/ckpt

A Hugging Face token is read from the standard cache / ``HF_TOKEN`` env var. If
the repo is private and no valid token is found, the download fails loudly
rather than silently producing an empty tree.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "ConvexBuchi/dr-lfd-ckpt"
REPO_TYPE = "model"

# task name -> subdirectory in both the remote repo and the local ckpt tree
TASK_DIRS = {
    "threading": "dmg_threading",
    "assembly": "dmg_assembly",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--task",
        choices=[*TASK_DIRS, "all"],
        default="all",
        help="Which DMG route's weights to fetch (default: all).",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path(__file__).resolve().parent / "ckpt",
        help="Destination ckpt directory (default: <repo-root>/ckpt).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tasks = list(TASK_DIRS) if args.task == "all" else [args.task]
    allow_patterns = [f"{TASK_DIRS[t]}/*" for t in tasks]

    args.dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {tasks} from {REPO_ID} -> {args.dest}")

    snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        local_dir=str(args.dest),
        allow_patterns=allow_patterns,
    )

    print("\nDone. Fetched files:")
    for task in tasks:
        task_dir = args.dest / TASK_DIRS[task]
        for f in sorted(task_dir.glob("*")):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {size_mb:8.1f} MB  {f.relative_to(args.dest)}")


if __name__ == "__main__":
    main()
