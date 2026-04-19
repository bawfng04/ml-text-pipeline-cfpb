#!/usr/bin/env python
"""
CLI entrypoint to execute a notebook pipeline.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute the CFPB notebook pipeline via jupyter nbconvert."
    )
    parser.add_argument(
        "--notebook",
        default="notebooks/main_pipeline.ipynb",
        help="Path to the notebook to execute.",
    )
    parser.add_argument(
        "--output-name",
        default="main_pipeline.executed",
        help="Base name for executed notebook output (without extension).",
    )
    parser.add_argument(
        "--output-dir",
        default="notebooks",
        help="Directory where executed notebook is written.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=-1,
        help="Cell execution timeout in seconds (-1 to disable timeout).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    notebook = Path(args.notebook)
    output_dir = Path(args.output_dir)

    if not notebook.exists():
        print(f"Notebook not found: {notebook}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        str(notebook),
        "--output",
        args.output_name,
        "--output-dir",
        str(output_dir),
        f"--ExecutePreprocessor.timeout={args.timeout}",
    ]

    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print(
            "Unable to run Jupyter. Install Jupyter Notebook/nbconvert first.",
            file=sys.stderr,
        )
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"Notebook execution failed with exit code {exc.returncode}.", file=sys.stderr)
        return exc.returncode

    print("Notebook execution completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
