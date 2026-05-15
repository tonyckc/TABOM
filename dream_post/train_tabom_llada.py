#!/usr/bin/env python3
"""
TABOM training entry for LLaDA (HF CausalLM) + LoRA; teacher is groundtruth only.

Equivalent to ``train_tabom.py`` with
``--teacher groundtruth --student_backend llada`` and default
``--dream_model GSAI-ML/LLaDA-8B-Instruct`` (overridable on the CLI).

Example under ``dream_tabom_release/``::

    bash dream_post/run_tabom_examples.sh llada_prm12k train
"""

from __future__ import annotations

import pathlib
import runpy
import sys


def main() -> None:
    root = pathlib.Path(__file__).resolve().parent
    preset_pairs = [
        ("--teacher", "groundtruth"),
        ("--student_backend", "llada"),
        ("--dream_model", "GSAI-ML/LLaDA-8B-Instruct"),
    ]
    argv = sys.argv[1:]
    keys = set()
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--") and "=" not in a:
            keys.add(a[2:])
            i += 2
        elif a.startswith("--") and "=" in a:
            keys.add(a.split("=", 1)[0][2:])
            i += 1
        else:
            i += 1

    merged: list[str] = []
    for flag, val in preset_pairs:
        key = flag[2:]
        if key not in keys:
            merged.extend([flag, val])
    sys.argv = [sys.argv[0]] + merged + argv
    runpy.run_path(str(root / "train_tabom.py"), run_name="__main__")


if __name__ == "__main__":
    main()
