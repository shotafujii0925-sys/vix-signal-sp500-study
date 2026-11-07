# -*- coding: utf-8 -*-
"""スナップショットから全段階を通しで再実行する（オンライン取得はしない）。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = ["run_01_strategies.py", "run_02_signal.py", "run_03_regression.py", "run_04_conformal.py",
         "run_05_figures.py", "build_report.py"]


def main() -> None:
    log_dir = ROOT / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    for step in STEPS:
        print(f"実行中: {step}", flush=True)
        with open(log_dir / f"{Path(step).stem}.log", "w", encoding="utf-8") as log:
            subprocess.run([sys.executable, "-X", "utf8", step], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                           check=True)
    print("完了")


if __name__ == "__main__":
    main()
