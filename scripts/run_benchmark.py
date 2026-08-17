"""Chạy benchmark đa mô hình theo Experimental Protocol (notes/02).

Chạy:
    python scripts/run_benchmark.py                          # 6 model, 5 seed
    python scripts/run_benchmark.py --models lr rf           # chọn model
    python scripts/run_benchmark.py --seeds 42 52            # chọn seed
    python scripts/run_benchmark.py --out experiments/run1   # thư mục riêng

Output:
    experiments/
    ├── EXP-ML-<MODEL>-<SEED>/   evidence package từng lần chạy
    ├── summary.json / summary.md / summary.csv
    └── README.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.models import available_models  # noqa: E402
from src.experiments.protocol import SEEDS  # noqa: E402
from src.experiments.runner import run_benchmark  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark đa mô hình theo protocol")
    parser.add_argument("--data", type=str, default="data/datasets/nhanes_2017_2018.csv")
    parser.add_argument("--models", nargs="+", default=None,
                        help=f"Mặc định: {available_models()}")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--out", type=str, default="experiments")
    args = parser.parse_args()

    keys = args.models or available_models()
    print(f"Dataset : {args.data}")
    print(f"Models  : {keys}")
    print(f"Seeds   : {args.seeds}")
    print("Bắt đầu benchmark...")

    summary = run_benchmark(
        dataset_path=args.data,
        model_keys=keys,
        seeds=args.seeds,
        out_dir=Path(args.out),
    )
    print("\n=== KẾT QUẢ (mean ± std) ===")
    print(f"{'Model':<14}{'ROC-AUC':>12}{'PR-AUC':>12}{'F1':>10}{'Brier':>10}")
    for key, m in summary["models"].items():
        print(
            f"{key:<14}"
            f"{m['roc_auc']['mean']:>10.3f}±{m['roc_auc']['std']:.3f}"
            f"{m['pr_auc']['mean']:>10.3f}±{m['pr_auc']['std']:.3f}"
            f"{m['f1']['mean']:>8.3f}±{m['f1']['std']:.3f}"
            f"{m['brier']['mean']:>8.3f}±{m['brier']['std']:.3f}"
        )
    print(f"\nKết quả lưu tại: {args.out}/")


if __name__ == "__main__":
    main()