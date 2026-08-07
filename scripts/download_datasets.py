"""Tải dataset chuẩn từ kho dữ liệu học thuật (UCI / GitHub mirror) về data/datasets.

Chạy:
    python scripts/download_datasets.py

Các bộ dữ liệu (đều là benchmark chuẩn trong nghiên cứu y học/học máy):
    1. Pima Indians Diabetes (UCI)     — nhị phân: có đái tháo đường hay không
    2. Cleveland Heart Disease (UCI)   — nhị phân: bệnh tim hay không

Lưu ý: đây là dữ liệu CẮT NGANG (mỗi bệnh nhân một dòng), dùng để huấn luyện
và đối chứng pipeline ML; không phải chuỗi thời gian. Dữ liệu dọc có nhãn thật
(MIMIC, OhioT1DM...) yêu cầu đăng ký quyền truy cập — xem docs/06.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

DATASETS: dict[str, dict[str, str]] = {
    "pima_diabetes.csv": {
        "url": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
        "note": "Pima Indians Diabetes — UCI",
        "columns": [
            "pregnancies", "glucose", "bp", "skin_thickness", "insulin",
            "bmi", "diabetes_pedigree", "age", "outcome",
        ],
    },
    "heart_cleveland.csv": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
        "note": "Heart Disease (Cleveland) — UCI",
        "columns": [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
            "exang", "oldpeak", "slope", "ca", "thal", "num",
        ],
    },
}


def download(out_dir: Path, force: bool = False) -> list[tuple[str, str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[tuple[str, str]] = []
    for fname, info in DATASETS.items():
        target = out_dir / fname
        if target.exists() and not force:
            results.append((fname, "đã có (bỏ qua)"))
            continue
        print(f"[{fname}] tải từ {info['url']} ...", flush=True)
        try:
            df = pd.read_csv(info["url"], header=None, names=info["columns"])
        except Exception as exc:  # noqa: BLE001
            results.append((fname, f"LỖI: {exc}"))
            continue
        df.to_csv(target, index=False)
        results.append((fname, f"OK — {df.shape[0]} dòng x {df.shape[1]} cột"))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Tải dataset chính thống về data/datasets")
    parser.add_argument("--out", type=str, default="data/datasets")
    parser.add_argument("--force", action="store_true", help="tải lại kể cả khi đã có")
    args = parser.parse_args()
    out = Path(args.out)
    for fname, msg in download(out, force=args.force):
        print(f"  {fname:<24} {msg}")
    print("Hoàn tất. Xem docs/06_Bao_cao_huong_train_va_gioi_han.md để biết hướng dùng.")


if __name__ == "__main__":
    sys.exit(main())
