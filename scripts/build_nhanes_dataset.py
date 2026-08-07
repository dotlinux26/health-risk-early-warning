"""Tải NHANES (CDC) và dựng dataset thật cho mô hình cảnh báo sớm.

Nguồn: Continuous NHANES 2017-2018 (CDC/NCHS), dữ liệu lâm sàng khảo sát toàn
dân Mỹ, miễn phí. Mỗi người tham gia được đo nhiều lần trong kỳ khám.

Chạy:
    python scripts/build_nhanes_dataset.py

Kết quả: data/datasets/nhanes_2017_2018.csv với các chỉ số TRÙNG schema hệ thống:
    systolic_bp, diastolic_bp, heart_rate, glucose_fasting, hba1c, creatinine, bmi
và nhãn bệnh thật (tăng huyết áp / đái tháo đường) xác định bằng ngưỡng lâm sàng
hoặc đang dùng thuốc.

Lưu ý khoa học: nhãn "tăng huyết áp" được định nghĩa một phần bằng chính huyết
áp đo được — mô hình học sẽ đạt AUC cao nhưng cần đọc kết quả với ngữ cảnh này
(xem docs/06). NHANES là khảo sát cắt ngang; để có chuỗi thời gian thực sự cần
ghép nhiều kỳ khám hoặc nguồn dữ liệu dọc (MIMIC, UK Biobank).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/{name}.XPT"

FILES = ["DEMO_J", "BPX_J", "BPQ_J", "GLU_J", "GHB_J", "BIOPRO_J", "BMX_J"]

# Cột cần lấy từ từng file (theo codebook NHANES 2017-2018)
KEEP = {
    "DEMO_J": ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDEXPRG"],
    "BPX_J": ["SEQN", "BPXSY1", "BPXSY2", "BPXSY3", "BPXDI1", "BPXDI2", "BPXDI3", "BPXPLS"],
    "BPQ_J": ["SEQN", "BPQ020", "BPQ050A"],
    "GLU_J": ["SEQN", "LBXGLU"],
    "GHB_J": ["SEQN", "LBXGH"],
    "BIOPRO_J": ["SEQN", "LBXSCR"],
    "BMX_J": ["SEQN", "BMXBMI"],
}


def fetch(name: str, cache_dir: Path) -> Path:
    target = cache_dir / f"{name}.XPT"
    if not target.exists():
        import urllib.request

        url = BASE.format(name=name)
        print(f"  tải {name} ...", flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, target.open("wb") as f:
            f.write(resp.read())
    return target


def mean_valid(series_df: pd.DataFrame, sys_cols: list[str], lo: float, hi: float) -> pd.Series:
    vals = series_df[sys_cols].to_numpy(dtype=float)
    valid = (vals >= lo) & (vals <= hi) & ~np.isnan(vals)
    out = np.full(len(series_df), np.nan)
    for i in range(len(series_df)):
        v = vals[i, valid[i]]
        if v.size:
            out[i] = v.mean()
    return pd.Series(out, index=series_df.index)


def build(out_csv: Path, cache_dir: Path) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for name in FILES:
        df = pd.read_sas(fetch(name, cache_dir), format="xport")
        frames.append(df[KEEP[name]])
    merged = frames[0]
    for df in frames[1:]:
        merged = merged.merge(df, on="SEQN", how="left")

    data = pd.DataFrame({"seqn": merged["SEQN"]})
    data["age"] = merged["RIDAGEYR"]
    data["sex"] = merged["RIAGENDR"]
    data["systolic_bp"] = mean_valid(merged, ["BPXSY1", "BPXSY2", "BPXSY3"], 40, 300)
    data["diastolic_bp"] = mean_valid(merged, ["BPXDI1", "BPXDI2", "BPXDI3"], 20, 160)
    data["heart_rate"] = merged["BPXPLS"]  # nhịp 60 giây
    data["glucose_fasting"] = merged["LBXGLU"] / 18.016  # mg/dL -> mmol/L
    data["hba1c"] = merged["LBXGH"]  # %
    data["creatinine"] = merged["LBXSCR"]  # mg/dL
    data["bmi"] = merged["BMXBMI"]

    # --- Nhãn lâm sàng thật ------------------------------------------------
    # Tăng huyết áp: HA đo cao HOẶC đang dùng thuốc huyết áp (BPQ050=1)
    htn = (data["systolic_bp"] >= 140) | (data["diastolic_bp"] >= 90)
    htn |= merged["BPQ050A"] == 1
    # Đái tháo đường: HbA1c >= 6.5% HOẶC đường huyết lúc đói >= 7.0 mmol/L
    dm = (data["hba1c"] >= 6.5) | (data["glucose_fasting"] >= 7.0)
    data["label_htn"] = htn.astype(int)
    data["label_dm"] = dm.astype(int)
    data["label"] = (htn | dm).astype(int)

    # --- Lọc đối tượng ------------------------------------------------------
    adult = (data["age"] >= 20) & (data["age"] <= 80)
    not_pregnant = merged["RIDEXPRG"] != 1
    has_bp = data["systolic_bp"].notna() | data["diastolic_bp"].notna()
    has_lab = data["hba1c"].notna() | data["glucose_fasting"].notna()
    has_meds = merged["BPQ050A"] == 1
    known_label = (has_bp & has_lab) | has_meds
    data = data[adult & not_pregnant & known_label].reset_index(drop=True)

    data = data.replace([np.inf, -np.inf], np.nan)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(out_csv, index=False)

    n = len(data)
    print(f"\nNHANES 2017-2018 -> {out_csv}")
    print(f"  n = {n} | positive (tăng HA/ĐTĐ) = {int(data['label'].sum())} ({data['label'].mean():.1%})")
    print(f"  missing: glucose_fasting {data['glucose_fasting'].isna().mean():.0%}, "
          f"hba1c {data['hba1c'].isna().mean():.0%}, creatinine {data['creatinine'].isna().mean():.0%}")
    print("  Thống kê chỉ số:")
    for col in ["systolic_bp", "diastolic_bp", "heart_rate", "glucose_fasting",
                "hba1c", "creatinine", "bmi"]:
        s = data[col].dropna()
        print(f"    {col:<16} mean={s.mean():.2f}  median={s.median():.2f}  min={s.min():.2f}  max={s.max():.2f}")
    return data


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/datasets/nhanes_2017_2018.csv")
    cache = Path("data/datasets/_nhanes_cache")
    build(out, cache)
