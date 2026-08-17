"""Tải NHANES (CDC) và dựng dataset thật cho mô hình cảnh báo sớm.

Nguồn: Continuous NHANES (CDC/NCHS), dữ liệu lâm sàng khảo sát toàn dân Mỹ,
miễn phí. Script hỗ trợ NHIỀU CHU KỲ (nhiều năm) — ghép các chu kỳ thành một
dataset lớn hơn. Mỗi người tham gia được đo nhiều lần trong kỳ khám.

Các chu kỳ đã phát hành (tính tới 2025):
    2015-2016 (I)  — đầy đủ
    2017-2018 (J)  — đầy đủ
    2019-2020 (P)  — CDC GỘP vào "pre-pandemic" (2017–Mar 2020) CÙNG bệnh nhân
                     với chu kỳ J, KHÔNG phải chu kỳ riêng — không thêm để
                     tránh trùng lặp mẫu khi gộp nhiều chu kỳ.
    2021-2023 (L)  — Aug 2021–Aug 2023, chu kỳ mới tách biệt (phát hành 09/2024)
    2023-2024, 2025 — chưa phát hành công khai tại thời điểm viết script.
                     Kiểm tra https://wwwn.cdc.gov/nchs/nhanes/ để cập nhật.

Chạy (mặc định: mọi chu kỳ khả dụng):
    python scripts/build_nhanes_dataset.py
    python scripts/build_nhanes_dataset.py --cycles 2015_2016 2017_2018 2021_2023

Kết quả: data/datasets/nhanes_merged.csv với chỉ số TRÙNG schema hệ thống:
    systolic_bp, diastolic_bp, heart_rate, glucose_fasting, hba1c, creatinine, bmi
và nhãn bệnh thật (tăng huyết áp / đái tháo đường) xác định bằng ngưỡng lâm sàng
hoặc đang dùng thuốc. Cột `cycle` đánh dấu chu kỳ gốc của mỗi bệnh nhân.

Lưu ý khoa học: nhãn "tăng huyết áp" được định nghĩa một phần bằng chính huyết
áp đo được — mô hình học sẽ đạt AUC cao nhưng cần đọc kết quả với ngữ cảnh này
(xem docs/06). NHANES là khảo sát cắt ngang; để có chuỗi thời gian thực sự cần
ghép nhiều kỳ khám hoặc nguồn dữ liệu dọc (MIMIC, UK Biobank).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

# Cấu hình từng chu kỳ NHANES.
#  - `path`: thư mục con của CDC (https://wwwn.cdc.gov/Nchs/Data/Nhanes/<path>/DataFiles/...)
#  - `files` / `keep`: file XPT cần tải và cột cần lấy
#  - `aliases`: cột nguồn cho từng chỉ số chuẩn hoá (BP mới dùng BPXOSY/BPXODI,
#    chu kỳ cũ dùng BPXSY/BPXDI)
#  - `htn_med_col`: cột "đang dùng thuốc tăng huyết áp" (BPQ050A cho chu kỳ cũ,
#    BPQ030 cho chu kỳ 2021-2023 theo codebook mới)
CYCLES: dict[str, dict[str, object]] = {
    "2015_2016": {
        "path": "Public/2015",
        "files": ["DEMO_I", "BPX_I", "BPQ_I", "GLU_I", "GHB_I", "BIOPRO_I", "BMX_I"],
        "keep": {
            "DEMO_I": ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDEXPRG"],
            "BPX_I": ["SEQN", "BPXSY1", "BPXSY2", "BPXSY3", "BPXDI1", "BPXDI2", "BPXDI3", "BPXPLS"],
            "BPQ_I": ["SEQN", "BPQ020", "BPQ050A"],
            "GLU_I": ["SEQN", "LBXGLU"],
            "GHB_I": ["SEQN", "LBXGH"],
            "BIOPRO_I": ["SEQN", "LBXSCR"],
            "BMX_I": ["SEQN", "BMXBMI"],
        },
        "aliases": {
            "systolic_bp": ["BPXSY1", "BPXSY2", "BPXSY3"],
            "diastolic_bp": ["BPXDI1", "BPXDI2", "BPXDI3"],
            "heart_rate": ["BPXPLS"],
            "glucose_fasting": ["LBXGLU"],
            "hba1c": ["LBXGH"],
            "creatinine": ["LBXSCR"],
            "bmi": ["BMXBMI"],
        },
        "htn_med_col": "BPQ050A",
    },
    "2017_2018": {
        "path": "Public/2017",
        "files": ["DEMO_J", "BPX_J", "BPQ_J", "GLU_J", "GHB_J", "BIOPRO_J", "BMX_J"],
        "keep": {
            "DEMO_J": ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDEXPRG"],
            "BPX_J": ["SEQN", "BPXSY1", "BPXSY2", "BPXSY3", "BPXDI1", "BPXDI2", "BPXDI3", "BPXPLS"],
            "BPQ_J": ["SEQN", "BPQ020", "BPQ050A"],
            "GLU_J": ["SEQN", "LBXGLU"],
            "GHB_J": ["SEQN", "LBXGH"],
            "BIOPRO_J": ["SEQN", "LBXSCR"],
            "BMX_J": ["SEQN", "BMXBMI"],
        },
        "aliases": {
            "systolic_bp": ["BPXSY1", "BPXSY2", "BPXSY3"],
            "diastolic_bp": ["BPXDI1", "BPXDI2", "BPXDI3"],
            "heart_rate": ["BPXPLS"],
            "glucose_fasting": ["LBXGLU"],
            "hba1c": ["LBXGH"],
            "creatinine": ["LBXSCR"],
            "bmi": ["BMXBMI"],
        },
        "htn_med_col": "BPQ050A",
    },
    "2021_2023": {
        "path": "Public/2021",
        "files": ["DEMO_L", "BPXO_L", "BPQ_L", "GLU_L", "GHB_L", "BIOPRO_L", "BMX_L"],
        "keep": {
            "DEMO_L": ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDEXPRG"],
            "BPXO_L": ["SEQN", "BPXOSY1", "BPXOSY2", "BPXOSY3", "BPXODI1", "BPXODI2", "BPXODI3", "BPXOPLS1", "BPXOPLS2", "BPXOPLS3"],
            "BPQ_L": ["SEQN", "BPQ020", "BPQ030"],
            "GLU_L": ["SEQN", "LBXGLU"],
            "GHB_L": ["SEQN", "LBXGH"],
            "BIOPRO_L": ["SEQN", "LBXSCR"],
            "BMX_L": ["SEQN", "BMXBMI"],
        },
        "aliases": {
            "systolic_bp": ["BPXOSY1", "BPXOSY2", "BPXOSY3"],
            "diastolic_bp": ["BPXODI1", "BPXODI2", "BPXODI3"],
            "heart_rate": ["BPXOPLS1", "BPXOPLS2", "BPXOPLS3"],
            "glucose_fasting": ["LBXGLU"],
            "hba1c": ["LBXGH"],
            "creatinine": ["LBXSCR"],
            "bmi": ["BMXBMI"],
        },
        "htn_med_col": "BPQ030",
    },
}


def base_url(cycle: str) -> str:
    return f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/{CYCLES[cycle]['path']}/DataFiles/{{name}}.XPT"


def fetch(cycle: str, name: str, cache_dir: Path) -> Path:
    target = cache_dir / f"{name}.XPT"
    if not target.exists():
        import urllib.request

        url = base_url(cycle).format(name=name)
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


def build_cycle(cycle: str, cache_dir: Path) -> pd.DataFrame:
    cfg = CYCLES[cycle]
    aliases = cfg["aliases"]
    frames: list[pd.DataFrame] = []
    for name in cfg["files"]:
        df = pd.read_sas(fetch(cycle, name, cache_dir), format="xport")
        frames.append(df[cfg["keep"][name]])
    merged = frames[0]
    for df in frames[1:]:
        merged = merged.merge(df, on="SEQN", how="left")

    data = pd.DataFrame({"seqn": merged["SEQN"]})
    data["cycle"] = cycle
    data["age"] = merged["RIDAGEYR"]
    data["sex"] = merged["RIAGENDR"]
    data["systolic_bp"] = mean_valid(merged, aliases["systolic_bp"], 40, 300)
    data["diastolic_bp"] = mean_valid(merged, aliases["diastolic_bp"], 20, 160)
    hr_cols = aliases["heart_rate"]
    if len(hr_cols) == 1:
        data["heart_rate"] = merged[hr_cols[0]]  # nhịp 60 giây
    else:
        data["heart_rate"] = mean_valid(merged, hr_cols, 20, 220)
    data["glucose_fasting"] = merged["LBXGLU"] / 18.016  # mg/dL -> mmol/L
    data["hba1c"] = merged["LBXGH"]  # %
    data["creatinine"] = merged["LBXSCR"]  # mg/dL
    data["bmi"] = merged["BMXBMI"]

    # --- Nhãn lâm sàng thật ------------------------------------------------
    # Tăng huyết áp: HA đo cao HOẶC đang dùng thuốc huyết áp (BPQ050/BPQ030)
    htn = (data["systolic_bp"] >= 140) | (data["diastolic_bp"] >= 90)
    htn |= merged[cfg["htn_med_col"]] == 1
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
    has_meds = merged[cfg["htn_med_col"]] == 1
    known_label = (has_bp & has_lab) | has_meds
    data = data[adult & not_pregnant & known_label].reset_index(drop=True)

    data = data.replace([np.inf, -np.inf], np.nan)

    n = len(data)
    print(f"\nNHANES {cycle} -> {n} dòng")
    print(f"  positive (tăng HA/ĐTĐ) = {int(data['label'].sum())} ({data['label'].mean():.1%})")
    print(f"  missing: glucose_fasting {data['glucose_fasting'].isna().mean():.0%}, "
          f"hba1c {data['hba1c'].isna().mean():.0%}, creatinine {data['creatinine'].isna().mean():.0%}")
    for col in ["systolic_bp", "diastolic_bp", "heart_rate", "glucose_fasting",
                "hba1c", "creatinine", "bmi"]:
        s = data[col].dropna()
        print(f"    {col:<16} mean={s.mean():.2f}  median={s.median():.2f}  "
              f"min={s.min():.2f}  max={s.max():.2f}")
    return data


def merge_cycles(cycles: list[str], cache_dir: Path) -> pd.DataFrame:
    frames = [build_cycle(c, cache_dir) for c in cycles]
    merged = pd.concat(frames, ignore_index=True)
    return merged.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tải NHANES nhiều chu kỳ và gộp thành dataset lớn")
    parser.add_argument(
        "--cycles", type=str, nargs="*",
        default=list(CYCLES.keys()),
        help=f"Chu kỳ cần tải (mặc định: {list(CYCLES.keys())}). Dùng --cycles 2015_2016 2017_2018",
    )
    parser.add_argument("--out", type=str, default="data/datasets/nhanes_merged.csv")
    parser.add_argument("--cache", type=str, default="data/datasets/_nhanes_cache")
    args = parser.parse_args()

    cycles = args.cycles or list(CYCLES.keys())
    cache = Path(args.cache)
    out = Path(args.out)
    data = merge_cycles(cycles, cache)

    out.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(out, index=False)
    print(f"\nNHANES gộp {len(cycles)} chu kỳ -> {out}")
    print(f"  tổng n = {len(data)} | positive = {int(data['label'].sum())} "
          f"({data['label'].mean():.1%})")
    print("  phân bố theo cycle:")
    for cycle, n in data["cycle"].value_counts().items():
        print(f"    {cycle:<10} {n}")
    print("  Sẵn sàng cho huấn luyện: python scripts/train_nhanes.py --data "
          f"{out}")


if __name__ == "__main__":
    sys.exit(main())
