"""Tải NHANES Public-Use Linked Mortality File (NCHS) và ghép với dataset NHANES.

Nguồn (công khai, không cần credential):
    https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/
File fixed-width .dat theo chu kỳ 1999-2018, mortality follow-up đến 31/12/2019.

Biến chính (codebook NCHS 2019 public-use LMF):
    SEQN         ID người tham gia (khớp với nhanes_merged.csv)
    ELIGSTAT     1=đủ điều kiện linkage, 2=<18 tuổi, 3=không đủ
    MORTSTAT     0=còn sống (assumed), 1=đã mất (assumed) — KHÔNG bị perturb
    UCOD_LEADING Nguyên nhân tử vương chính:
                 1=bệnh tim, 2=ung thư, ..., 5=đột quỵ, 7=ĐTĐ, 9=bệnh thận,
                 10=nguyên nhân khác. LƯU Ý: file công khai đã bị perturbation
                 — kiểm chứng thực tế chu kỳ 2015-2018 chỉ còn giá trị 1, 2, 10
                 (đột quỵ/ĐTĐ/bệnh thận bị gộp vào "nguyên nhân khác").
    DIABETES/HYPERTEN  cờ MCOD (nguyên nhân đa重) — có thể thiếu
    PERMTH_INT/EXM     số tháng follow-up từ ngày phỏng vấn / ngày khám MEC

Chạy:
    python scripts/fetch_nhanes_mortality.py                # mọi chu kỳ 1999-2018
    python scripts/fetch_nhanes_mortality.py --cycles 2015_2016 2017_2018

Kết quả:
    data/datasets/nhanes_mortality_raw.csv   — LMF đã parse (seqn, cycle)
    data/datasets/nhanes_mortality.csv       — ghép với nhanes_merged.csv + nhãn
                                               death_5y / cvd_death_5y
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

BASE = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality"

# Bố cục fixed-width cho file NHANES (theo R_ReadInProgramAllSurveys.R của NCHS,
# vị trí cột đếm từ 1): seqn 1-6 | eligstat 15 | mortstat 16 | ucod 17-19 |
# diabetes 20 | hyperten 21 | permth_int 43-45 | permth_exm 46-48.
FWF_SPEC = {
    "seqn": (0, 6),
    "eligstat": (14, 15),
    "mortstat": (15, 16),
    "ucod_leading": (16, 19),
    "diabetes": (19, 20),
    "hyperten": (20, 21),
    "permth_int": (42, 45),
    "permth_exm": (45, 48),
}

ALL_CYCLES = [
    "1999_2000", "2001_2002", "2003_2004", "2005_2006", "2007_2008",
    "2009_2010", "2011_2012", "2013_2014", "2015_2016", "2017_2018",
]


def download(cycle: str, cache_dir: Path) -> Path:
    target = cache_dir / f"NHANES_{cycle}_MORT_2019_PUBLIC.dat"
    if target.exists():
        return target
    import urllib.request

    url = f"{BASE}/NHANES_{cycle}_MORT_2019_PUBLIC.dat"
    print(f"  tải {Path(url).name} ...", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=180) as resp, target.open("wb") as f:
        f.write(resp.read())
    return target


def parse_dat(path: Path, cycle: str) -> pd.DataFrame:
    df = pd.read_fwf(path, colspecs=list(FWF_SPEC.values()), names=list(FWF_SPEC),
                     na_values=[".", ""], dtype="Float64")
    df["cycle"] = cycle
    # Chuẩn hoá về int nullable (giữ NA cho ineligible)
    for c in ["seqn", "eligstat", "mortstat", "ucod_leading", "diabetes",
              "hyperten", "permth_int", "permth_exm"]:
        df[c] = df[c].astype("Int64")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Tải NHANES Linked Mortality File")
    parser.add_argument("--cycles", nargs="*", default=ALL_CYCLES)
    parser.add_argument("--merged", default="data/datasets/nhanes_merged.csv")
    parser.add_argument("--cache", default="data/datasets/_nhanes_cache")
    parser.add_argument("--out-raw", default="data/datasets/nhanes_mortality_raw.csv")
    parser.add_argument("--out", default="data/datasets/nhanes_mortality.csv")
    args = parser.parse_args()

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    frames = []
    for cycle in args.cycles:
        frames.append(parse_dat(download(cycle, cache), cycle))
    lmf = pd.concat(frames, ignore_index=True)
    Path(args.out_raw).parent.mkdir(parents=True, exist_ok=True)
    lmf.to_csv(args.out_raw, index=False)

    elig = lmf[lmf["eligstat"] == 1]
    dead = int((elig["mortstat"] == 1).sum())
    print(f"\nLMF {len(args.cycles)} chu kỳ: {len(lmf)} dòng | eligible={len(elig)} "
          f"| deceased={dead} ({dead / max(len(elig), 1):.1%} trong eligible)")

    merged_path = Path(args.merged)
    if not merged_path.exists():
        print(f"Không tìm thấy {merged_path} — chỉ xuất LMF thô.")
        return

    feat = pd.read_csv(merged_path)
    out = feat.merge(
        lmf[["seqn", "cycle", "eligstat", "mortstat", "ucod_leading",
             "diabetes", "hyperten", "permth_exm"]],
        on="seqn", how="left", suffixes=("", "_lmf"),
    )
    hit = out["cycle_lmf"].notna()
    assert (out.loc[hit, "cycle"] == out.loc[hit, "cycle_lmf"]).all(), \
        "cycle lệch giữa feature và LMF"
    out = out.drop(columns=["cycle_lmf"])

    linked = out["mortstat"].notna().sum()
    dead = out["mortstat"].eq(1).fillna(False)
    cvd_cause = out["ucod_leading"].eq(1).fillna(False)  # bệnh tim (bản công khai không tách đột quỵ)
    out["death_all"] = dead.astype(int)
    out["death_cvd"] = (dead & cvd_cause.fillna(False)).astype(int)
    months = out["permth_exm"].astype("Float64")
    out["death_5y"] = (dead & months.le(60).fillna(False)).astype(int)
    out["cvd_death_5y"] = ((out["death_cvd"] == 1) & months.le(60).fillna(False)).astype(int)
    out["followup_months"] = months

    out.to_csv(args.out, index=False)
    cov = out.groupby("cycle")["mortstat"].apply(lambda s: s.notna().mean())
    print(f"\nGhép với {merged_path}: n={len(out)} | khớp LMF={linked} ({linked / len(out):.1%})")
    for cyc, r in cov.items():
        d5 = out.loc[out['cycle'] == cyc, 'death_5y'].mean()
        print(f"  {cyc:<10} coverage={r:.1%}  death_5y={d5:.2%}")
    print(f"\nXong -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
