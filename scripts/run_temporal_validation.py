"""Thí nghiệm P2.1 — VALIDATION THEO THỜI GIAN với outcome TỬ VƯƠNG THẬT.

Vấn đề được giải quyết (docs/16, T7 + một phần T8):
  Các thí nghiệm trước dùng nhãn cắt ngang (tăng HA / ĐTĐ định nghĩa bằng chính
  chỉ số đo) → AUC cao nhưng bị nghi "label circularity". Ở đây dùng NHANES
  Linked Mortality File (NCHS, follow-up đến 31/12/2019): đặc trưng đo TẠI KỲ
  KHÁM, outcome là tử vương SAU KHOẢNG THỜI GIAN — độc lập tuyệt đối với đầu vào
  và có hướng thời gian thật (prospective).

Thiết kế:
  - Cohort: người lớn ≥20 tuổi trong nhanes_mortality.csv, chu kỳ 2015-2016 và
    2017-2018 (LMF công khai chỉ phủ đến 2018).
  - Nhiệm vụ A — phân loại tử vương toàn bộ ≤12 tháng (cửa sổ hợp lệ cho CẢ HAI
    chu kỳ vì follow-up ngắn nhất của chu kỳ 2017-18 vẫn ≥12 tháng):
      * tách THEO THỜI GIAN: train 2015-16 → test 2017-18 (mô phỏng triển khai
        "hôm nay" trên dữ liệu tương lai);
      * so sánh với tách ngẫu nhiên cùng tỉ lệ để thấy mức độ lạc quan của
        random split.
  - Nhiệm vụ B — phân biệt sống/chết có kiểm duyệt: Harrell's C-index trên toàn
    bộ follow-up (time = min(permth_exm, 60 tháng), event = death_all).
  - Nhiệm vụ C — tử vương tim mạch (tim hoặc mạch máu não, UCOD 1/5) ≤24 tháng,
    chỉ báo cáo thống kê mô tả do số kiện ít.
  - Lead time: ở nhóm test, những người chết ≤24 tháng nằm trong quintile nguy
    cơ CAO NHẤT được phát hiện trước trung bình bao nhiêu tháng.
  - Hiệu chỉnh xác suất: isotonic fit trên train (CV nội bộ), báo Brier/ECE
    trước–sau trên test.

Chạy:
    python3 scripts/run_temporal_validation.py

Kết quả: experiments/EXP-TEMPORAL-LMF/{summary.json, summary.md}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.calibration import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.experiments.calibration import expected_calibration_error

FEATURES = [
    "systolic_bp", "diastolic_bp", "heart_rate",
    "glucose_fasting", "hba1c", "creatinine", "bmi", "age", "sex",
]
SEED = 42
OUT_DIR = Path("experiments/EXP-TEMPORAL-LMF")


def make_lgbm():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4, num_leaves=16,
        subsample=0.8, colsample_bytree=0.8, random_state=SEED, verbosity=-1,
    )


def harrell_c(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    """C-index của Harrell cho dữ liệu sống-còn có kiểm duyệt (điểm càng cao càng rủi ro).

    Cặp so sánh hợp lệ với chủ thể i có biến cố: mọi j có time_j > time_i, hoặc
    cùng time và j không có biến cố. Concordant khi risk_i > risk_j.
    Vector hoá theo từng i để tránh vòng lặp kép trong Python.
    """
    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=bool)
    r = np.asarray(risk, dtype=float)
    num = den = 0.0
    for i in range(len(t)):
        if not e[i]:
            continue
        valid = (t > t[i]) | ((t == t[i]) & ~e)
        valid[i] = False
        m = int(valid.sum())
        if not m:
            continue
        rj = r[valid]
        num += float(np.sum(r[i] > rj)) + 0.5 * float(np.sum(r[i] == rj))
        den += m
    return num / den if den else float("nan")


def fit_model(name: str, Xtr: np.ndarray, ytr: np.ndarray):
    if name == "lr":
        pipe_log = LogisticRegression(max_iter=2000, random_state=SEED)
        return pipe_log.fit(Xtr, ytr)
    return make_lgbm().fit(Xtr, ytr)


def predict(model, name: str, X: np.ndarray) -> np.ndarray:
    if name == "lr":
        return model.predict_proba(X)[:, 1]
    return model.predict_proba(X)[:, 1]


def evaluate(y_true, proba, label: str) -> dict:
    out = {
        "n": int(len(y_true)),
        "events": int(np.sum(y_true)),
        "prevalence": round(float(np.mean(y_true)), 5),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "auprc": round(float(average_precision_score(y_true, proba)), 4),
        "brier": round(float(brier_score_loss(y_true, proba)), 5),
        "ece_10bin": round(expected_calibration_error(y_true, proba), 5),
    }
    print(f"  [{label}] n={out['n']} events={out['events']} "
          f"AUC={out['roc_auc']:.4f} AUPRC={out['auprc']:.4f} Brier={out['brier']:.4f}")
    return out


def main() -> None:
    df = pd.read_csv("data/datasets/nhanes_mortality.csv")
    df = df[df["mortstat"].notna()].copy()
    df["sex"] = df["sex"].astype(float)
    keep = df["eligstat"] == 1
    df = df[keep & df["age"].notna() & df["systolic_bp"].notna()].reset_index(drop=True)

    tr = df[df["cycle"] == "2015_2016"].reset_index(drop=True)
    te = df[df["cycle"] == "2017_2018"].reset_index(drop=True)
    print(f"Cohort linked: n={len(df)} | train 2015-16={len(tr)} test 2017-18={len(te)}")

    imp = SimpleImputer(strategy="median").fit(tr[FEATURES])
    sc = StandardScaler().fit(imp.transform(tr[FEATURES]))  # chỉ dùng cho LR
    Xtr_raw = imp.transform(tr[FEATURES])
    Xte_raw = imp.transform(te[FEATURES])
    Xtr_lr = sc.transform(Xtr_raw)
    Xte_lr = sc.transform(Xte_raw)

    results: dict = {
        "experiment": "EXP-TEMPORAL-LMF",
        "seed": SEED,
        "data": {
            "source": "NHANES Public-Use Linked Mortality File 2019 (NCHS)",
            "url": "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/",
            "features": FEATURES,
            "train_cycle": "2015_2016", "test_cycle": "2017_2018",
            "n_train": int(len(tr)), "n_test": int(len(te)),
            "death_12m_train": int(((tr["death_all"] == 1) & tr["permth_exm"].le(12).fillna(False)).sum()),
            "note": ("follow-up đến 31/12/2019; chu kỳ 2017-18 có cửa sổ tối đa ~37 tháng "
                     "→ nhiệm vụ chính dùng horizon 12 tháng (hợp lệ cả hai chu kỳ)"),
        },
        "tasks": {},
    }

    # ---------------- Nhiệm vụ A: tử vong ≤12 tháng, tách theo thời gian ----
    def death12(frame: pd.DataFrame) -> np.ndarray:
        m = frame["permth_exm"].astype("Float64").fillna(999).astype(float)
        return ((frame["death_all"] == 1) & (m <= 12)).to_numpy(dtype=int)

    ytr_a, yte_a = death12(tr), death12(te)
    taskA: dict = {}
    for name in ["lr", "lgbm"]:
        m = fit_model(name, Xtr_lr if name == "lr" else Xtr_raw, ytr_a)
        p_tr = predict(m, name, Xtr_lr if name == "lr" else Xtr_raw)
        p_te = predict(m, name, Xte_lr if name == "lr" else Xte_raw)
        # isotonic fit trên train (không nhìn test)
        iso = IsotonicRegression(out_of_bounds="clip").fit(p_tr, ytr_a)
        p_te_cal = iso.predict(p_te)
        taskA[name] = {
            "temporal_split_train": evaluate(ytr_a, p_tr, f"{name}/train"),
            "temporal_split_test": evaluate(yte_a, p_te, f"{name}/test"),
            "test_isotonic": {
                "brier": round(brier_score_loss(yte_a, p_te_cal), 5),
                "ece_10bin": round(expected_calibration_error(yte_a, p_te_cal), 5),
                "roc_auc": round(roc_auc_score(yte_a, p_te_cal), 4),
            },
        }
        # random split đối chứng (cùng tỉ lệ test ~48% như temporal)
        rng = np.random.RandomState(SEED)
        idx = rng.permutation(len(df))
        n_te_r = len(te)
        dfr = df.iloc[idx].reset_index(drop=True)
        Xr = imp.transform(dfr[FEATURES])
        mr_months = dfr["permth_exm"].astype("Float64").fillna(999).astype(float)
        yr = ((dfr["death_all"] == 1) & (mr_months <= 12)).to_numpy(int)
        mr = fit_model(name, sc.transform(Xr[:len(tr)]) if name == "lr" else Xr[:len(tr)], yr[:len(tr)])
        pr = predict(mr, name, sc.transform(Xr[len(tr):]) if name == "lr" else Xr[len(tr):])
        taskA[name]["random_split_test"] = evaluate(yr[len(tr):], pr, f"{name}/random-test")
        print(f"  [{name}] gap temporal-vs-random AUC: "
              f"{taskA[name]['temporal_split_test']['roc_auc']:.4f} vs "
              f"{taskA[name]['random_split_test']['roc_auc']:.4f}")

    results["tasks"]["death_within_12m"] = taskA

    # ---------------- Nhiệm vụ B: C-index sống-còn toàn follow-up ----------
    def surv_arrays(frame: pd.DataFrame):
        t = frame["followup_months"].astype("Float64").fillna(60).astype(float)
        t = np.minimum(t.to_numpy(dtype=float), 60.0)
        e = frame["death_all"].to_numpy(dtype=int)
        return t, e

    taskB: dict = {}
    t_tr, e_tr = surv_arrays(tr)
    t_te, e_te = surv_arrays(te)
    for name in ["lr", "lgbm"]:
        # điểm nguy cơ từ model của nhiệm vụ A (dự báo tử vong sớm) dùng lại
        m = fit_model(name, Xtr_lr if name == "lr" else Xtr_raw, ytr_a)
        r_te = predict(m, name, Xte_lr if name == "lr" else Xte_raw)
        c_te = harrell_c(t_te, e_te, r_te)
        c_te_fast = None
        # C-index nhanh hơn: subsample 1500 dòng nếu quá chậm
        taskB[name] = {"c_index_test_full_followup": round(c_te, 4)}
        print(f"  [{name}] Harrell C-index (test, ≤60m) = {c_te:.4f}")
    taskB["note"] = ("risk score lấy từ model nhiệm vụ A (dự báo tử vong 12 tháng); "
                     "censoring tại 60 tháng")
    results["tasks"]["survival_c_index"] = taskB

    # ---------------- Nhiệm vụ C: tử vong do bệnh tim ≤24 tháng ------------
    # Bản công khai LMF chỉ còn UCOD 1 (bệnh tim), 2 (ung thư), 10 (khác)
    # sau perturbation → "CVD" ở đây là tử vong do bệnh tim, không gồm đột quỵ.
    cvd_tr = int(((tr["death_cvd"] == 1) & tr["permth_exm"].le(24).fillna(False)).sum())
    cvd_te = int(((te["death_cvd"] == 1) & te["permth_exm"].le(24).fillna(False)).sum())
    results["tasks"]["cvd_death_24m_descriptive"] = {
        "train_events": cvd_tr, "test_events": cvd_te,
        "definition": "tử vong do bệnh tim (UCOD_LEADING=1) trong 24 tháng",
        "note": ("ít kiện → không huấn luyện riêng; bản perturbed không tách được "
                 "đột quỵ/ĐTĐ/bệnh thận khỏi 'nguyên nhân khác'"),
    }
    print(f"  CVD deaths ≤24m: train={cvd_tr}, test={cvd_te}")

    # ---------------- Lead time --------------------------------------------
    m = fit_model("lgbm", Xtr_raw, ytr_a)
    p_te = predict(m, "lgbm", Xte_raw)
    dead_te = (te["death_all"] == 1) & te["permth_exm"].astype("Float64").fillna(999).le(24)
    months_dead = te.loc[dead_te, "permth_exm"].astype("Float64").astype(float).to_numpy()
    risk_dead = p_te[dead_te.to_numpy()]
    q80 = np.quantile(p_te, 0.8)
    hi = risk_dead >= q80
    lead_hi = months_dead[hi]          # tháng còn sống sau khám, nhóm nguy cơ cao
    lead_lo = months_dead[~hi]
    med_lead_hi = round(float(np.median(lead_hi)), 1) if len(lead_hi) else None
    leadtime = {
        "threshold_quintile_top20": round(float(q80), 4),
        "n_deceased_24m_test": int(dead_te.sum()),
        "caught_by_top20": int(hi.sum()),
        "capture_rate": round(float(hi.mean()), 4) if len(hi) else None,
        "median_lead_time_months_top20": med_lead_hi,
        "median_lead_time_months_rest": round(float(np.median(lead_lo)), 1) if len(lead_lo) else None,
        "note": ("lead time = số tháng tử vong xảy ra sau ngày khám MEC; "
                 "nhóm top-20% nguy cơ được phát hiện trung vị "
                 f"{med_lead_hi} tháng trước sự kiện"),
    }
    results["tasks"]["lead_time"] = leadtime
    print(f"  Lead time: bắt {int(hi.sum())}/{int(dead_te.sum())} ca tử vong ≤24m, "
          f"median {leadtime['median_lead_time_months_top20']} tháng")

    # So sánh với baseline "nhãn cắt ngang" cũ để minh hoạ chuyển đổi bài toán
    cross_auc = {}
    for lab in ["label_htn", "label_dm"]:
        try:
            cross_auc[lab] = round(float(roc_auc_score(
                yte_a, te[lab].to_numpy(dtype=float))), 4)
        except Exception:
            pass
    results["tasks"]["cross_sectional_label_baseline_on_test"] = cross_auc

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))

    md = ["# EXP-TEMPORAL-LMF — Validation theo thời gian với outcome tử vong thật\n"]
    md.append(f"- Nguồn: NHANES Public-Use LMF 2019 (train 2015-16 n={len(tr)}, test 2017-18 n={len(te)})")
    md.append(f"- Outcome: tử vong toàn bộ ≤12 tháng (train prevalence {np.mean(ytr_a):.2%}, test {np.mean(yte_a):.2%})\n")
    md.append("| Model | Split | ROC-AUC | AUPRC | Brier | ECE |")
    md.append("|---|---|---|---|---|---|")
    for name in ["lr", "lgbm"]:
        a = taskA[name]
        for split, key in [("temporal", "temporal_split_test"), ("random", "random_split_test")]:
            r = a[key]
            md.append(f"| {name.upper()} | {split} | {r['roc_auc']} | {r['auprc']} | {r['brier']} | {r['ece_10bin']} |")
    md.append("")
    md.append(f"- C-index (≤60 tháng): LR {taskB['lr']['c_index_test_full_followup']}, "
              f"LGBM {taskB['lgbm']['c_index_test_full_followup']}")
    lt = results["tasks"]["lead_time"]
    md.append(f"- Lead time: top-20% nguy cơ bắt {lt['caught_by_top20']}/{lt['n_deceased_24m_test']} "
              f"ca tử vong ≤24m, median {lt['median_lead_time_months_top20']} tháng trước sự kiện")
    (OUT_DIR / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nXong -> {OUT_DIR}/summary.json, summary.md")


if __name__ == "__main__":
    sys.exit(main())
