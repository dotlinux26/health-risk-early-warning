"""P0.3 — Độ ổn định của đường cơ sở cá nhân theo độ dài cửa sổ N.

docs/15 mục 5.3/5.5 + nhận xét GV L2 ("cá nhân hóa nhưng chưa chứng minh
baseline ổn định"). Trả lời bằng thực nghiệm thay vì lời nói:

    N ∈ {3, 5, 7, 14, 30, 90} ngày
    → μ, σ, z-score thay đổi bao nhiêu so với tham chiếu N=90?
    → bao nhiêu % ngày bị đổi "vùng nguy cơ" (band z-score)?

Dữ liệu: chuỗi thời gian tổng hợp có đặc tính giống đo sức khỏe cá nhân
(baseline riêng từng người, noise tự tương quan, drift chậm, spike ngẫu nhiên).
Seed cố định → tái lập được.

Chạy:
    python scripts/run_baseline_stability.py

Output:
    experiments/BASELINE-STABILITY/{summary.json, summary.md}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT_DIR = Path("experiments/BASELINE-STABILITY")
WINDOWS = [3, 5, 7, 14, 30, 90]
REFERENCE_N = 90
METRICS = ["systolic_bp", "diastolic_bp", "heart_rate", "glucose_fasting"]

# Tham số mô phỏng (tham chiếu phạm vi lâm sàng thông thường)
SPEC = {
    "systolic_bp":     {"base": 128.0, "sigma": 9.0},
    "diastolic_bp":    {"base": 80.0,  "sigma": 6.0},
    "heart_rate":      {"base": 74.0,  "sigma": 7.0},
    "glucose_fasting": {"base": 5.8,   "sigma": 0.55},
}


def simulate_patient(rng: np.random.Generator, days: int) -> pd.DataFrame:
    """Một bệnh nhân: baseline riêng + AR(1) noise + drift chậm + spike."""
    ts = pd.date_range("2025-01-01", periods=days, freq="D")
    out = {"timestamp": ts}
    n = len(ts)
    for col, s in SPEC.items():
        noise = np.zeros(n)
        e = 0.0
        for i in range(n):
            e = 0.55 * e + rng.normal(0, s["sigma"] * 0.75)
            noise[i] = e
        drift = np.linspace(0, rng.normal(0, s["sigma"] * 0.35), n)
        spikes = (
            (rng.random(n) < 0.02).astype(float)
            * rng.normal(s["sigma"] * 2.4, s["sigma"], n)
        )
        out[col] = s["base"] + rng.normal(0, s["sigma"] * 0.25) + noise + drift + spikes
        if col == "glucose_fasting":
            # mô phỏng thiếu dữ liệu: chỉ đo ~70% ngày
            mask = rng.random(n) < 0.30
            out[col][mask] = np.nan
    return pd.DataFrame(out)


def band(z: np.ndarray) -> np.ndarray:
    """Vùng z-score: 0 = thường (<1σ), 1 = hơi cao (1–2σ), 2 = cao (≥2σ)."""
    z = np.asarray(z, dtype=float)
    b = np.zeros(len(z), dtype=int)
    b[z >= 1.0] = 1
    b[z >= 2.0] = 2
    return b


def main() -> None:
    rng = np.random.default_rng(42)
    patients = [simulate_patient(rng, 150) for _ in range(150)]
    print(f"Mô phỏng {len(patients)} bệnh nhân x 150 ngày, "
          f"{len(METRICS)} chỉ số (glucose thiếu ~30% ngày)")

    # Tham chiếu N=90 cho toàn bộ patient/metric
    ref: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    for pi, pdf in enumerate(patients):
        b90 = pdf.set_index("timestamp").rolling(f"{REFERENCE_N}D", min_periods=5)
        ref[pi] = {
            col: {
                "mean": b90[col].mean().to_numpy(),
                "std": b90[col].std().to_numpy(),
            }
            for col in METRICS
        }

    rows = []
    for N in WINDOWS:
        dev_mu, dev_sd, band_flips, risk_flips, n_eval = [], [], [], [], []
        for pi, pdf in enumerate(patients):
            idx = pdf.set_index("timestamp")
            roll = idx.rolling(f"{N}D", min_periods=max(3, N // 2))
            start = max(45, REFERENCE_N)  # bỏ giai đoạn khởi động
            for col in METRICS:
                mu = roll[col].mean().to_numpy()
                sd = roll[col].std().to_numpy()
                r_mu, r_sd = ref[pi][col]["mean"], ref[pi][col]["std"]
                val = idx[col].to_numpy()
                z_n = (val - mu) / np.where(sd > 1e-9, sd, np.nan)
                z_r = (val - r_mu) / np.where(r_sd > 1e-9, r_sd, np.nan)
                ok = (
                    (np.arange(len(val)) >= start)
                    & ~np.isnan(val) & ~np.isnan(mu) & ~np.isnan(r_mu)
                    & ~np.isnan(sd) & ~np.isnan(r_sd) & (r_sd > 1e-9) & (sd > 1e-9)
                )
                if not ok.any():
                    continue
                dev_mu.extend(np.abs(mu[ok] - r_mu[ok]) / r_sd[ok])
                dev_sd.extend(np.abs(sd[ok] - r_sd[ok]) / r_sd[ok])
                bn, br = band(z_n[ok]), band(z_r[ok])
                band_flips.append(float((bn != br).mean()))
                risk_flips.append(float(((bn >= 2) != (br >= 2)).mean()))
                n_eval.append(int(ok.sum()))
        rows.append({
            "window_days": N,
            "delta_mean_in_sigma": round(float(np.mean(dev_mu)), 4),
            "p95_mean_in_sigma": round(float(np.percentile(dev_mu, 95)), 4),
            "delta_std_relative": round(float(np.mean(dev_sd)), 4),
            "band_flip_rate": round(float(np.mean(band_flips)), 4),
            "risk_flag_flip_rate": round(float(np.mean(risk_flips)), 4),
            "n_obs": int(sum(n_eval)),
        })
        print(f"  N={N:>3}: |Δμ|={rows[-1]['delta_mean_in_sigma']:.3f}σ "
              f"| Δσ={rows[-1]['delta_std_relative']:.3f} "
              f"| đổi band={rows[-1]['band_flip_rate']:.1%} "
              f"| đổi cờ z>=2σ={rows[-1]['risk_flag_flip_rate']:.1%}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(
        json.dumps(
            {
                "design": "150 patient tong hop x 150 ngay, seed=42, tham chieu N=90",
                "metrics_simulated": list(SPEC.keys()),
                "windows": WINDOWS,
                "results": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    md = [
        "# P0.3 Ổn định đường cơ sở cá nhân theo độ dài cửa sổ N",
        "",
        "- Thiết kế: 150 bệnh nhân tổng hợp × 150 ngày × 4 chỉ số; seed=42.",
        "- Tham chiếu: cửa sổ 90 ngày (giống production). So sánh μ, σ, z-band.",
        "",
        "| N (ngày) | \\|Δμ\\| (đơn vị σ₉₀) | P95 \\|Δμ\\| | \\|Δσ\\| tương đối | Đổi vùng z | Đổi cờ z≥2σ | Số quan sát |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md.append(
            f"| {r['window_days']} | {r['delta_mean_in_sigma']:.3f} "
            f"| {r['p95_mean_in_sigma']:.3f} | {r['delta_std_relative']:.3f} "
            f"| {r['band_flip_rate']:.1%} | {r['risk_flag_flip_rate']:.1%} "
            f"| {r['n_obs']} |"
        )
    md += [
        "",
        "## Cách đọc",
        "",
        "- `|Δμ|` nhỏ (≪1σ) nghĩa là mean baseline hội tụ sớm; nếu tăng nhanh khi",
        "  N giảm, baseline ngắn dễ bị nhiễu kéo lệch.",
        "- `Đổi cờ z≥2σ` = tỉ lệ ngày mà kết luận \"bất thường rõ\" đảo trạng thái",
        "  so với tham chiếu 90 ngày — đây là con số ảnh hưởng trực tiếp đến UX:",
        "  quyết định hiển thị trạng thái baseline (CHƯA ỔN ĐỊNH / ĐỦ ĐIỀU KIỆN)",
        "  trong UI theo bảng này, không phải cảm tính.",
        "",
        "> Giới hạn: dữ liệu tổng hợp phục vụ định lượng hành vi thống kê của",
        "> cửa sổ trượt; khi có dữ liệu dọc thật sẽ lặp lại đúng protocol này.",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nKết quả lưu tại: {OUT_DIR}/")


if __name__ == "__main__":
    main()
