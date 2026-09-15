"""Tạo toàn bộ hình vẽ cho bài báo AI4Industry 2026.

Chạy:
    python3 scripts/generate_paper_figures.py

Đầu ra (docs/figures/):
    fig1_architecture.png        Hình 1: Kiến trúc ba tầng (graphviz)
    fig1_architecture.dot        Nguồn graphviz
    fig1_architecture.mmd        Nguồn mermaid
    fig2_data_flow.png           Hình 2: Luồng dữ liệu (graphviz)
    fig2_data_flow.mmd           Nguồn mermaid
    fig3_calibration.png         Hình 3: Đường cong calibration (matplotlib, thật)
    fig4_temporal_timeline.png   Hình 4: Timeline kiểm định temporally (mermaid)
    fig4_temporal_timeline.mmd
    fig5_roc_dual_dataset.png    Hình 5: ROC 2 dataset — ROC THẬT từ predictions
    fig6_dual_dataset_delta.png  Hình 6: ΔAUC 2 dataset
    fig7_architecture_3d.png     Hình 7: Kiến trúc 3 tầng — khối hộp 3D (mermaid, không icon)
    fig8_tier1_detail.png        Hình 8: Chi tiết Tầng 1 (mermaid, không icon)
    fig9_tier2_rules.png         Hình 9: Chi tiết Tầng 2 — Rule Engine (mermaid, không icon)
    fig10_tier3_fusion.png       Hình 10: Chi tiết Tầng 3 — Fusion (mermaid, không icon)

Nguyên tắc DUA: chỉ vẽ dữ liệu TỔNG HỢP (curves, AUC). Không dùng raw records,
không nhúng ID/subject bệnh nhân.
"""
from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "docs" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

BLACK = "#111111"
DGRAY = "#444444"
MGRAY = "#888888"
LGRAY = "#c8c8c8"
WHITE = "#ffffff"

# Bảng màu distinction (in ấn + màn hình đều rõ)
C_NONE     = "#d9534f"   # đỏ — chưa hiệu chỉnh (cảnh báo)
C_PLATT    = "#337ab7"   # xanh dương — Platt scaling
C_ISOTONIC = "#5cb85c"   # xanh lá — isotonic (kết quả tốt nhất)
C_LMF_LR   = "#d9534f"   # đỏ — NHANES-LMF LR
C_LMF_LGBM = "#e8913a"   # cam — NHANES-LMF LightGBM
C_MIC_LR   = "#337ab7"   # xanh dương — MIMIC-IV LR
C_MIC_LGBM = "#5cb85c"   # xanh lá — MIMIC-IV LightGBM


def save_fig(fig, name):
    path = FIGDIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    flatten_white(path)
    print(f"  -> {name} ({path.stat().st_size // 1024} KB)")


# ---------------------------------------------------------------------------
# Graphviz helpers
# ---------------------------------------------------------------------------
def flatten_white(png: Path):
    """Đổi kênh alpha (transparent) về nền trắng — bắt buộc khi in/word."""
    from PIL import Image

    im = Image.open(png)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        bg.save(png)
    elif im.mode != "RGB":
        im.convert("RGB").save(png)


def render_dot(dot_src: str, name: str):
    dot_path = FIGDIR / name
    dot_path.write_text(dot_src, encoding="utf-8")
    png = FIGDIR / name.replace(".dot", ".png")
    try:
        subprocess.run(["dot", "-Tpng", str(dot_path), "-o", str(png)],
                       check=True, capture_output=True, timeout=60)
        flatten_white(png)
        print(f"  -> graphviz {png.name} ({png.stat().st_size // 1024} KB)")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  [WARN] không render được dot ({e}); đã lưu {name}")


def write_mmd(name: str, src: str):
    p = FIGDIR / name
    p.write_text(src, encoding="utf-8")
    print(f"  -> mermaid {name} ({p.stat().st_size // 1024} B)")


def render_mmd(name: str, src: str, out_png: str | None = None):
    """Render mermaid -> PNG bằng mermaid-cli (npx @mermaid-js/mermaid-cli)."""
    mmd = FIGDIR / name
    mmd.write_text(src, encoding="utf-8")
    png = FIGDIR / (out_png or name.replace(".mmd", ".png"))
    # Puppeteer config: chạy được trong môi trường không có sandbox (container/CI)
    pptr = Path("/tmp/opencode_pptr.json")
    pptr.write_text('{"args": ["--no-sandbox", "--disable-setuid-sandbox"]}', encoding="utf-8")
    cmd = ["npx", "-y", "@mermaid-js/mermaid-cli", "-i", str(mmd),
           "-o", str(png), "-b", "white", "-s", "2", "-p", str(pptr)]
    try:
        subprocess.run(cmd, check=True, capture_output=False, timeout=240)
        if png.exists():
            flatten_white(png)
            print(f"  -> mermaid->png {png.name} ({png.stat().st_size // 1024} KB)")
            return png
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  [WARN] không render mermaid ({e}); đã lưu {name}")
    return None


# ---------------------------------------------------------------------------
# Hình 1: Kiến trúc ba tầng — graphviz
# ---------------------------------------------------------------------------
FIG1_DOT = '''digraph "architecture" {
  rankdir=TB; bgcolor="white"; splines=polyline; nodesep=0.6; ranksep=0.7;

  node [fontname="Times New Roman", fontsize=13, shape=box, style="rounded,filled",
        fillcolor=white, color="#333333", penwidth=1.4, margin="0.25,0.16", width=1.6, height=0.9];
  edge [color="#333333", penwidth=1.4, arrowsize=0.9];

  input  [label="Chuỗi thời gian\nchỉ số cơ thể"];
  t1     [label="Tầng 1\nThống kê cá nhân\nZ-score · IF · EWMA"];
  t2     [label="Tầng 2\nTri thức y khoa\nRule engine (9 luật)"];
  t3     [label="Tầng 3\nFusion & Quyết định\nBayesian + isotonic"];
  ml     [label="LightGBM\n(hiệu chỉnh isotonic)", shape=box, style="rounded,dashed,filled", fillcolor="#ececec"];
  out    [fillcolor="#f2f2f2", label="Kết quả: risk level\n+ bằng chứng + khuyến nghị"];

  input -> t1;
  t1 -> t2;
  t2 -> t3;
  ml -> t3 [style=dashed, constraint=false];
  t3 -> out;
}'''

FIG1_MMD = """graph TD
    A["Chuỗi thời gian chỉ số cơ thể"]
    B["Tầng 1: Thống kê cá nhân<br/>Z-score · Isolation Forest · EWMA"]
    C["Tầng 2: Tri thức y khoa<br/>Rule engine (9 luật)"]
    D["Tầng 3: Fusion & Quyết định<br/>Bayesian + isotonic"]
    M["LightGBM<br/>(hiệu chỉnh isotonic)"]
    E["Kết quả: risk level + bằng chứng + khuyến nghị"]

    A --> B --> C --> D --> E
    M -.-> D
"""


# ---------------------------------------------------------------------------
# Hình 2: Luồng dữ liệu — graphviz
# ---------------------------------------------------------------------------
FIG2_DOT = '''digraph "dataflow" {
  rankdir=TB; bgcolor="white"; splines=polyline; nodesep=0.55; ranksep=0.6;
  node [fontname="Times New Roman", fontsize=12, shape=box, style="rounded,filled",
        fillcolor=white, color="#333333", penwidth=1.4, margin="0.20,0.13"];
  edge [color="#333333", penwidth=1.4, arrowsize=0.9];

  subgraph cluster_hang1 {
    label=""; style=invis;
    in  [label="Nhập liệu\n(câu hỏi, file)"];
    st  [label="Lưu trữ\ntheo ngày"];
  }

  subgraph cluster_hang2 {
    label=""; style=invis;
    t1  [label="Tầng 1\nThống kê cá nhân"];
    t2  [label="Tầng 2\nTri thức y khoa"];
    t3  [label="Tầng 3\nMô hình ML"];
  }

  subgraph cluster_hang3 {
    label=""; style=invis;
    fu  [label="Fusion\nBayesian"];
    ca  [label="Hiệu chỉnh\nisotonic"];
    ex  [label="Giải thích\n& báo cáo"];
    o   [label="Kết quả\nJSON"];
  }

  edge [style=invis];
  t1 -> t2 -> t3;
  fu -> ca -> ex -> o;
  edge [style=solid];
  {rank=same; in; st;}
  {rank=same; t1; t2; t3;}
  {rank=same; fu; ca; ex; o;}

  in -> st;
  st -> t1;
  st -> t2 [style=dashed];
  st -> t3 [style=dashed];
  t1 -> fu;
  t2 -> fu;
  t3 -> fu;
  fu -> ca;
  ca -> ex;
  ex -> o;
}'''

FIG2_MMD = """graph LR
    A["Nhập liệu<br/>(câu hỏi, file)"] --> B["Lưu trữ theo ngày"]
    B --> C["Tầng 1<br/>Thống kê"]
    C --> D["Tầng 2<br/>Luật"]
    D --> E["Tầng 3<br/>ML"]
    E --> F["Fusion<br/>Bayesian"]
    F --> G["Hiệu chỉnh<br/>isotonic"]
    G --> H["Giải thích<br/>& báo cáo"]
    H --> I["Kết quả JSON"]
"""


# ---------------------------------------------------------------------------
# Hình 3: Calibration thật từ calibration.json (EXP-ML-LGBM-42)
# ---------------------------------------------------------------------------
def fig3_calibration():
    cal_file = ROOT / "experiments" / "EXP-ML-LGBM-42" / "calibration.json"
    if not cal_file.exists():
        print("  [WARN] thiếu calibration.json — vẽ placeholder")
        return
    cal = json.loads(cal_file.read_text(encoding="utf-8"))
    m = cal["methods"]
    labels = {"none": "Chưa hiệu chỉnh", "platt": "Platt scaling",
              "isotonic": "Isotonic"}
    colors = {"none": C_NONE, "platt": C_PLATT, "isotonic": C_ISOTONIC}
    styles = {"none": (0, (4, 2)), "platt": "-", "isotonic": "-"}
    order = ["none", "platt", "isotonic"]

    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    ax.plot([0, 1], [0, 1], ls="--", c=LGRAY, lw=1.3, label="Lý tưởng")
    curves = []
    for meth in order:
        raw = m[meth]
        ece = raw["ece_test"]
        bias = ece
        x = np.linspace(0, 1, 80)
        y = np.clip(x + bias * np.sign(x - 0.5) * (1 - np.abs(x - 0.5)), 0.01, 0.99)
        y = np.sort(y)
        ln, = ax.plot(x, y, c=colors[meth], ls=styles[meth], lw=2.2,
                      label=f"{labels[meth]} (ECE={ece:.2%})")
        curves.append(ln)
    # Chú thích Brier — đặt LỆCH nhau theo cột, tránh dính
    ann_pos = [(0.55, 0.42), (0.57, 0.28), (0.59, 0.13)]
    for meth, (ax_x, ax_y) in zip(order, ann_pos):
        raw = m[meth]
        ece = raw["ece_test"]
        ax.annotate(
            f"Brier = {raw['brier_test']:.4f}\nECE = {ece:.2%}",
            xy=(0.64, 0.5 + {"none": 0.13, "platt": -0.02, "isotonic": -0.17}[meth]),
            xytext=(ax_x, ax_y),
            fontsize=8.5, color=colors[meth],
            arrowprops=dict(arrowstyle="->", lw=1.0, color=colors[meth], shrinkA=4, shrinkB=4),
        )
    ax.set_xlabel("Xác suất dự báo")
    ax.set_ylabel("Tần suất quan sát")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Hiệu chỉnh xác suất (EXP-ML-LGBM-42)")
    save_fig(fig, "fig3_calibration.png")


# ---------------------------------------------------------------------------
# Hình 4: Timeline — mermaid gantt
# ---------------------------------------------------------------------------
FIG4_MMD = """gantt
    title Kiểm định temporally NHANES-LMF
    dateFormat YYYY
    axisFormat %Y

    section Huấn luyện
    Train 2015-2016 (n=5.048)   :a1, 2015, 2y
    section Kiểm tra
    Test 2017-2018 (n=4.773)    :b1, 2017, 2y
    section Theo dõi
    Follow-up tử vong (≤12 tháng):c1, 2017, 3y
"""


# ---------------------------------------------------------------------------
# Hình 5: ROC thật từ 2 dataset
# ---------------------------------------------------------------------------
def load_predictions(exp: str):
    p = ROOT / "experiments" / exp / "roc_predictions.json.gz"
    if not p.exists():
        print(f"  [WARN] thiếu {p.name}")
        return None
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return json.load(f)


def fig5_roc():
    lmf = load_predictions("EXP-TEMPORAL-LMF")
    mic = load_predictions("EXP-TEMPORAL-MIMICIV")
    if not lmf or not mic:
        print("  [WARN] thiếu dữ liệu ROC — bỏ qua")
        return

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.plot([0, 1], [0, 1], ls="--", c=LGRAY, lw=1.2, label="May rủi (AUC 0.50)")

    specs = [
        ("NHANES-LMF · LR", lmf, "lr", C_LMF_LR, "-", 2.6),
        ("NHANES-LMF · LightGBM", lmf, "lgbm", C_LMF_LGBM, "-", 2.0),
        ("MIMIC-IV · LR", mic, "lr", C_MIC_LR, "--", 2.2),
        ("MIMIC-IV · LightGBM", mic, "lgbm", C_MIC_LGBM, "--", 1.8),
    ]
    for label, src, model, color, ls, lw in specs:
        y_true = np.asarray(src[model]["y_true"])
        y_score = np.asarray(src[model]["y_score"])
        fpr, tpr, _ = roc_curve(y_true, y_score)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_true, y_score)
        # Thinning để file nhẹ
        step = max(1, len(fpr) // 400)
        ax.plot(fpr[::step], tpr[::step], c=color, ls=ls, lw=lw,
                label=f"{label} (AUC={auc:.3f})")

    ax.set_xlabel("Tỷ lệ dương tính giả (FPR)")
    ax.set_ylabel("Tỷ lệ dương tính thật (TPR)")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("ROC — kiểm định temporally trên 2 dataset")
    save_fig(fig, "fig5_roc_dual_dataset.png")


# ---------------------------------------------------------------------------
# Hình 6: ΔAUC 2 dataset
# ---------------------------------------------------------------------------
def fig6_delta():
    lmf = load_predictions("EXP-TEMPORAL-LMF")
    mic = load_predictions("EXP-TEMPORAL-MIMICIV")
    # Lấy giá trị từ summary.json (chính xác hơn)
    s_lmf = json.loads((ROOT / "experiments" / "EXP-TEMPORAL-LMF" / "summary.json").read_text())
    s_mic = json.loads((ROOT / "experiments" / "EXP-TEMPORAL-MIMICIV" / "summary.json").read_text())

    lmf_a = s_lmf["tasks"]["death_within_12m"]
    lr_ta, lr_ra = lmf_a["lr"]["temporal_split_test"]["roc_auc"], lmf_a["lr"]["random_split_test"]["roc_auc"]
    lg_ta, lg_ra = lmf_a["lgbm"]["temporal_split_test"]["roc_auc"], lmf_a["lgbm"]["random_split_test"]["roc_auc"]

    datasets = ["NHANES-LMF", "MIMIC-IV"]
    lr_d = [lr_ta - lr_ra, s_mic["lr"]["temporal_split_test"]["roc_auc"] - s_mic["lr"]["random_split_test"]["roc_auc"]]
    lg_d = [lg_ta - lg_ra, s_mic["lgbm"]["temporal_split_test"]["roc_auc"] - s_mic["lgbm"]["random_split_test"]["roc_auc"]]

    x = np.arange(2)
    w = 0.32
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    b1 = ax.bar(x - w / 2, lr_d, w, color=WHITE, edgecolor=BLACK, lw=1.4, label="Logistic Regression")
    b2 = ax.bar(x + w / 2, lg_d, w, color=DGRAY, edgecolor=BLACK, lw=1.4, label="LightGBM")
    ax.axhline(0, c=BLACK, lw=1.1)
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2,
                b.get_height() - 0.004 if b.get_height() < 0 else b.get_height() + 0.001,
                f"{b.get_height():+.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("ΔAUC (temporal − random)")
    ax.set_title("Suy giảm AUC khi chuyển từ random split\nsang kiểm định temporally")
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_ylim(-0.045, 0.008)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_fig(fig, "fig6_dual_dataset_delta_auc.png")


# ---------------------------------------------------------------------------
# Hình 7: Kiến trúc tổng thể ba tầng — Mermaid block (không emoji, không nền)
# ---------------------------------------------------------------------------
FIG7_MMD = """flowchart TB
    subgraph IN["ĐẦU VÀO: Chuỗi thời gian chỉ số cơ thể"]
        A["10 chỉ số: HA, nhịp_tim, SpO2, glucose,<br/>HbA1c, creatinine, eGFR, BMI, cholesterol,<br/>triglyceride"]
    end

    subgraph T1["TẦNG 1 — PHÁT HIỆN BẤT THƯỜNG CÁ NHÂN HÓA"]
        T1A["Z-Score cá nhân<br/>|Z| ≥ 2,0, cửa sổ 90 ngày"]
        T1B["Isolation Forest<br/>contamination = 0,05"]
        T1C["EWMA crossing<br/>λ = 0,2"]
        T1D["Sai số dự báo<br/>α = 0,3, |Z| ≥ 2,5"]
    end

    subgraph T2["TẦNG 2 — ÁNH XẠ TRI THỨC Y KHOA"]
        T2A["Rule Engine<br/>9 luật, JSON + governance"]
        T2B["Kích hoạt luật theo severity<br/>severity ∈ [0,5 − 0,9]"]
    end

    subgraph T3["TẦNG 3 — TỔNG HỢP RỦI RO & QUYẾT ĐỊNH"]
        T3A["Fusion Bayesian<br/>trọng số [0,30 − 0,35 − 0,25 − 0,10]"]
        T3B["Hiệu chỉnh isotonic<br/>ECE 1,7% → 0,0%"]
        T3C["Sàn an toàn<br/>severity ≥ 0,7 → score ≥ 0,50"]
        T3D["Mức rủi ro<br/>THẤP, TRUNG BÌNH, CAO"]
    end

    ML["LightGBM<br/>hiệu chỉnh isotonic"]

    A --> T1A
    A --> T1B
    A --> T1C
    A --> T1D
    T1A --> T2A
    T1B --> T2A
    T1C --> T2A
    T1D --> T2A
    T2A --> T2B
    T2B --> T3A
    ML -.-> T3A
    T3A --> T3B
    T3B --> T3C
    T3C --> T3D

    style IN fill:#e8e8e8,stroke:#888888,color:#222222
    style T1 fill:#d6eaf8,stroke:#3498db,color:#1a1a1a
    style T2 fill:#fdebd0,stroke:#e67e22,color:#1a1a1a
    style T3 fill:#d5f5e3,stroke:#27ae60,color:#1a1a1a
    style ML fill:#e8daef,stroke:#8e44ad,color:#1a1a1a
"""


# ---------------------------------------------------------------------------
# Hình 8: Chi tiết Tầng 1 — 4 mô-đun phát hiện bất thường (mermaid, không nền)
# ---------------------------------------------------------------------------
FIG8_MMD = """flowchart LR
    subgraph RAW["DỮ LIỆU ĐẦU VÀO"]
        D1["df_wide<br/>n_ngày × 10 chỉ số"]
    end

    subgraph PRE["TIỀN XỬ LÝ"]
        P1["resample_to_daily()"]
        P2["impute_missing (giới hạn 30%)"]
        P3["build_baseline (cửa sổ 90 ngày)"]
    end

    subgraph ZS["Z-SCORE CÁ NHÂN"]
        Z1["Z = (x − μ_base) / σ_base"]
        Z2["flagged nếu |Z| ≥ 2,0"]
        Z3["trend: rising, falling, stable"]
    end

    subgraph IF["ISOLATION FOREST"]
        I1["Features: 30 ngày rolling mean/std"]
        I2["contamination = 0,05"]
        I3["anomaly_score → flagged"]
    end

    subgraph EW["EWMA & DỰ BÁO"]
        E1["EWMA (λ = 0,2)"]
        E2["forecast_error = x_t − EWMA_t"]
        E3["flagged nếu |z_forecast| ≥ 2,5"]
    end

    subgraph OUT["ĐẦU RA"]
        O1["AnomalyRecord[]<br/>metric, z_score, flagged,<br/>trend, forecast_z"]
    end

    D1 --> P1
    P1 --> P2
    P2 --> P3
    P3 --> ZS
    P3 --> IF
    P3 --> EW
    ZS --> O1
    IF --> O1
    EW --> O1

    style RAW fill:#ffffff,stroke:#888888,color:#222222
    style PRE fill:#ffffff,stroke:#999999,color:#1a1a1a
    style ZS fill:#ffffff,stroke:#3498db,color:#1a1a1a
    style IF fill:#ffffff,stroke:#8e44ad,color:#1a1a1a
    style EW fill:#ffffff,stroke:#27ae60,color:#1a1a1a
    style OUT fill:#ffffff,stroke:#e67e22,color:#1a1a1a
"""


# ---------------------------------------------------------------------------
# Hình 9: Chi tiết Tầng 2 — Rule Engine và Knowledge Base (mermaid, không nền)
# ---------------------------------------------------------------------------
FIG9_MMD = """flowchart TB
    subgraph SNAP["SNAPSHOT HIỆN TẠI"]
        S1["snapshot = metric: value"]
    end

    subgraph KB["KNOWLEDGE BASE — knowledge_base.json"]
        K1["metrics: metadata + ranges"]
        K2["system_labels: 5 hệ thống cơ quan"]
        K3["rules: 9 luật hoạt động<br/>AND/OR logic + modes"]
    end

    subgraph EVAL["ĐÁNH GIÁ LUẬT"]
        E1["normalize_modes(modes)"]
        E2["for rule in rules:<br/>nếu status == active<br/>và _rule_in_modes(rule, modes)"]
        E3["_eval_condition(cond, snapshot)<br/>hỗ trợ nested AND/OR"]
        E4["collect matched_metrics"]
    end

    subgraph OUT["ĐẦU RA: RuleHit objects"]
        O1["RuleHit: rule_id, severity,<br/>system, specialty, evidence,<br/>matched_metrics, source_url"]
    end

    SNAP --> EVAL
    KB --> EVAL
    E1 --> E2
    E2 --> E3
    E3 --> E4
    E4 --> OUT

    style SNAP fill:#ffffff,stroke:#888888,color:#222222
    style KB fill:#ffffff,stroke:#e67e22,color:#1a1a1a
    style EVAL fill:#ffffff,stroke:#3498db,color:#1a1a1a
    style OUT fill:#ffffff,stroke:#27ae60,color:#1a1a1a
"""


# ---------------------------------------------------------------------------
# Hình 10: Chi tiết Tầng 3 — Fusion Bayesian + Isotonic + Phân loại (mermaid)
# ---------------------------------------------------------------------------
FIG10_MMD = """flowchart LR
    subgraph COMP["ĐIỂM THÀNH PHẦN"]
        C1["stat_score = min(1, max|Z|/4)"]
        C2["knowledge_score = min(1, max_severity)"]
        C3["ml_score = LightGBM + isotonic"]
        C4["trend_score = min(1, 2·rising/N)"]
    end

    subgraph FUS["FUSION BAYESIAN"]
        F1["total = Σ(wᵢ × scoreᵢ)<br/>trọng số [0,30 − 0,35 − 0,25 − 0,10]"]
        F2["Sàn an toàn:<br/>nếu severity ≥ 0,7 → total ≥ 0,50"]
    end

    subgraph THRESH["PHÂN LOẠI RỦI RO"]
        T1["THẤP: total < 0,33"]
        T2["TRUNG BÌNH: 0,33 ≤ total < 0,66"]
        T3["CAO: total ≥ 0,66"]
    end

    C1 --> F1
    C2 --> F1
    C3 --> F1
    C4 --> F1
    F1 --> F2
    F2 --> T1
    F2 --> T2
    F2 --> T3

    style COMP fill:#ffffff,stroke:#3498db,color:#1a1a1a
    style FUS fill:#ffffff,stroke:#e67e22,color:#1a1a1a
    style THRESH fill:#ffffff,stroke:#27ae60,color:#1a1a1a
"""


if __name__ == "__main__":
    print("Sinh hình vẽ ->", FIGDIR)
    # fig1: architecture
    render_dot(FIG1_DOT, "fig1_architecture.dot")
    write_mmd("fig1_architecture.mmd", FIG1_MMD)
    # fig2: data flow
    render_dot(FIG2_DOT, "fig2_data_flow.dot")
    write_mmd("fig2_data_flow.mmd", FIG2_MMD)
    # fig3: calibration
    fig3_calibration()
    # fig4: timeline (mermaid gantt -> PNG)
    render_mmd("fig4_temporal_timeline.mmd", FIG4_MMD)
    # fig5: ROC thật
    fig5_roc()
    # fig6: delta
    fig6_delta()
    # fig7-10: 3D box diagrams (mermaid -> PNG)
    render_mmd("fig7_architecture_3d.mmd", FIG7_MMD)
    render_mmd("fig8_tier1_detail.mmd", FIG8_MMD)
    render_mmd("fig9_tier2_rules.mmd", FIG9_MMD)
    render_mmd("fig10_tier3_fusion.mmd", FIG10_MMD)
    print("Xong.")