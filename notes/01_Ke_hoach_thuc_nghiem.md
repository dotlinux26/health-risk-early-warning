# Kế hoạch thực nghiệm & triển khai (ghi chép nội bộ)

> Ghi chép nội bộ — đối chiếu giữa **báo cáo đề cương** (`docs/AI y tế.md`)
> và **hiện trạng code thực tế** trong repo. Mục tiêu: khóa thứ tự làm việc để
> kết quả chạy ra khớp với báo cáo, không viết một đằng chạy một nẻo.

---

## 1. Tình hình hiện tại (đã kiểm tra code thực tế)

### 1.1. Báo cáo đề cương

- **Mở đầu**: đầy đủ — tính cấp thiết, mục tiêu, đối tượng/phạm vi, phương pháp,
  đóng góp mới, bố cục.
- **Chương 1**: đầy đủ — CDSS, EHR, ML trong cảnh báo y tế, khoảng trống nghiên
  cứu, đề xuất khung (mục 1.4).
- **Chương 2 & 3**: còn trống (chỉ có tiêu đề chương) — phần này sẽ được viết
  dựa trên quá trình thực nghiệm bên dưới.

### 1.2. Code thực tế

| Thành phần | Hiện trạng |
|---|---|
| Dataset | `data/datasets/nhanes_2017_2018.csv` (NHANES CDC), Pima, Cleveland |
| Model sản xuất | `data/models/risk_lgbm_real.joblib` (LightGBM) |
| Benchmark (thực nghiệm) | `src/experiments/` — 6 model × 5 seed, split 70/15/15, evidence package ra `experiments/` |
| Giao diện xem kết quả | `/benchmark` — bảng tổng hợp + so sánh luận giải từng model |
| Đánh giá hiện tại | `scripts/train_nhanes.py`: Stratified 5-fold CV, báo AUC + AUPRC |
| Feature | `src/data/features.py`: rolling, slope, z-score, EWMA |
| Pipeline hệ thống | 3 tầng: stat (Tầng 1) → rule (Tầng 2) → ML + fusion (Tầng 3) |
| Luật y khoa | `knowledge_base.json` v0.2.0 (9 luật, metadata nguồn đầy đủ) |

### 1.3. Lỗ hổng so với chuẩn khoa học

1. Chưa có Train/Validation/Test split chuẩn, test chưa được khóa lại.
2. Chưa có feature dictionary (danh sách đặc trưng, ý nghĩa, nguồn).
3. Mới chỉ chạy 1 model (LightGBM), chưa có baseline so sánh.
4. Chưa chạy đa seed (hiện chỉ seed 42), chưa báo `mean ± std`.
5. Chưa có calibration (Brier, calibration curve).
6. Chưa có Experiment ID / evidence package để truy ngược kết quả.
7. Chưa có thực nghiệm chứng minh contribution của fusion (ML-only vs Fusion).

---

## 2. Kế hoạch làm việc theo thứ tự

### Bước 1 — Khóa outline Chương 2 & Chương 3

- **Chương 2** (phương pháp): CDSS overview → kiến trúc fusion 3 tầng → thống
  kê cá nhân hóa → tri thức y khoa/luật → bài toán ML → các mô hình khảo sát →
  cơ chế fusion. Công thức fusion lấy đúng từ code thực tế, không chốt sớm khi
  implementation cuối chưa hoàn thiện.
- **Chương 3** (thực nghiệm): dataset → feature engineering → split → các model
  → metrics → kết quả → ca kiểm thử hệ thống.

### Bước 2 — Experimental Protocol v1.0

Khóa các quyết định trước khi chạy bất kỳ model nào, gồm: dataset, inclusion/
exclusion, target definition, feature definition, missing-value policy,
train/val/test split, CV strategy, preprocessing, models, hyperparameter policy,
random seeds, metrics, statistical comparison, explainability, evidence
artifacts, reproducibility.

Quyết định ban đầu:

- **Split**: 70 / 15 / 15 (train / val / test), test khóa lại, không dùng để tuning.
- **CV**: Stratified K-fold trên train (5-fold).
- **Seeds**: 42, 52, 62, 72, 82 → báo `mean ± std`.
- **Metrics**: ROC-AUC, PR-AUC, Accuracy, Precision, Recall, Specificity, F1,
  Brier Score.
- **Preprocessing**: imputer/scaler chỉ fit trên train để tránh data leakage.

### Bước 3 — Xác định dataset / target / features

- Xác nhận lại số mẫu NHANES thực tế (`n`, positive ratio) trước khi đưa vào
  báo cáo.
- Lập **feature dictionary** (bảng: feature, ý nghĩa, loại, nguồn) cho NHANES.

### Bước 4 — Pipeline thực nghiệm so sánh các model

Phase 1 (bắt buộc, **đã xong — xem `notes/03`**): Logistic Regression, Random
Forest, XGBoost, LightGBM, MLP, FT-Transformer — cùng feature, cùng split, cùng
metrics → bảng so sánh. Chi tiết protocol: `notes/02`.

Phase 2 (tùy chọn): TabNet, TabPFN, bản FT-Transformer đầy đủ hơn.

### Bước 5 — Evidence package

Mỗi model sinh ra thư mục `experiments/EXP-ML-<MODEL>-<SEED>/` chứa: config,
metrics.json, predictions.csv, ROC/PR curve, feature importance, model.joblib.
Kèm `summary.json/md/csv` tổng hợp mean ± std theo model (**đã triển khai** —
xem `experiments/`, trang `/benchmark`).
Mỗi lần chạy là một record → trả lời được câu hỏi "con số này lấy ở đâu" cho
hội đồng.

### Bước 6 — Thực nghiệm chứng minh contribution của fusion

3 mức:

- **EXP-A**: ML only (dữ liệu → ML → risk).
- **EXP-B**: từng nguồn riêng (stat only / rule only / trend only / ML only).
- **EXP-C**: fusion đầy đủ 4 nguồn.

So sánh để hỗ trợ claim *"fusion mang lại giá trị bổ sung so với ML đơn lẻ"*.
Nếu kết quả không cải thiện, báo cáo trung thực và chuyển contribution sang khả
năng tích hợp, giải thích và robustness — không ép kết quả.

### Bước 7 — Viết Chương 2 & 3 khớp với kết quả thực nghiệm

- Bê bảng kết quả, biểu đồ, evidence từ Bước 4–6 vào Chương 3.
- Chương 2 viết công thức fusion đúng với code thực tế.

---

## 3. Checklist (phần báo cáo)

- Viết Chương 2 dựa trên kiến trúc thực tế (`docs/03`, `docs/07`, `docs/08`).
- Chương 3 chờ kết quả thực nghiệm (Bước 4–6) rồi mới viết, tránh viết trước số liệu.
- Ghi chính xác dataset: tên, nguồn, version, số mẫu, positive ratio, missing.
- Đồng nhất thuật ngữ: "Statistical Anomaly Detection" (Tầng 1), "fusion", không
  nói "AI là chuẩn quyết định".
- Số liệu AUC/SHAP phải truy ra từ Experiment ID, không viết kiểu "chạy Python ra".

---

*Ghi chép nội bộ. Không thay thế chẩn đoán của bác sĩ.*
