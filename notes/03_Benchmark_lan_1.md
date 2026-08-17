# Kết quả benchmark lần 1 — Baseline 6 mô hình (NHANES 2017–2018)

> Ghi chép nội bộ. Số liệu **từng là kết quả thực tế** chạy bằng
> `python scripts/run_benchmark.py`, lưu tại `experiments/` (evidence package
> truy ngược từng Experiment ID). Không viết trước số liệu.

## 1. Bối cảnh

- **Dataset**: `data/datasets/nhanes_2017_2018.csv` (NHANES CDC), n = 4949,
  positive = 2503 (~50,6%), target = `label` (nguy cơ tăng huyết áp hoặc đái tháo
  đường).
- **Features (7)**: systolic_bp, diastolic_bp, heart_rate, glucose_fasting,
  hba1c, creatinine, bmi.
- **Split**: 70 / 15 / 15, test khóa lại, không dùng để tuning.
- **Seeds**: 42, 52, 62, 72, 82 → báo mean ± std.
- **Preprocessing**: imputer median fit trên train (tránh data leakage), áp
  chung cho mọi model.
- **Protocol**: `notes/02`; mã nguồn: `src/experiments/`.

## 2. Kết quả (mean ± std trên test set)

| Model | ROC-AUC | PR-AUC | Accuracy | Precision | Recall | Specificity | F1 | Brier |
|---|---|---|---|---|---|---|---|---|
| lr | 0.918±0.008 | 0.930±0.007 | 0.822±0.012 | 0.841±0.017 | 0.798±0.011 | 0.845±0.020 | 0.819±0.011 | 0.116±0.006 |
| rf | 0.945±0.006 | 0.959±0.004 | 0.883±0.004 | 0.967±0.009 | 0.797±0.008 | 0.972±0.008 | 0.873±0.004 | 0.086±0.003 |
| lgbm | 0.945±0.006 | 0.959±0.004 | 0.887±0.006 | 0.952±0.009 | 0.819±0.011 | 0.958±0.009 | 0.880±0.007 | 0.082±0.004 |
| xgb | 0.946±0.005 | 0.960±0.004 | 0.882±0.008 | 0.939±0.009 | 0.820±0.013 | 0.946±0.009 | 0.876±0.009 | 0.083±0.004 |
| mlp | 0.842±0.067 | 0.867±0.057 | 0.771±0.058 | 0.807±0.075 | 0.729±0.071 | 0.814±0.089 | 0.763±0.056 | 0.167±0.038 |
| fttransformer | 0.933±0.008 | 0.949±0.005 | 0.864±0.010 | 0.903±0.023 | 0.819±0.013 | 0.910±0.025 | 0.859±0.009 | 0.099±0.007 |

## 3. Đọc kết quả — không nhảy kết luận

1. **Nhóm boosting (XGB / LGBM / RF) gần nhau**: chênh lệch ROC-AUC giữa XGB và
   LGBM là 0.001 — chưa đủ để kết luận model nào hơn; cần statistical test nếu
   đưa vào báo cáo (đề xuất: paired test trên cùng test fold).
2. **FT-Transformer chạy tốt (0.933) dù chưa tuning** — đáng đầu tư tuning vì là
   đại diện họ Transformer tabular, khác biệt về kiến trúc với boosting.
3. **MLP yếu nhất và phương sai lớn (0.842±0.067)** — phù hợp kỳ vọng trên
   dataset nhỏ, không có pretraining; ghi vào báo cáo như baseline neural đơn
   giản, không gắng ép MLP.
4. **LR (0.918) là baseline tuyến tính hợp lý** — chênh với boosting ~0.03, đủ
   để minh chứng "model phi tuyến cải thiện rõ trên bài toán này".
5. Nhận xét cuối cho báo cáo **chờ** thực nghiệm fusion (Bước 6) rồi mới viết.

## 4. Việc tiếp theo

- Bước 6: thực nghiệm fusion (ML-only / Stat+Rule+Trend / Full Fusion) — các
  thí nghiệm EXP-F0x như `notes/02` mục 3.2.
- Xem kết quả trực quan tại trang **/benchmark** (bảng tổng hợp, so sánh luận
  giải từng model trên cùng một ca, evidence package từng thí nghiệm).
- Nếu cần độ chắc cho số liệu: chạy lại với nhiều seed hơn hoặc paired test.

---

*Ghi chép nội bộ. Không thay thế chẩn đoán của bác sĩ.*