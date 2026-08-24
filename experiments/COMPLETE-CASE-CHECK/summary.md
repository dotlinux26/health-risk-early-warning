# P2.2 Complete-case check — impute median vs bỏ mẫu thiếu

- Dataset: `data/datasets/nhanes_merged.csv` (glucose_fasting khuyết 52%)
- Cùng split + seed cho hai arm; complete-case loại mẫu thiếu trong từng phần.

| Model | AUC imputed | AUC complete-case | Δ AUC | PR-AUC Δ | Brier Δ | Giữ được % test |
|---|---|---|---|---|---|---|
| lgbm | 0.9349±0.0031 | 0.9290±0.0060 | -0.0059 | -0.0079 | +0.0029 | 45.1% |
| xgb | 0.9356±0.0028 | 0.9311±0.0057 | -0.0045 | -0.0069 | +0.0022 | 45.1% |
| lr | 0.8844±0.0078 | 0.9002±0.0070 | +0.0158 | +0.0126 | -0.0129 | 45.1% |

**Kết luận: ổn định cho lgbm, xgb (|ΔAUC| trung bình < 0.01); lệch có hệ thống cho lr (+0.0158). Sản xuất dùng LightGBM → impute chấp nhận được; mô hình tuyến tính bị impute median làm giảm AUC (glucose đặc thành 1 giá trị ở 52% mẫu).**

> Tiêu chí nghiệm thu docs/16 §3.5: ổn định < 0.01 → impute vô hại;
> lệch lớn → báo cáo kèm phương sai do impute. Δ dương = complete-case tốt hơn.