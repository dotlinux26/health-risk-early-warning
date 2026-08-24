# Thư mục thực nghiệm (experiments/)

Nơi lưu **kết quả benchmark đa mô hình** theo Experimental Protocol v1.0
(xem `notes/01`, `notes/02`). Mỗi lần chạy là một evidence package — truy ngược
được toàn bộ: config, metrics, predictions, đồ thị, feature importance.

## Cách chạy

```bash
python scripts/run_benchmark.py                          # 6 model × 5 seed
python scripts/run_benchmark.py --models lr rf           # chỉ một số model
python scripts/run_benchmark.py --seeds 42 52 62         # chọn seed
python scripts/run_benchmark.py --out experiments/run1   # thư mục riêng
```

Ngoài benchmark chính, các thí nghiệm độ bền vững / validation:

| Thư mục | Script | Câu hỏi |
|---|---|---|
| `LABEL-SENSITIVITY/` | `scripts/run_label_sensitivity.py` | Đổi định nghĩa nhãn thì AUC đổi bao nhiêu? (K2) |
| `BASELINE-STABILITY/` | `scripts/run_baseline_stability.py` | Cửa sổ baseline N ngày ảnh hưởng Tầng 1? (K3) |
| `WEIGHT-SENSITIVITY/` | `scripts/run_weight_sensitivity.py` | Trọng số fusion thay đổi thì kết luận lật? (K4) |
| `EXP-ML-*/calibration.json` | `scripts/run_calibration*.py` (xem docs/15 §5F) | Xác suất có "đúng là xác suất"? (K1) |
| **`EXP-TEMPORAL-LMF/`** | `scripts/fetch_nhanes_mortality.py` + `scripts/run_temporal_validation.py` | Dự báo outcome tương lai thật (tử vong NHANES-LMF), split theo thời gian + lead time (P2, docs/18) |
| **`COMPLETE-CASE-CHECK/`** | `scripts/run_complete_case_check.py` | Impute median glucose 52% có vô hại? (P2.2, docs/16 §3.5) |

- Dataset mặc định: `data/datasets/nhanes_merged.csv` (NHANES 3 chu kỳ
  2015–2016, 2017–2018, 2021–2023; n = 16.314, positive ~49%).
- Split: 70 / 15 / 15 (train / val / test), test khóa lại.
- Seeds: 42, 52, 62, 72, 82 → kết quả báo `mean ± std`.
- Preprocessing (imputer median) fit trên **train** để tránh data leakage.
- Mô hình cần thư viện chưa cài (xgboost, torch...) tự bỏ qua, không làm hỏng chạy.

## Cấu trúc output

```
experiments/
├── EXP-ML-<MODEL>-<SEED>/      # evidence package từng lần chạy
│   ├── config.json             # cấu hình experiment
│   ├── metrics.json            # ROC-AUC, PR-AUC, F1, Brier, ...
│   ├── predictions.csv         # y_true + proba trên test set
│   ├── curves.png              # ROC / PR / Calibration
│   ├── feature_importance.csv  # top đặc trưng đóng góp
│   └── model.joblib            # model đã huấn luyện (tái sử dụng cho inference)
├── summary.json                # tổng hợp mean ± std theo model
├── summary.md                  # bảng markdown đưa thẳng vào báo cáo
├── summary.csv                 # bảng CSV
└── README.md
```

## Xem kết quả trực quan

Mở server (`./run_api.sh start`) rồi vào **http://127.0.0.1:8000/benchmark**:

1. **Bảng tổng hợp** — so sánh 6 model (có cột tốt nhất được tô đậm).
2. **So sánh luận giải** — nhập chỉ số của một bệnh nhân, xem từng model đánh
   giá nguy cơ dựa vào đặc trưng nào (phương pháp perturbation, không cần SHAP).
3. **Chi tiết thí nghiệm** — evidence package từng lần chạy kèm đồ thị ROC/PR.

## Ghi chú khoa học

- Mỗi con số trong báo cáo phải truy ra từ Experiment ID, ví dụ
  `EXP-ML-LGBM-42` → `metrics.json`. Không trích số kiểu "chạy Python ra".
- Test set cuối cùng khóa lại; nếu chỉnh model/hyperparameter phải chạy lại và
  ghi đè Experiment ID mới (thêm `--out experiments/run2`).