# 06. Báo cáo huấn luyện mô hình: Kết quả, Giới hạn và Hướng chuẩn mực hóa

> Trạng thái: tháng 08/2026. Tài liệu này tóm tắt kết quả huấn luyện hiện tại,
> thẳng thắn nêu các giới hạn của quy trình và đề xuất hướng đi để mô hình
> thông minh hơn, tin cậy hơn và có thể công bố.

---

## 1. Tổng quan

Trước giai đoạn này, mô hình ML của hệ thống (LightGBM ở Tầng 3) chỉ là khung:
trọng số `ml` trong điểm rủi ro luôn bằng 0, mọi cảnh báo đều do thống kê
(Tầng 1) và luật y khoa (Tầng 2) tạo ra. Đợt triển khai này bổ sung:

1. **Dữ liệu tổng hợp có nhãn** (`src/models/synthetic_data.py`) — bệnh nhân ảo
   có bệnh lý ẩn, đặc trưng 60 ngày đầu dự đoán sự kiện trong 30 ngày tiếp theo.
2. **Huấn luyện LightGBM thật** (`src/models/train.py`) — model đã được train và
   đưa vào điểm rủi ro (`ml_score`) trong `assess_patient`.
3. **Tải dataset chính thống** (`scripts/download_datasets.py`) — Pima Diabetes
   và Cleveland Heart Disease từ kho UCI, và đánh giá mô hình trên đó
   (`scripts/train_real_datasets.py`).
4. **Phát hiện bất thường theo sai số dự báo** (`src/tier1_anomaly/forecast.py`)
   — so giá trị thực với dự báo một bước (EWMA offline, sẵn sàng nâng cấp
   Chronos/TimesFM), phần bổ sung của hướng time-series foundation model.

---

## 2. Kết quả huấn luyện hiện tại

### 2.1. Trên dữ liệu tổng hợp (đánh giá nội bộ pipeline)

Bài toán: đặc trưng 60 ngày đầu → dự đoán sự kiện vượt ngưỡng trong ngày 61–90.

| Tham số | Giá trị |
|---|---|
| Số bệnh nhân | 480 (khỏe mạnh 120, tăng huyết áp 120, đái tháo đường 120, suy thận 120) |
| Số ca dương tính (sự kiện trong cửa sổ) | 143 (30%) |
| Đặc trưng | 99 cột (rolling mean/std, delta, pct, slope, ewma) |
| Train / Test | 360 / 120 (chia theo bệnh nhân) |
| AUC test | **0.9874** |
| AUPRC test | **0.9096** |

Nhận xét: AUC rất cao vì tín hiệu trong dữ liệu tổng hợp mạnh và rõ ràng
(đường cong bệnh được sinh theo quy luật có kiểm soát). Đây là điểm cần thận
trọng — số cao không chứng minh được chất lượng trên dữ liệu thực.

### 2.2. Trên dữ liệu thật (UCI) — Cross-validation 5-fold theo bệnh nhân

| Dataset | N (dương tính) | AUC | AUPRC |
|---|---|---|---|
| Pima Indians Diabetes (UCI) | 768 (268) | **0.8116 ± 0.0290** | **0.6769** |
| Cleveland Heart Disease (UCI) | 303 (139) | **0.8906 ± 0.0142** | **0.8838** |

Đặc trưng quan trọng nhất (Cleveland): vùng đau thắt ngực, thalach (nhịp tim
tối đa khi gắng sức), oldpeak (chênh ST) — khớp với y văn. Với Pima: glucose,
bmi, age — khớp với y văn.

Nhận xét: mức AUC này là **thực tế** (0.81–0.89), ngang bằng với kết quả các
nghiên cứu công bố trên cùng bộ dữ liệu. Điều này xác nhận pipeline huấn luyện
(đặc trưng hóa → LightGBM → đánh giá CV) hoạt động đúng trên dữ liệu thật.

---

## 3. Giới hạn của quy trình huấn luyện hiện tại

### 3.1. Giới hạn về dữ liệu

| Giới hạn | Hệ quả |
|---|---|
| Dữ liệu tổng hợp không phải dữ liệu lâm sàng | AUC 0.987 gây ảo tưởng; mô hình học đúng "quy luật sinh" chứ không phải sinh lý thực |
| Dataset thật là **cắt ngang** (một dòng/bệnh nhân) | Không kiểm chứng được bài toán cốt lõi: cảnh báo sớm theo chuỗi thời gian |
| Chưa có **dữ liệu dọc có nhãn thật** | Không đánh giá được thời điểm phát hiện sớm so với khởi phát bệnh |
| Dân số hẹp | Pima là phụ nữ người Mỹ gốc Pima Ấn Độ; Cleveland là dân số tham chiếu nghiên cứu — không đại diện người Việt |
| Mất cân bằng lớp và định nghĩa nhãn khác nhau | Pima nhãn theo tiêu chuẩn 1988; không tương thích trực tiếp với nhãn "sự kiện" của bài toán hiện tại |
| Kích thước mẫu nhỏ | Phương sai ước lượng cao; khó suy diễn ở phân nhóm |

### 3.2. Giới hạn về phương pháp

1. **Nhãn tổng hợp không có "ground truth" lâm sàng** — sự kiện trong dữ liệu
   tổng hợp được định nghĩa bằng ngưỡng, không qua xác nhận bác sĩ.
2. **Chưa có chiến lược đánh giá theo thời gian** (temporal split) — chia ngẫu
   nhiên theo bệnh nhân dễ lạc quan hơn thực tế triển khai.
3. **Chưa chuẩn hoá cách xử lý dữ liệu thiếu** giữa train và inference — hiện
   chỉ fill 0 cho cột thiếu; cần thống nhất pipeline.
4. **Chưa hiệu chuẩn (calibration)** — xác suất đầu ra chưa phải xác suất thật;
   chưa tối ưu ngưỡng quyết định theo chi phí lâm sàng.
5. **Điểm số tổng hợp (stat/knowledge/ml/trend) dùng trọng số tĩnh** — chưa
   được tối ưu bằng dữ liệu, chưa có lý thuyết phân bổ.
6. **Chưa có quy trình phiên bản hoá** dữ liệu, mô hình, số liệu đánh giá —
   khó tái lập kết quả.
7. **Thiếu báo cáo tuân theo khuyến nghị TRIPOD-AI** — chưa có bảng đặc trưng,
   mô tả mẫu, phân tích rủi ro thiên lệch.

---

## 4. Hướng huấn luyện chuẩn mực và thông minh hơn

### 4.1. Về dữ liệu (ưu tiên cao nhất)

| Nguồn | Truy cập | Phù hợp cho |
|---|---|---|
| **NHANES (CDC)** | Miễn phí, tải XPT trực tiếp | Chuỗi dài 2 kỳ khám, huyết áp, đường huyết, creatinine; xây nhãn tăng huyết áp/tiểu đường/giảm eGFR |
| **MIMIC-IV** | Đăng ký PhysioNet | Dữ liệu bệnh viện, nhiều chỉ số dọc; cần thủ tục truy cập |
| **OhioT1DM** | Đăng ký | CGM đường huyết liên tục — tốt cho Tầng 1 |
| **UK Biobank** | Đăng ký, chi phí | Quy mô lớn, dọc; phục vụ cấp 3 |
| Bộ dữ liệu nội bộ bệnh viện | Thỏa thuận | Tốt nhất về độ khớp dân số Việt Nam |

Việc đầu tiên nên làm: **xây bộ dữ liệu longitudinal có nhãn** (ví dụ từ NHANES:
ghép 2–4 kỳ khám theo người tham gia, nhãn = xuất hiện bệnh ở kỳ sau). Đây là
điều kiện tiên quyết để mô hình có ý nghĩa thực sự cho cảnh báo sớm.

### 4.2. Về phương pháp huấn luyện

1. **Split theo bệnh nhân + theo thời gian** — `GroupKFold` theo patient_id;
   với dữ liệu dọc, train trên khoảng thời gian trước, test trên khoảng sau
   (không rò rỉ tương lai).
2. **Thống nhất pipeline thiếu dữ liệu** — cùng một `SimpleImputer`/feature
   pipeline cho train và inference (dùng `sklearn.Pipeline`), không fill 0 tùy tiện.
3. **Hiệu chuẩn xác suất** — `CalibratedClassifierCV` (Platt/Isotonic) để đầu ra
   đọc được như xác suất thật; tối ưu ngưỡng bằng chỉ số Youden hoặc net benefit.
4. **Đánh giá bằng cả AUPRC và độ lớn tác động lâm sàng** — AUPRC quan trọng
   hơn AUC khi lớp dương hiếm; thêm đường cong chi phí quyết định.
5. **Tối ưu siêu tham số có kiểm chứng** — `Optuna` + CV nested để tránh
   overfit khi chọn tham số.
6. **Quản lý thí nghiệm** — phiên bản hoá data/model bằng mã định danh
   (hash dữ liệu + tham số), lưu metric mỗi lần train; mô hình lưu kèm
   `feature_names` và siêu tham số.
7. **Giải thích & kiểm toán** — SHAP trên cả train lẫn inference; kiểm tra mô
   hình có dùng "học nhiễu" (feature như mã bệnh nhân, ngày tháng) không.
8. **Báo cáo chuẩn TRIPOD-AI** — mô tả mẫu, cách xử lý thiếu, cách chia dữ
   liệu, phân phối điểm số, bảng đặc trưng, độ không chắc chắn.

### 4.3. Về kiến trúc mô hình

- **Time-series foundation model (Cấp 2 trong docs/05):** tích hợp Chronos /
  TimesFM để dự báo chuỗi; bất thường = lỗi dự báo vượt ngưỡng cá nhân. Phần
  `forecast.py` đã có sẵn backend `chronos` — cài `chronos-forecasting` và
  `torch` là chạy được, tự fallback về EWMA khi chưa cài.
- **Ensemble nhẹ:** gộp LightGBM + LogisticRegression trên cùng đặc trưng để
  giảm phương sai; hoặc stacking với mô hình xác suất đơn giản.
- **Uncertainty:** dùng đầu ra phân phối của mô hình dự báo hoặc phân phối
  posterior để trả về "khoảng tin cậy" thay vì con số cứng.
- **Trọng số 4 nguồn điểm** nên được học từ dữ liệu (huấn luyện meta-classifier
  trên stat/knowledge/ml/trend) thay vì để tĩnh.

---

## 5. Lộ trình đề xuất cho giai đoạn tiếp theo

| Giai đoạn | Việc cần làm | Đầu ra mong đợi |
|---|---|---|
| G1 (1–2 tuần) | Xây dataset longitudinal từ NHANES (huyết áp, đường huyết, creatinine, eGFR theo kỳ khám) | `data/datasets/nhanes_*` + bộ tiền xử lý chuẩn |
| G2 | Huấn luyện LightGBM trên NHANES với `GroupKFold` + temporal split, calibration, ngưỡng tối ưu | Báo cáo AUC/AUPRC/calibration trên test thật |
| G3 | Tích hợp Chronos/TimesFM vào Tầng 1, benchmark lỗi dự báo vs Z-Score | Báo cáo so sánh 2 phương pháp |
| G4 | Tối ưu trọng số tổng hợp bằng dữ liệu; phiên bản hoá + TRIPOD-AI | Báo cáo mô hình chuẩn hoá, tái lập được |

---

## 6. Cách chạy lại các script

```bash
# 1. Train LightGBM trên dữ liệu tổng hợp có nhãn (model -> data/models/risk_lgbm.joblib)
python -m src.models.train --n-per-condition 120 --seed 42

# 2. Tải dataset chính thống (UCI) về data/datasets/
python scripts/download_datasets.py

# 3. Train + đánh giá LightGBM trên dữ liệu thật (kết quả -> report/train_real_results.json)
python scripts/train_real_datasets.py
```

---

*Tài liệu nội bộ. Không thay thế chẩn đoán của bác sĩ.*
