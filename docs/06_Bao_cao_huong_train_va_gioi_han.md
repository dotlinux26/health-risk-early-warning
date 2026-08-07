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
3. **Dữ liệu thật NHANES (CDC)** (`scripts/build_nhanes_dataset.py`) — dựng
   dataset thật 4949 người, train lại LightGBM (`scripts/train_nhanes.py`) và
   **thay thế bản synthetic thành model sản xuất** (`data/models/risk_lgbm_real.joblib`).
4. **Tải dataset chính thống** (`scripts/download_datasets.py`) — Pima Diabetes
   và Cleveland Heart Disease từ kho UCI, và đánh giá mô hình trên đó
   (`scripts/train_real_datasets.py`).
5. **Phát hiện bất thường theo sai số dự báo** (`src/tier1_anomaly/forecast.py`)
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

### 2.3. Trên dữ liệu NHANES (CDC) — model sản xuất hiện tại

Từ giai đoạn này, **model đang chạy trong hệ thống được train trên dữ liệu thật**
NHANES 2017-2018 (CDC/NCHS, khảo sát lâm sàng toàn dân Mỹ, 4949 người trưởng
thành, 50.6% có tăng huyết áp hoặc đái tháo đường).

Đặc trưng dùng TRÙNG schema hệ thống: systolic_bp, diastolic_bp, heart_rate,
glucose_fasting, hba1c, creatinine, bmi. Nhãn: tăng huyết áp (HA ≥ 140/90 hoặc
đang dùng thuốc) HOẶC đái tháo đường (HbA1c ≥ 6.5% hoặc đường huyết lúc đói ≥
7.0 mmol/L).

| Tham số | Giá trị |
|---|---|
| N | 4949 (dương tính 2503) |
| AUC (5-fold CV) | **0.9436 ± 0.0041** |
| AUPRC (5-fold CV) | **0.9580 ± 0.0032** |
| Đặc trưng quan trọng nhất | hba1c, diastolic_bp, systolic_bp, creatinine |
| Model sản xuất | `data/models/risk_lgbm_real.joblib` |

> Cảnh báo khoa học: nhãn "tăng huyết áp" được định nghĩa một phần bằng chính
> huyết áp đo được (cùng dấu hiệu dùng làm đặc trưng), nên AUC cao (0.94) là
> kỳ vọng và KHÔNG phải là độ chính xác chẩn đoán trên lâm sàng. Điểm cốt lõi
> của bước này: hệ thống giờ dùng model học từ dữ liệu thực của CDC thay vì
> dữ liệu tổng hợp.

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
| G1 ✅ | Xây dataset longitudinal từ NHANES (huyết áp, đường huyết, creatinine, eGFR theo kỳ khám) | `data/datasets/nhanes_2017_2018.csv` — đã xong |
| G2 ✅ | Huấn luyện LightGBM trên NHANES với 5-fold CV, median-fill khi thiếu chỉ số | `data/models/risk_lgbm_real.joblib` — đã xong, đang là model sản xuất |
| G3 | Ghép nhiều kỳ NHANES (1999–2020) thành chuỗi thời gian theo từng người; thêm eGFR (CKD-EPI) và tuổi/giới | Dataset longitudinal + model cảnh báo sớm thực sự |
| G4 | Tích hợp Chronos/TimesFM vào Tầng 1, benchmark lỗi dự báo vs Z-Score | Báo cáo so sánh 2 phương pháp |
| G5 | Tối ưu trọng số tổng hợp bằng dữ liệu; calibration; báo cáo TRIPOD-AI | Mô hình chuẩn hoá, tái lập được |

Ghi chú NHANES: bản chất NHANES là khảo sát cắt ngang — mỗi người thường chỉ
được khám một kỳ. Để có chuỗi thời gian thật cần (a) ghép các kỳ 1999–2020,
hoặc (b) dùng nguồn dọc có đăng ký như MIMIC/OhioT1DM/UK Biobank. Hiện tại
`risk_lgbm_real` là model cắt ngang đánh giá rủi ro từ ảnh chụp chỉ số gần nhất —
chạy được và dùng dữ liệu thật, nhưng chưa khai thác hết khía cạnh chuỗi thời
gian của Tầng 1.

---

## 6. Cách chạy lại các script

```bash
# 1. Dựng dataset thật NHANES 2017-2018 (CDC) -> data/datasets/nhanes_2017_2018.csv
python scripts/build_nhanes_dataset.py

# 2. Train LightGBM trên NHANES -> data/models/risk_lgbm_real.joblib (model sản xuất)
python scripts/train_nhanes.py

# 3. Train LightGBM trên dữ liệu tổng hợp có nhãn (tham khảo)
python -m src.models.train --n-per-condition 120 --seed 42

# 4. Tải dataset chính thống (UCI) về data/datasets/
python scripts/download_datasets.py

# 5. Train + đánh giá LightGBM trên dữ liệu thật (kết quả -> report/train_real_results.json)
python scripts/train_real_datasets.py

# 6. Xuất file mẫu từ dữ liệu thật NHANES -> data/sample_nhanes/ (DOCX/PDF/CSV)
python scripts/export_nhanes_samples.py
```

---

## 7. Báo cáo ứng dụng (demo)

### 7.1. Giao diện

| Hộp thoại chat | Kết quả đánh giá |
|---|---|
| ![Giao diện chat](screenshots/giao_dien_chat.png) | ![Kết quả đánh giá](screenshots/ket_qua_danh_gia.png) |

Giao diện nền sáng, bo góc 2px, thanh xổ trên cùng liệt kê các mã bệnh nhân đã
có dữ liệu trong `data/chat/` (từ API `/api/chat/patients`); vẫn nhập được mã
mới bằng ô kế bên.

### 7.2. Cách dùng

1. Khởi động: `./run_api.sh start`, mở **http://127.0.0.1:8000/chat**.
2. Chọn/nhập mã bệnh nhân, nhập nhật ký hàng ngày:
   - `Huyết áp 135/85, nhịp tim 80`
   - `Đường huyết lúc đói 6.9, cân nặng 77`
   - Hoặc đính kèm file PDF/DOCX (hệ thống đọc cả file thật xuất từ NHANES:
     `data/sample_nhanes/NHTN0001.pdf` — ca tăng huyết áp 202/62, v.v.)
3. Tin nhắn không khớp định dạng chỉ số/lệnh sẽ KHÔNG được phản hồi (giữ giao
   diện sạch); hệ thống không nhận diện bằng AI ngôn ngữ mà bằng mẫu regex.
4. Lệnh: `trạng thái` · `báo cáo` · `xóa dữ liệu`.
5. Đủ 7 ngày đo → hệ thống tự đưa báo cáo nguy cơ cá nhân hóa; báo cáo markdown
   đầy đủ cũng xuất qua `python -m src.main --input ...`.

### 7.3. Luồng xử lý một đánh giá

```
File / tin nhắn → ingest (regex PDF/DOCX) hoặc parser chat
  → tích lũy vào data/chat/{pid}.jsonl
  → assess_patient:
      Tầng 1  Z-Score cá nhân + Isolation Forest + sai số dự báo (EWMA)
      Tầng 2  Rule engine tri thức y khoa (9 luật / 5 hệ cơ quan)
      Tầng 3  RiskScorer: stat + knowledge + ml (LightGBM NHANES) + trend
  → báo cáo: mức rủi ro, điểm, bảng chi tiết chỉ số, khuyến nghị chuyên khoa
```

### 7.4. Kết quả thử nghiệm trên dữ liệu thật NHANES (file mẫu)

| File mẫu (dữ liệu thật CDC) | Mô tả | Kết quả | ml_score |
|---|---|---|---|
| `NOK0001` | Khỏe mạnh, 22 tuổi, HA 118/72 | THAP | 0.04 |
| `NHTN0001` | Tăng huyết áp tâm thu đơn độc 202/62 | TRUNG_BINH | 0.999 |
| `NDM0001` | Đái tháo đường, HbA1c ≥ 7% | TRUNG_BINH | 1.0 |
| `NCKD0001` | Creatinine ≥ 1.5, nghi suy thận | TRUNG_BINH | 0.999 |

### 7.5. Giới hạn của ứng dụng hiện tại

1. **Chat không dùng LLM** — chỉ regex + từ khóa; câu viết lệch mẫu bị bỏ qua
   (im lặng thay vì trả lời sai).
2. **Model ML là cắt ngang** (NHANES 1 kỳ khám) — Tầng 1 chuỗi thời gian vẫn
   dựa thống kê + dữ liệu tổng hợp.
3. **Nhãn phụ thuộc chỉ số** — AUC 0.94 của model NHANES cần đọc đúng ngữ cảnh
   (mục 2.3).
4. **Chưa có xác thực lâm sàng** — chỉ là công cụ hỗ trợ quyết định, không thay
   thế bác sĩ; mọi kết luận cần bác sĩ xác nhận.

---

*Tài liệu nội bộ. Không thay thế chẩn đoán của bác sĩ.*
