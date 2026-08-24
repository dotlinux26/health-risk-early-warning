# 06. Báo cáo huấn luyện mô hình: Kết quả, Giới hạn và Hướng chuẩn mực hóa

> Trạng thái: tháng 08/2026. Tài liệu này tóm tắt kết quả huấn luyện hiện tại,
> thẳng thắn nêu các giới hạn của quy trình và đề xuất hướng đi để mô hình
> thông minh hơn, tin cậy hơn và có thể công bố.

> **Định vị:** các chỉ số AUC/AUPRC dưới đây là của **thành phần mô hình ML
> (LightGBM, Tầng 3)** trong khung CDSF — không phải "độ chính xác" của toàn hệ
> thống, và không dùng để tuyên bố mô hình "chính xác hơn" các hệ thống tạo sinh
> như Delphi. Đề tài đóng góp ở cấp khung. Xem thêm
> [docs/07](07_Dinh_vi_de_tai.md).

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

**Model đang chạy trong hệ thống được train trên dữ liệu thật NHANES gộp 3 chu kỳ**
(2015–2016, 2017–2018, 2021–2023; CDC/NCHS, khảo sát lâm sàng toàn dân Mỹ):
16 314 người trưởng thành, 48.98% có tăng huyết áp hoặc đái tháo đường.

Đặc trưng dùng TRÙNG schema hệ thống: systolic_bp, diastolic_bp, heart_rate,
glucose_fasting, hba1c, creatinine, bmi. Nhãn: tăng huyết áp (HA ≥ 140/90 hoặc
đang dùng thuốc) HOẶC đái tháo đường (HbA1c ≥ 6.5% hoặc đường huyết lúc đói ≥
7.0 mmol/L).

| Tham số | Giá trị |
|---|---|
| N | 16 314 (dương tính 7991) |
| AUC (5-fold CV, seed 42) | **0.9356 ± 0.0016** |
| AUPRC (5-fold CV) | **0.9494** |
| Đặc trưng quan trọng nhất | hba1c, systolic_bp, diastolic_bp, creatinine |
| Model sản xuất | `data/models/risk_lgbm_real.joblib` |

> Cảnh báo khoa học: nhãn "tăng huyết áp" được định nghĩa một phần bằng chính
> huyết áp đo được (cùng dấu hiệu dùng làm đặc trưng), nên AUC cao (~0.94) là
> kỳ vọng và KHÔNG phải là độ chính xác chẩn đoán trên lâm sàng. Đã lượng hóa
> mức độ vòng lặp nhãn này ở `experiments/LABEL-SENSITIVITY/` (K2) và chứng minh
> hướng khắc phục bằng outcome tử vong thật (docs/18).

### 2.4. Hiệu chỉnh xác suất trong sản xuất (bổ sung 23/08/2026)

Xác suất của model sản xuất được **hiệu chỉnh isotonic** trước khi vào điểm rủi
ro (`load_ml_calibrator` trong `src/core/pipeline.py`, calibrator từ
`experiments/EXP-ML-LGBM-42/`). Kết quả trên test:

| Model | ECE trước | ECE sau | Brier trước | Brier sau |
|---|---|---|---|---|
| LightGBM | 2.24% | 1.62% | 0.1944 | 0.1938 |
| Random Forest | 4.46% | 1.69% | — | — |
| Logistic Regression | 5.12% | 1.84% | — | — |

Chi tiết đầy đủ (30 evidence package, 6 kiến trúc): `experiments/summary.md`
và trang `/benchmark`.

---

## 3. Giới hạn của quy trình huấn luyện hiện tại

### 3.1. Giới hạn về dữ liệu

| Giới hạn | Hệ quả | Trạng thái 24/08 |
|---|---|---|
| Dữ liệu tổng hợp không phải dữ liệu lâm sàng | AUC 0.987 gây ảo tưởng; mô hình học đúng "quy luật sinh" chứ không phải sinh lý thực | Đã thay bằng NHANES thật làm sản xuất (§2.3) |
| Dataset thật là **cắt ngang** (một dòng/bệnh nhân) | Không kiểm chứng được bài toán cốt lõi: cảnh báo sớm theo chuỗi thời gian | Đã kiểm chứng cấp cohort qua NHANES-LMF (AUC tử vong 0.821, lead time 9 tháng — docs/18); chuỗi theo ngày vẫn chờ |
| Chưa có **dữ liệu dọc có nhãn thật** | Không đánh giá được thời điểm phát hiện sớm so với khởi phát bệnh | LMF cho lead time cấp tháng; P2.4/P2.5 chờ dữ liệu dọc theo ngày |
| Dân số hẹp | Pima là phụ nữ người Mỹ gốc Pima Ấn Độ; Cleveland là dân số tham chiếu nghiên cứu — không đại diện người Việt | Vẫn mở — chỉ dùng làm tham chiếu ngoài |
| Mất cân bằng lớp và định nghĩa nhãn khác nhau | Pima nhãn theo tiêu chuẩn 1988; không tương thích trực tiếp với nhãn "sự kiện" của bài toán hiện tại | Giữ nguyên ghi chú |
| Kích thước mẫu nhỏ | Phương sai ước lượng cao; khó suy diễn ở phân nhóm | NHANES gộp 3 chu kỳ giảm một phần |

### 3.2. Giới hạn về phương pháp

1. **Nhãn tổng hợp không có "ground truth" lâm sàng** — sự kiện trong dữ liệu
   tổng hợp được định nghĩa bằng ngưỡng, không qua xác nhận bác sĩ.
2. ~~**Chưa có chiến lược đánh giá theo thời gian** (temporal split)~~
   **ĐÃ GIẢI QUYẾT 24/08**: `experiments/EXP-TEMPORAL-LMF` — split theo thời gian
   2015-16 → 2017-18, random split chỉ lạc quan hơn 0.01–0.02 AUC.
3. ~~**Chưa chuẩn hoá cách xử lý dữ liệu thiếu**~~ **ĐÃ GIẢI QUYẾT**: cùng
   `SimpleImputer(median)` fit-train xuyên suốt train/inference; đã đối chiếu với
   complete-case (`experiments/COMPLETE-CASE-CHECK/`) — vô hại cho LightGBM.
4. ~~**Chưa hiệu chuẩn (calibration)**~~ **ĐÃ GIẢI QUYẾT 23/08**: isotonic fit
   trên validation, tích hợp vào production pipeline (§2.4); ECE test ≤ 1.7%.
5. **Điểm số tổng hợp (stat/knowledge/ml/trend) dùng trọng số tĩnh** — đã đo độ
   nhạy với 5 bộ trọng số (`experiments/WEIGHT-SENSITIVITY/`, đồng thuận ≥96%)
   nhưng vẫn chưa học từ dữ liệu.
6. **Quy trình phiên bản hoá** dữ liệu, mô hình, số liệu đánh giá — mỗi thí
   nghiệm giờ có evidence package riêng trong `experiments/EXP-*/`; TRIPOD-AI
   đầy đủ chưa có.
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

> **Đã thực hiện một phần 24/08 (docs/18):** tích hợp NHANES Public-Use Linked
> Mortality File — nhãn biến cố tương lai (tử vong, follow-up đến 31/12/2019)
> cho cùng schema SEQN. Temporal validation đầu tiên: AUC 0.821 (split theo
> thời gian), lead time trung vị 9 tháng. Chuỗi chỉ số theo ngày cho từng cá
> nhân vẫn chờ kênh nhập hệ thống (P2.4).

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
| G2 ✅ | Huấn luyện LightGBM trên NHANES với 5-fold CV, median-fill khi thiếu chỉ số | `data/models/risk_lgbm_real.joblib` — đã xong, đang là model sản xuất (nay train trên 3 chu kỳ gộp) |
| G3 ◐ | Ghép nhiều kỳ NHANES thành chuỗi thời gian theo từng người; **đã làm được ở cấp cohort qua Linked Mortality File** (`data/datasets/nhanes_mortality.csv`, 10 065 người có outcome tử vong — docs/18); chuỗi theo ngày từng người chưa có | Dataset longitudinal + outcome biến cố tương lai |
| G4 | Tích hợp Chronos/TimesFM vào Tầng 1, benchmark lỗi dự báo vs Z-Score | Báo cáo so sánh 2 phương pháp |
| G5 ◐ | ~~calibration~~ ✅ đã làm (`EXP-ML-*/calibration.json` + production isotonic); trọng số học từ dữ liệu và TRIPOD-AI còn mở | Mô hình chuẩn hoá, tái lập được |

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

> **Cập nhật 23/08/2026 (P1):** giao diện chat regex đã được **thay thế bằng
> ứng dụng form-based** tại `http://127.0.0.1:8000/` — nhập bản ghi theo bảng,
> quản trị luật full-screen kèm audit trail, trang nghiên cứu `/benchmark`.
> Ảnh chụp chat dưới đây giữ làm tư liệu lịch sử. Chi tiết trải nghiệm thực tế:
> docs/17.

### 7.1. Giao diện

| Hộp thoại chat (cũ) | Kết quả đánh giá |
|---|---|
| ![Giao diện chat](screenshots/giao_dien_chat.png) | ![Kết quả đánh giá](screenshots/ket_qua_danh_gia.png) |

Giao diện nền sáng, bo góc 2px, thanh xổ trên cùng liệt kê các mã bệnh nhân đã
có dữ liệu trong `data/chat/` (từ API `/api/chat/patients`); vẫn nhập được mã
mới bằng ô kế bên.

### 7.2. Cách dùng (hiện tại)

1. Khởi động: `./run_api.sh start`, mở **http://127.0.0.1:8000/**.
2. Tab "Đánh giá": chọn mã bệnh nhân → đánh giá 4 tầng, evidence đầy đủ
   (luật kích hoạt kèm tên tiếng Việt + nguồn tham chiếu, bảng chỉ số với
   z-score/xu hướng, panel mô tả ML với cặp điểm Raw vs Calibrated).
3. Tab "Bản ghi": bảng theo ngày, bộ chọn cột trong 10 chỉ số hệ thống,
   thêm ngày/xóa ngày trực tiếp.
4. Tab "Luật & Quản trị": thêm/sửa luật, chuyển trạng thái
   draft→review→approved→active, xem audit trail.
5. Trang `/benchmark`: kết quả benchmark + nghiên cứu độ bền vững
   (K2–K4, calibration, temporal validation).

Cách dùng cũ qua chat (regex): tin nhắn như `Huyết áp 135/85, nhịp tim 80`
hoặc đính kèm PDF/DOCX (`data/sample_nhanes/NHTN0001.pdf`); lệnh `trạng thái`
· `báo cáo` · `xóa dữ liệu`.

### 7.3. Luồng xử lý một đánh giá

```
File / form nhập → ingest (regex PDF/DOCX) hoặc API records
  → tích lũy vào data/chat/{pid}.jsonl
  → assess_patient:
      Tầng 1  Z-Score cá nhân + Isolation Forest + sai số dự báo (EWMA)
      Tầng 2  Rule engine tri thức y khoa (9 luật active / 5 hệ cơ quan,
              có governance draft→review→approved→active + audit trail)
      Tầng 3  RiskScorer: stat + knowledge + ml (LightGBM NHANES 3 chu kỳ,
              isotonic-calibrated) + trend
  → báo cáo trình bày theo thứ tự:
      1) Luật lâm sàng kích hoạt — kèm link tham chiếu nguồn (source_url)
      2) Bảng chỉ số theo dõi (giá trị, đường cơ sở, xu hướng, Z-Score, phạm vi)
      3) Hỗ trợ mô hình ML — kèm ghi chú "suy luận bổ sung, không phải chẩn đoán"
         và bảng Raw vs Calibrated score
```

### 7.4. Kết quả thử nghiệm trên dữ liệu thật (cập nhật 24/08/2026)

Các ca đo lại trực tiếp qua `POST /api/assess` sau khi tích hợp hiệu chỉnh
isotonic (điểm `ml` là xác suất SAU hiệu chỉnh — isotonic là hàm bậc thang nên
vùng điểm cao dồn về 1.0):

| Bệnh nhân | Mô tả | Mức nguy cơ | Điểm tổng | ml (calibrated) |
|---|---|---|---|---|
| `P001` | Nhóm khỏe mạnh (seeded từ sample_long) | THẤP | 0.067 | 0.2675 |
| `P002–P005` | Nhóm bệnh mãn tính nhẹ → vừa | THẤP/TRUNG BÌNH | 0.25–0.55 | 0.64–1.0 |
| `NHTN0001` | Tăng huyết áp tâm thu đơn độc 202/62 (file mẫu NHANES) | TRUNG BÌNH | 0.500 | 1.0 |
| `DEMO_HYPERTENSIVE` | Ca tổng hợp drift HA + spike 7 ngày cuối (z=+2.37σ) | CAO | 0.841 | 1.0 |
| `DEMO_DIABETIC` | Ca tổng hợp glucose/HbA1c tăng dần | CAO | 0.748 | 1.0 |

Tái lập: `python3 scripts/seed_demo_data.py --force` rồi gọi API assess từng
bệnh nhân. Kịch bản sử dụng thực tế đầy đủ xem docs/17.

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
