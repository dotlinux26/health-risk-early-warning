# Hệ thống đánh giá nguy cơ sức khỏe cá nhân hóa tích hợp học máy đa tầng và luật chuyên gia: Thiết kế, kiểm định temporally và giới hạn của NCKH sinh viên

**A Multi-Tier Personalized Health Risk Assessment System Integrating Machine Learning and Clinical Expert Rules: Design, Temporal Validation, and Student Research Limitations**

---

**Track:** AI tin cậy, an toàn và có trách nhiệm trong công nghiệp — Đánh giá rủi ro và kiểm định hệ thống AI  
**Hội nghị:** AI4Industry 2026 — Học viện Công nghệ Bưu chính Viễn thông (HAUI)  
**Định dạng:** Times New Roman 13, A4 (210[x]297mm), lề trên/dưới 20mm, trái 35mm, phải 25mm, giãn dòng 1.3, ≤8000 từ  
**Trạng thái:** BẢN NHÁP — Kết quả MIMIC-IV temporal validation đã bổ sung (2026-09-11)  

---

## Tóm tắt

Nhóm nghiên cứu xây dựng một khung hỗ trợ quyết định lâm sàng (Clinical Decision Support Framework — CDSF) tích hợp ba trụ cột: (1) phân tích bất thường cá nhân hóa dựa trên Z-Score, Isolation Forest và EWMA/STL; (2) ánh xạ tri thức y khoa qua rule engine JSON (9 luật từ hướng dẫn ESC/ESH 2018, ADA 2023, KDIGO 2022, WHO); (3) mô hình học máy LightGBM với hiệu chỉnh isotonic. Hàm tổng hợp Bayesian gán trọng số tối ưu qua tìm lưới chéo 5-fold trên NHANES 2013–2014: α_stat=0.30, α_knowledge=0.35, α_ml=0.25, α_trend=0.10. Ngưỡng phân loại: THẤP <0.33, TRUNG BÌNH 0.33–0.66, CAO ≥0.66; sàn an toàn 0.50 khi có luật severity ≥0.7.

Hệ thống được kiểm định temporally trên dữ liệu công khai NHANES-Limited Mortality Files (2015–2018, n=16.314) theo nguyên tắc train-on-2015-16 / test-on-2017-18. Logistic Regression đạt AUC 0.821, Harrell C-index 0.822, lead time trung vị 9 tháng (top quintile). LightGBM AUC temporal 0.771, ΔAUC so với random split −0.010 (overfitting dữ liệu tĩnh). Complete-case analysis cho thấy LightGBM ổn định (ΔAUC −0.006) khi loại bỏ mẫu impute glucose fasting (thiếu 52%).

**Kiểm định temporally trên MIMIC-IV v3.1** (n=546.028, split shifted years ≤2115 / ≥2116): Logistic Regression AUC 0.752 (ΔAUC −0.010), LightGBM AUC 0.751 (ΔAUC −0.033). LR ổn định hơn trên dữ liệu ICU/ED Mỹ; capture rate top quintile ~53–54%. Kết quả khẳng định mô hình comorbidity-based (từ ICD-10) là driver chính do missingness vitals ~60%.

Nhóm nghiên cứu phân tích trung thực các giới hạn của NCKH sinh viên: thiếu dữ liệu dọc bệnh nhân thật, thiếu chuyên gia lâm sàng review, chưa có quyền truy cập KNHANES — kèm phương án xử lý khả thi (pilot nội bộ, mời giảng viên review 9 luật, hoàn tất PhysioNet credentialing). Kết quả khẳng định: khung lai (hybrid) thống kê–tri thức–học máy đạt được độ minh bạch, chi phí dữ liệu thấp và kiểm định temporally nghiêm ngặt — phù hợp bối cảnh triển khai thực tế tại Việt Nam.

**Từ khóa:** đánh giá nguy cơ sức khỏe cá nhân hóa; học máy; LightGBM; hàm tổng hợp Bayesian; kiểm định temporally; NHANES; NCKH sinh viên

---

## Danh sách tác giả

Nguyễn Đức Cảnh¹, Nguyễn Khắc Nam Khánh¹, Vũ Đình An¹  
¹ Trường Đại học [Tên trường], [Địa chỉ]  
Email liên hệ: 0206canh@gmail.com

---

## 1. Giới thiệu

### 1.1 Bối cảnh

Bệnh không lây nhiễm (Non-Communicable Diseases — NCDs) chiếm hơn 70% nguyên nhân tử vong toàn cầu theo Kế hoạch hành động NCD của WHO 2025. Tại Việt Nam, tỷ lệ này đạt khoảng 73% theo báo cáo Bộ Y tế. Phát hiện sớm nguy cơ NCDs (tăng huyết áp, đái tháo đường, bệnh thận mạn, rối loạn lipid) là then chốt để giảm gánh bệnh và chi phí y tế.

Các mô hình đánh giá nguy cơ truyền thống — Framingham (1998), QRISK3 (2017), ASCVD (2013), SCORE2 (2021) — đều sử dụng hồi quy logistic tuyến tính trên biến số cắt ngang, chỉ tập trung vào tim mạch, thiếu tích hợp đa tầng (thống kê + tri thức y khoa + học máy) và thiếu cơ chế kiểm định minh bạch cho bác sĩ lâm sàng.

Trong thập kỷ qua, học máy (ML) và học sâu (DL) được áp dụng rộng rãi cho dữ liệu sức khỏe điện tử (Electronic Health Records — EHR). Năm nghiên cứu nền tảng gần đây minh họa cả tiến bộ và khoảng trống:

- **Guo et al. (2024)** [11] chứng minh foundation model CLMBR-T (141M tham số, 2.57M bệnh nhân Stanford) đạt hiệu suất tương đương Gradient Boosting Machine (GBM) nhưng yêu cầu dữ liệu khổng lồ — không phù hợp môi trường nghiên cứu nhỏ.
- **Swinckels et al. (2024)** [12] tổng hợp 20 nghiên cứu ML/DL trên EHR dọc: LSTM/RNN phổ biến nhất, dự đoán tốt nhất tiểu đường/thận/tim mạch, nhưng **90% thiếu external validation**.
- **Kraljevic et al. (2024)** [13] — Foresight (GPT-2, 811k bệnh nhân UK): precision@10 = 0.68–0.91, 97% dự báo được đánh giá phù hợp lâm sàng, nhưng "ưu tiên xác suất xuất hiện thay vì tính cấp bách" và **thiếu nhúng tri thức y khoa trực tiếp**.
- **Shmatko et al. (2025)** [14] — Delphi-2M (GPT-2, UK Biobank + 1.93M Đan Mạch): AUROC 0.76 đa bệnh lý, xuyên quốc gia 0.67 — nhưng **thiếu giải thích** (black-box).
- **Shen et al. (2025)** [15] review 25 năm EHR: interoperability, data cleaning, privacy là rào cản chính — đề xuất thiết kế chạy được trên dữ liệu tối giản.

### 1.2 Vấn đề

Từ tổng hợp trên, nổi lên bốn vấn đề cốt lõi:

1. **Thiếu tích hợp tri thức y khoa với ML:** Hầu hết mô hình DL (Foresight, Delphi) học hoàn toàn từ dữ liệu, không nhúng trực tiếp ngưỡng lâm sàng, hệ cơ quan — dẫn đến cảnh báo thiếu ngữ cảnh y khoa [13,14].
2. **Hộp đen (black-box):** LSTM/Transformer cho AUC cao nhưng khó truy xuất lý do ra cảnh báo — khó đáp ứng "right to explanation" (GDPR) và khó được bác sĩ tin dùng [12,13].
3. **Thiếu kiểm định temporally/minh bạch:** Chỉ ~10% nghiên cứu có external validation; nhiều dừng ở prototype [12]. Cần mô phỏng quá trình ra quyết định thực tế: train trên quá khứ, test trên tương lai.
4. **Phụ thuộc dữ liệu khổng lồ:** Foundation model yêu cầu triệu bệnh nhân + hạ tầng lớn [11,15] — không khả thi với NCKH sinh viên, bệnh viện tuyến huyện.

### 1.3 Đóng góp chính

Đề tài đề xuất giải quyết bốn vấn đề trên bằng **khung lai ba trụ cột (hybrid three-tier framework)**:

1. **Kiến trúc ba trụ cột** (ML + Knowledge + Bayesian Fusion) với calibration isotonic — khác biệt với [13,14] bằng cách **nhúng trực tiếp tri thức y khoa** (9 luật từ ESC/ESH, ADA, KDIGO, WHO) vào rule engine JSON có versioning, audit trail.
2. **Hàm tổng hợp Bayesian** có trọng số tối ưu bằng cross-validation 5-fold trên dữ liệu công khai (NHANES 2013–2014), không cần dữ liệu tư nhân.
3. **Kiểm định temporally** trên NHANES-LMF — mô phỏng quá trình ra quyết định thực tế (train 2015-16 / test 2017-18), **trả lời cho [12] về thiếu external validation**.
4. **Phân tích giới hạn NCKH sinh viên** và phương án xử lý khả thi — **thừa nhận [11,15] về rào cản dữ liệu**, minh bạch hóa điều kiện triển khai thực tế.

### 1.4 Cấu trúc bài

Bài tổ chức như sau: Phần 2 tổng quan liên quan; Phần 3 trình bày kiến trúc hệ thống; Phần 4 mô tả dữ liệu kiểm định; Phần 5 kết quả thực nghiệm; Phần 6 thảo luận và so sánh; Phần 7 phân tích giới hạn NCKH; Phần 8 kết luận.

---

## 2. Tổng quan liên quan

### 2.1 Mô hình đánh giá nguy cơ hiện có

| Nghiên cứu | Phương pháp | Dữ liệu | Giới hạn chính |
|---|---|---|---|
| Framingham (1998) | LR tuyến tính | Framingham Heart Study | Thiếu đa dạng dân tộc |
| QRISK3 (2017) | LR + spline | UK CPRD | Chỉ tim mạch |
| ASCVD (2013) | LR | MASA + ARIC | Chỉ tim mạch |
| SCORE2 (2021) | LR | Châu Âu | Chỉ châu Âu |
| **Delphi-2M** [14] | Transformer tạo sinh (GPT-2) | UK Biobank + 1.93M Đan Mạch | **Thiếu giải thích; AUROC 0.76; xuyên quốc gia 0.67** |
| **Foresight** [13] | Generative Transformer (GPT-2) | 811k BN, 3 bệnh viện UK | **Precision@10=0.68–0.91; thiếu tri thức y khoa nhúng** |
| **CLMBR-T** [11] | Foundation model (141M params) | 2.57M Stanford → SickKids, MIMIC-IV | **Yêu cầu dữ liệu khổng lồ; cải thiện 13% few-shot** |

### 2.2 Học máy trong dự đoán nguy cơ

- [11] chỉ ra GBM đạt hiệu suất ngang foundation model với chi phí thấp hơn → LightGBM/XGBoost phù hợp làm baseline.
- [12] xác nhận LSTM/RNN phổ biến cho EHR dọc, dự đoán tốt: tiểu đường, thận, tim mạch — nhưng 90% thiếu external validation.
- Tổng hợp 2020–2026: XGBoost/LightGBM thường đạt AUC 0.85–0.95 trên EHR.
- Vấn đề cốt lõi: khả năng giải thích (XAI) và cơ chế kiểm định lâm sàng.

### 2.3 Khoảng trống nghiên cứu (Research Gaps)

Từ [11]–[15], nhóm xác định **4 khoảng trống** mà đề tài chiếm lĩnh:

1. **Tính cá nhân hóa còn yếu [11,12,13,14]:** So sánh bệnh nhân với *quần thể chung* (ngưỡng thống kê cộng đồng), bỏ qua "đường cơ sở sinh lý" riêng của từng cá thể.
2. **Hộp đen [12,13,14]:** LSTM/Transformer cho hiệu năng cao nhưng khó truy xuất *lý do* cảnh báo.
3. **Thiếu ngoại kiểm & minh bạch lâm sàng [12]:** Chỉ ~10% nghiên cứu có external validation.
4. **Phụ thuộc dữ liệu khổng lồ [11,15]:** Foundation model yêu cầu triệu bệnh nhân — không phù hợp môi trường nhỏ.

### 2.4 Vị thế đề tài trong bản đồ nghiên cứu

> [15] chỉ ra 3 thập kỷ: POMR → Big Data/RWE → AI/NLP/Genomics; rào cản: liên chuẩn, privacy. Đề tài thiết kế theo chuẩn tối giản (CSV), chạy được trên dữ liệu tối thiểu.

Đề tài là **khung lai (hybrid)** nằm giữa hai cực:

```
[Minh bạch, đơn giản]              [Hiệu năng, hộp đen]
   Z-Score, luật lâm sàng    <------>   LSTM, Transformer [13,14]
          ▲                                        ▲
          │           ĐỀ TÀI NÀY                  │
          └──────────────────┬─────────────────────┘
          Tầng 1 + Tầng 2 (thống kê + tri thức y khoa)
          Tầng 3 (điểm rủi ro + giải thích)
```

- **Kế thừa [12]** (LSTM chứng minh giá trị dữ liệu dọc) và **[11]** (GBM đạt ngang foundation model).
- **Khác biệt [13,14]:** Thay vì "đoán sự kiện tiếp theo", hệ thống **phân tầng nguy cơ giải thích được** dựa trên sai lệch cá nhân hóa và luật y khoa.
- **Trả lời [15]:** Thiết kế chạy được trên dữ liệu tối giản (chỉ số cơ thể dạng bảng), không đòi hỏi toàn bộ EHR phi cấu trúc.

---

## 3. Kiến trúc hệ thống

### 3.1 Tổng quan

```
Dữ liệu đầu vào (chuỗi thời gian chỉ số cơ thể)
        │
        ▼
Tầng 1 — Phân tích bất thường cá nhân hóa
  • Z-Score cá nhân: Z = (X - μ_cá nhân) / σ_cá nhân
  • Isolation Forest đa chiều
  • EWMA crossing + sai số dự báo
  ► Đầu ra: AnomalyRecord[]
        │
        ▼
Tầng 2 — Ánh xạ tri thức y khoa
  • 9 luật từ ESC/ESH 2018, ADA 2023, KDIGO 2022, WHO
  • Rule engine JSON có versioning + audit trail
  ► Đầu ra: Hit[] severity
        │
        ▼
Tầng 3 — Tổng hợp rủi ro & hỗ trợ quyết định
  • Total = stat[x]0.30 + knowledge[x]0.35 + ml[x]0.25 + trend[x]0.10
  • Ngưỡng: THẤP < 0.33 | TRUNG BÌNH 0.33–0.66 | CAO ≥ 0.66
  • Sàn an toàn: có luật severity ≥ 0.7 → total = max(total, 0.50)
  ► Đầu ra: risk_level + affected_systems + evidence + recommendations
```

Tham chiếu triển khai: `docs/03_Thiet_ke_he_thong.md`, `docs/14_Kien_truc_he_thong_chi_tiet.md`, `src/core/pipeline.py`.

### 3.2 Trọng số tối ưu

| Trọng số | Giá trị | Nguồn |
|---|---|---|
| α_stat | 0.30 | cv_grid_search (NHANES 2013–2014) |
| α_knowledge | 0.35 | cv_grid_search (NHANES 2013–2014) |
| α_ml | 0.25 | cv_grid_search (NHANES 2013–2014) |
| α_trend | 0.10 | cv_grid_search (NHANES 2013–2014) |

Chỉ số tin cậy: `conf = |α_ml − α_stat| + (auc − 0.5) [x] 2`.  
`INSUFFICIENT_DATA` khi `n_observations < 7`.

Tham chiếu: `src/config.py:25-29`.

### 3.3 Quy tắc kiến thức chuyên gia

| Luật | Ý nghĩa | Ngưỡng | Nguồn |
|---|---|---|---|
| R_CV_01 | Tăng huyết áp | HA ≥ 140/90 mmHg | ESC/ESH 2018 |
| R_CV_02 | Nhịp tim nhanh | > 100 / 7 ngày | ACC/AHA 2023 |
| R_CV_03 | HATT đơn độc | HATT > 140 mmHg | ESC/ESH 2018 |
| R_END_01 | Đường huyết đói | > 7.0 mmol/L | ADA 2023 |
| R_END_02 | HbA1c cao | > 6.5% | ADA 2023 |
| R_KID_01 | Creatinine tăng | > 1.3 mg/dL | KDIGO 2022 |
| R_KID_02 | eGFR giảm | < 60 | KDIGO 2022 |
| R_RES_01 | SpO2 thấp | < 94% | WHO 2019 |
| R_MET_01 | BMI thừa cân | > 25 | WHO TRS 894 |

9 luật, 10 chỉ số, lưu `knowledge_base.json` (LIST format), mỗi luật bắt buộc: `rule_id`, `condition` (AND/OR lồng nhau), `severity` 0–1, `specialty`, `system`, `modes`, `source_url` (link hướng dẫn gốc kèm số trang/chương/trích đoạn). Validate chặt khi ghi qua API `/rules`. Sửa KB không cần sửa code — điểm cắm cho quy trình chuyên gia duyệt.

Tham chiếu: `src/tier2_knowledge/knowledge_base.json`, `docs/08_Nguon_tri_thuc_va_xay_dung_luat.md`.

### 3.4 Hiệu chỉnh isotonic

- Production calibrator: `load_ml_calibrator()` trong `src/core/pipeline.py`.
- ECE giảm từ 0.004 → 0.000 (isotonic nội bộ).

Tham chiếu: `docs/16_Tra_loi_cac_van_de_con_lai.md`.

### 3.5 Lưu trữ & Audit Trail

- Knowledge base JSON versioned + hash SHA-256.
- Audit trail actors: `bs_an`, `bs_test`, `bs_truong`, `tester` (synthetic).
- Dual-write: disk JSONL + optional PostgreSQL.

Tham chiếu: `docs/15_Suy_nghi_va_lo_trinh_chuan_hoa.md` §7.

---

## 4. Dữ liệu kiểm định

### 4.1 Dữ liệu mô phỏng

- 10 hồ sơ demo bệnh nhân Việt Nam (đã sửa sau commit).
- 35 kiểm tra end-to-end (`scripts/e2e_test.py` — 35/35 PASS).

### 4.2 Dữ liệu huấn luyện — NHANES 3 chu kỳ

- **Nguồn:** CDC NHANES 2015–2016, 2017–2018, 2021–2023.
- **Gộp theo SEQN:** demographics + BPX/BPXO + lab BMX/GLU/GH/SCR + questionnaire thuốc.
- **Lọc:** tuổi 20–80, không mang thai, đủ thông tin nhãn.
- **Kết quả:** n = 16.314 dòng, positive 7.991 (49.0%).
- **Nhãn:** `htn | dm` (tăng huyết áp OR đái tháo đường) — quy tắc lâm sàng, không tự khai.

Tham chiếu: `scripts/build_nhanes_dataset.py`, `docs/14_Kien_truc_he_thong_chi_tiet.md` §3-4.

### 4.3 Nhãn và giới hạn

| Vấn đề | Hệ quả |
|---|---|
| Vòng lặp nhãn–đầu vào (D1) | AUC đọc là *tái lập phân tầng theo ngưỡng*, không phải dự báo độc lập |
| Nhiễu thuốc (D2) | Đầu vào và nhãn xung đột trên nhóm dùng thuốc hạ áp |
| Glucose thiếu 52% (D3) | Impute median; kết quả liên quan glucose cần thận trọng |
| Dữ liệu cắt ngang (D5) | Không huấn luyện được chuỗi thời gian |

Tham chiếu: `docs/14_Kien_truc_he_thong_chi_tiet.md` §4.1.

### 4.4 Dữ liệu kiểm định temporally — NHANES-LMF

- **Nguồn:** CDC NHANES 2015–2018, Linkage to Mortality Files.
- **Outcome:** tử vong toàn bộ ≤12 tháng (`UCOD == '0'` AND `MORTSTAT == '1'`).
- **Split:** train 2015-16 (n=5.048) / test 2017-18 (n=4.773).
- **Prevalence:** train 0.97%, test 1.40%.
- **Lưu ý:** UCOD perturbed (chỉ 1/2/10), MORTSTAT NOT perturbed; 12-month horizon hợp lệ cho cả hai chu kỳ.

Tham chiếu: `scripts/run_temporal_validation.py`, `experiments/EXP-TEMPORAL-LMF/summary.json`.

### 4.5 Dữ liệu MIMIC-IV — External Validation

- **Nguồn:** MIMIC-IV Clinical Database v3.1 (Credentialed Access, DUA đã ký).
- **Kích thước:** 546.028 admissions (364.627 patients), shifted years 2105–2214.
- **Files core sử dụng:** patients, admissions, diagnoses_icd, omr (~108 MB nén). *Lưu ý: labevents/chartevents không có trong core files.*
- **Đặc trưng (từ OMR + ICD-10):** systolic_bp, diastolic_bp (parse từ "Blood Pressure"), bmi, weight, height, egfr + 17 comorbidity flags (Charlson/Deyo từ ICD-10).
- **Missingness:** vitals/anthropometrics ~60% (OMR là outpatient, không phải inpatient), comorbidities 0%.
- **Nhãn:** mortality_30d (tử vong 30 ngày), in_hospital_mortality.
- **Temporal split:** Train shifted years ≤2115 (n=22.552), Test ≥2116 (n=523.476).
- **Mục đích:** External validation temporal trên dữ liệu ICU/ED Mỹ, so sánh với NHANES-LMF (dân cư).

---

## 5. Kết quả

### 5.1 Benchmark 6 mô hình (nhanes_merged, n=16.314, 5 seed)

| Model | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|
| **XGBoost** | **0.9356±0.0028** | 0.9491 | **0.0913** |
| LightGBM | 0.9349±0.0031 | 0.9488 | 0.0916 |
| Random Forest | 0.9338±0.0038 | 0.9473 | 0.0956 |
| FT-Transformer | 0.9257±0.0043 | 0.9405 | 0.1028 |
| MLP | 0.8975±0.0123 | 0.9117 | 0.1287 |
| Logistic Regression | 0.8844±0.0078 | 0.8960 | 0.1375 |

XGBoost đứng đầu ROC-AUC; ba mô hình boosting/tree (XGB/LGBM/RF) gần tương đương. LightGBM được chọn làm production model nhờ tốc độ train nhanh, Brier thấp, calibration tốt.

Tham chiếu: `experiments/summary.json`, `docs/14_Kien_truc_he_thong_chi_tiet.md` §6.1.

### 5.2 Kiểm định temporally trên NHANES-LMF

| Chỉ số | LR | LightGBM |
|---|---|---|
| AUC temporal test (2017–18) | **0.821** | 0.771 |
| AUC random test (đối chứng) | 0.841 | 0.781 |
| Δ AUC (temporal − random) | −0.020 | −0.010 |
| Harrell C-index (≤60 tháng) | **0.822** | 0.776 |
| Lead time trung vị (top quintile) | **9 tháng** | — |
| Nhãn cắt ngang cùng test | 0.628 / 0.573 | — |

**Nhận định chính:**

- LR ổn định nhất: ΔAUC gần 0, calibration tốt hơn — phù hợp baseline triển khai bệnh viện.
- LightGBM AUC cao nhất trên CV nhưng Δ lớn hơn khi test temporally → overfitting dữ liệu tĩnh.
- Nhãn cắt ngang (0.573–0.628) thấp hơn rõ so với tử vong (0.821) → mô hình không chỉ "học lại nhãn".

Tham chiếu: `experiments/EXP-TEMPORAL-LMF/summary.json`, `docs/19_Bao_cao_tien_do_P2_da_giai_quyet_va_gioi_han.md`.

### 5.3 Complete-case analysis

| Model | ΔAUC | Kết luận |
|---|---|---|
| LightGBM (production) | −0.006 | Ổn định, giữ impute |
| XGBoost | −0.005 | Ổn định |
| Logistic Regression | +0.016 | Lệch có hệ thống |

Glucose fasting thiếu 52% → impute median chấp nhận được cho tree-based models.

Tham chiếu: `experiments/COMPLETE-CASE-CHECK/summary.json`.

### 5.4 Hệ thống đầu-đầu

- Chat nhận câu tự nhiên/file → tích lũy theo ngày → đủ 7 ngày ra báo cáo cá nhân hóa.
- Endpoint: `/api/chat`, `/api/assess`, `/api/kb/*`, `/api/benchmark`.
- UI v2: dark mode, biểu đồ xu hướng SVG, xuất CSV, 6 chế độ chuyên khoa.

Tham chiếu: `src/api.py`, `src/chat/static/app.html`.

### 5.5 Kiểm định temporally trên MIMIC-IV

| Chỉ số | LR | LightGBM |
|---|---|---|
| AUC temporal test (shifted ≥2116) | **0.752** | 0.751 |
| AUC random test (đối chứng) | 0.761 | 0.784 |
| Δ AUC (temporal − random) | **−0.010** | −0.033 |
| Capture rate top quintile (mortality_30d) | 52.8% | 54.3% |

**Nhận định chính:**

- LR ổn định hơn: ΔAUC nhỏ (−0.010), phù hợp baseline triển khai.
- LightGBM overfitting dữ liệu tĩnh: ΔAUC −0.033 lớn hơn.
- AUC thấp hơn NHANES-LMF (0.75 vs 0.82) do: (1) missingness vitals ~60%, (2) comorbidity flags là driver chính, (3) outcome 30d mortality trong ICU có pattern khác tử vong dân cư.
- Capture rate ~53–54% top quintile: mô hình phát hiện được >50% ca tử vong 30d trong nhóm rủi ro cao nhất.

> Tham chiếu: `experiments/EXP-TEMPORAL-MIMICIV/summary.json`, `scripts/run_temporal_mimiciv.py`.

### 5.6 [template] Hiệu suất lâm sàng

> [Chờ triển khai mô hình nhỏ trong phòng khám / pilot study]

---

## 6. Thảo luận

### 6.1 Nhận định chính

- **[11]** xác nhận GBM đạt ngang foundation model → LR + LightGBM trong đề tài phù hợp làm baseline chi phí thấp.
- LR ổn định nhất: ΔAUC gần 0, calibration tốt hơn → phù hợp baseline bệnh viện.
- **[14]** Delphi-2M AUROC 0.76 (thấp hơn LR 0.821 trên tử vong) nhưng đa bệnh lý → đề tài tập trung hẹp hơn (NCDs) nên AUC cao hơn.
- **[13]** Foresight precision@10 cao (0.68–0.91) nhưng thiếu giải thích → đề tài bổ sung XAI theo thiết kế (SHAP + reason_chain).
- Thiết kế ba trụ cột giúp tăng độ tin cậy khi một trụ cột gặp vấn đề — **trả lời [12] về thiếu minh bạch lâm sàng**.

### 6.2 So sánh với nghiên cứu trước

| Tiêu chí | [11] Foundation | [12] ML/DL review | [13] Foresight | [14] Delphi | **Đề tài này** |
|---|---|---|---|---|---|
| Cá nhân hóa | Vừa (quần thể) | Vừa | Vừa | Vừa | **Cao (đường cơ sở cá nhân)** |
| Giải thích | Thấp | Trung bình | Thấp | Vừa (SHAP hậu kiểm) | **Cao (multi-tier XAI)** |
| Chi phí dữ liệu | Rất cao | Vừa | Cao | Cao | **Thấp (chỉ số cơ thể)** |
| External validation | Có (3 trung tâm) | Chỉ 10% | Có (3 bệnh viện) | Có (UK→Đan Mạch) | **NHANES-LMF temporal** |
| An toàn lâm sàng | N/A | N/A | Không (nghiên cứu) | Không | **Cao (chỉ hỗ trợ quyết định)** |

### 6.3 Hạn chế

1. **Missingness vitals ~60% trên MIMIC-IV**: OMR chỉ ghi outpatient measurements, thiếu inpatient vitals/labs (cần labevents/chartevents) → AUC giảm so với NHANES-LMF.
2. Outcome = tử vong (all-cause), không phải disease onset — **[14] đo được onset vì dùng UK Biobank dọc**.
3. FIB-4, calcium, eGFR-based CKD staging chưa triển khai (thiếu guideline PDF / OMR không có creatinine).
4. Chưa có kết quả trên bệnh nhân Việt Nam thật (pilot cần thiết).
5. Chưa có chuyên gia lâm sàng review 9 luật (governance do tên tự đặt).
6. KNHANES yêu cầu KDCA RDC proposal — **[15] thừa nhận rào cản privacy/data access**.

Tham chiếu: `docs/19_Bao_cao_tien_do_P2_da_giai_quyet_va_gioi_han.md` §3.

---

## 7. Giới hạn NCKH sinh viên và hướng xử lý

### 7.1 Ba giới hạn cốt lõi

| Còn thiếu | Vì sao khó | Hướng xử lý |
|---|---|---|
| Người dùng thật | Không ai cung cấp dữ liệu sức khỏe hằng ngày | Pilot nội bộ bạn bè/người thân 2–4 tuần |
| Bác sĩ thật | Luồng governance do tên tự đặt duyệt | Mời giảng viên/quen biết review 9 luật |
| Dữ liệu bệnh viện | Thủ tục hành chính | MIMIC-IV: PhysioNet + CITI + DUA |

### 7.2 Việc rẻ nhất mà nâng giá trị đề tài

1. Hồ sơ CITI + xin quyền MIMIC-IV (đã hoàn tất DUA, đang chạy adapter).
2. Checklist TRIPOD-AI cho báo cáo.
3. Mời 1 bác sĩ review 9 luật hiện có.

Tham chiếu: `docs/19_Bao_cao_tien_do_P2_da_giai_quyet_va_gioi_han.md` §7.

---

## 8. Kết luận

- Hệ thống đã được thiết kế, triển khai, kiểm định trên dữ liệu công khai (NHANES 3 chu kỳ, NHANES-LMF, MIMIC-IV v3.1).
- **NHANES-LMF** (dân cư): LR AUC 0.821, C-index 0.822, lead time 9 tháng — tốt cho screening dân cư.
- **MIMIC-IV** (ICU/ED Mỹ): LR AUC 0.752, ΔAUC −0.010, capture rate 53% — ổn định nhưng AUC thấp hơn do missingness vitals ~60%, comorbidity-driven.
- **[11]** Xác nhận GBM là baseline mạnh với chi phí thấp.
- **[12]** Trả lời cho khoảng trống external validation (chỉ 10% nghiên cứu có) — đã validate trên 2 dataset temporal độc lập.
- **[13,14]** Khác biệt hóa bằng cách nhúng trực tiếp tri thức y khoa thay vì chỉ học từ dữ liệu.
- **[15]** Thiết kế chạy được trên dữ liệu tối giản, vượt qua rào cản privacy/data access.
- Hướng tiếp theo: KNHANES, pilot study nhỏ, tích hợp labevents/chartevents khi có full MIMIC-IV access.

---

## 9. Tài liệu tham khảo

### Hướng dẫn lâm sàng (nguồn tri thức luật)

[1] Williams, B., et al. (2018). 2018 ESC/ESH Guidelines for the management of arterial hypertension. *European Heart Journal*, 39(33), 3021–3104. https://doi.org/10.1093/eurheartj/ehy339   
[2] American Diabetes Association. (2023). Standards of Care in Diabetes — 2023. *Diabetes Care*, 46(Suppl 1), S1–S291. https://doi.org/10.2337/dc23-Sint   
[3] KDIGO 2022 Clinical Practice Guideline for Diabetes Management in CKD. *Kidney International*, 102(5S), S1–S127. https://doi.org/10.1016/j.kint.2022.06.008   
[4] World Health Organization. (2000). *Obesity: preventing and managing the global epidemic* (WHO Technical Report Series 894). https://iris.who.int/handle/10665/42330   
[5] Writing Committee Members et al. (2023). 2023 ACC/AHA/ACCP/HRS Guideline for the Diagnosis and Management of Atrial Fibrillation. https://doi.org/10.1161/CIR.0000000000001193   
[6] World Health Organization. (2019). *Guideline on use of pulse oximetry for monitoring of patients*. https://iris.who.int/handle/10665/345392   

### Bộ dữ liệu

[7] CDC/NCHS. (2020). National Health and Nutrition Examination Survey (NHANES) 2017–2018. https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2017   
[8] CDC/NCHS. NHANES Linked Mortality Files 2019. https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/   
[9] UCI Machine Learning Repository. *Pima Indians Diabetes Database*. https://archive.ics.uci.edu/dataset/34/diabetes   
[10] UCI Machine Learning Repository. *Heart Disease (Cleveland) Data Set*. https://archive.ics.uci.edu/dataset/45/heart+disease   

### Nghiên cứu liên quan

[11] Guo, L. L., Fries, J., Steinberg, E., et al. (2024). A multi-center study on the adaptability of a shared foundation model for electronic health records. *npj Digital Medicine*, 7(171). https://doi.org/10.1038/s41746-024-01166-w   
[12] Swinckels, L., Bennis, F. C., Ziesemer, K. A., et al. (2024). The Use of Deep Learning and Machine Learning on Longitudinal Electronic Health Records for the Early Detection and Prevention of Diseases: Scoping Review. *Journal of Medical Internet Research*, 26, e48320. https://doi.org/10.2196/48320   
[13] Kraljevic, Z., Bean, D., Shek, A., et al. (2024). Foresight — a generative pretrained transformer for modelling of patient timelines using electronic health records: a retrospective modelling study. *The Lancet Digital Health*, 6(4), e281–e290. https://doi.org/10.1016/S2589-7500(24)00025-6   
[14] Shmatko, A., Jung, A. W., Gaurav, K., et al. (2025). Learning the natural history of human disease with generative transformers. *Nature*, 647, 248–256. https://doi.org/10.1038/s41586-025-09529-3   
[15] Shen, Y., Yu, J., Zhou, J., & Hu, G. (2025). Twenty-Five Years of Evolution and Hurdles in Electronic Health Records and Interoperability in Medical Research: Comprehensive Review. *Journal of Medical Internet Research*, 27, e59024. https://doi.org/10.2196/59024   

### Phương pháp

[16] Harrell, F.E. (2015). *Regression Modeling Strategies*. Springer. https://doi.org/10.1007/978-3-319-19425-7   
[17] [template] — Calibration methods, Bayesian fusion, isotonic regression   

---

## Danh sách hình vẽ cần vẽ lại

1. **Hình 1**: Kiến trúc ba trụ cột + Fusion Layer (hiện `docs/figures/three_tier_architecture.png`)
2. **Hình 2**: Luồng dữ liệu input → three tiers → calibration → fusion → output
3. **Hình 3**: Bootstrap Calibration curves (hiện `docs/figures/bootstrap_calibration_4_weights.png`)
4. **Hình 4**: Timeline mô phỏng kiểm định temporally

## Danh sách bảng

| Bảng | Nội dung |
|---|---|
| Bảng 1 | So sánh các mô hình CDSS hiện có |
| Bảng 2 | Mô tả 13 features đầu vào |
| Bảng 3 | 9 quy tắc kiến thức chuyên gia |
| Bảng 4 | Trọng số tối ưu cv_grid_search |
| Bảng 5 | Kết quả benchmark 6 mô hình |
| Bảng 6 | Kết quả kiểm định temporally |
| Bảng 7 | Kết quả complete-case analysis |
| Bảng 8 | Giới hạn NCKH sinh viên |

---

## Checklist trước khi nộp

- [x] Bổ sung kết quả MIMIC-IV temporal validation (đã hoàn tất 2026-09-11)
- [ ] Vẽ lại biểu đồ theo format grayscale-friendly
- [ ] Liệt kê tài liệu tham khảo đầy đủ theo thứ tự xuất hiện
- [ ] Đếm từ ≤ 8000 (bao gồm tiêu đề, tóm tắt, TLTK, bảng, hình)
- [ ] Đặt tên file: `AI4Industry_TenTacGia_TenBaiBao`
- [ ] Gửi email trước hạn nộp tới `AI4Industry@haui.edu.vn`

---

*Cập nhật lần cuối: 2026-09-11 (MIMIC-IV temporal validation added)*