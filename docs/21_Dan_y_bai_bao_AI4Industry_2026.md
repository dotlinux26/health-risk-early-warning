# 21. Dàn ý bài báo

> **Track**: AI tin cậy, an toàn và có trách nhiệm trong công nghiệp  
> **Format**: Times New Roman 13, A4, ≤8000 chữ, giãn dòng 1.3  
> **Trạng thái**: BẢN NHÁP — `[template]` cho phần chưa có kết quả

---

## Tiêu đề

**Hệ thống đánh giá nguy cơ sức khỏe cá nhân hóa tích hợp học máy đa tầng và luật chuyên gia: Thiết kế, kiểm định temporally và giới hạn của NCKH sinh viên**

> *A Multi-Tier Personalized Health Risk Assessment System Integrating Machine Learning and Clinical Expert Rules: Design, Temporal Validation, and Student Research Limitations*

---

## Tóm tắt (~250 chữ)

[template] — Mẫu khung:

*Nhóm nghiên cứu xây dựng một khung hỗ trợ quyết định lâm sàng (CDSF) tích hợp ba trụ cột: (1) phân tích bất thường cá nhân hóa (Z-Score, Isolation Forest, EWMA), (2) luật tri thức y khoa từ hướng dẫn ESC/ESH 2018, ADA 2023, KDIGO 2022, WHO, và (3) mô hình học máy LightGBM với hiệu chỉnh isotonic. Hàm tổng hợp Bayesian gán trọng số stat=0.30, knowledge=0.35, ml=0.25, trend=0.10. Hệ thống được kiểm định temporally trên dữ liệu công khai NHANES-Limited Mortality Files (2015–2018, n=16.314) trên nguyên tắc train-on-2015-16 / test-on-2017-18, đạt AUC 0,821 (Logistic Regression) và Harrell C-index 0,822. Nhóm nghiên cứu phân tích giới hạn của NCKH sinh viên trong bối cảnh thiếu dữ liệu dọc, thiếu chuyên gia lâm sàng review, và chưa có quyền truy cập MIMIC-IV/KNHANES, kèm phương án xử lý khả thi.*

**Từ khóa**: đánh giá nguy cơ sức khỏe cá nhân hóa; học máy; LightGBM; hàm tổng hợp Bayesian; kiểm định temporally; NHANES; NCKH sinh viên

---

## Danh sách tác giả

[template]

---

## 1. Giới thiệu (~1000 chữ)

### 1.1 Bối cảnh

- Bệnh không lây nhiễm (NCDs) chiếm >70% nguyên nhân tử vong toàn cầu (WHO NCD Action Plan 2025)
- Việt Nam: NCDs chiếm ~73% tử vong; phát hiện sớm nguy cơ là then chốt
- Các mô hình hiện có (Framingham, QRISK3, ASCVD, SCORE2) chỉ dùng một phương pháp, thiếu tích hợp đa tầng
- **[11]** Guo et al. (2024) chứng minh foundation model cho EHR đạt hiệu suất tương đương GBM nhưng yêu cầu dữ liệu khổng lồ (2.57M bệnh nhân) → không phù hợp môi trường nghiên cứu nhỏ
- **[12]** Swinckels et al. (2024) tổng hợp 20 nghiên cứu: LSTM/RNN phổ biến nhất cho EHR dọc nhưng 90% thiếu external validation
- **[13]** Kraljevic et al. (2024) — Foresight GPT-2: precision@10 = 0.68–0.91 nhưng "ưu tiên xác suất xuất hiện thay vì tính cấp bách" → thiếu tri thức y khoa nhúng trực tiếp
- **[14]** Delphi-2M (2025): AUROC 0.76 đa bệnh lý, xuyên quốc gia 0.67 — nhưng thiếu giải thích
- **[15]** EHR 25 năm (2025): interoperability, data cleaning, privacy là rào cản chính

### 1.2 Vấn đề

- Học máy trong y tế: AUC cao nhưng thiếu cơ chế kiểm định minh bạch cho bác sĩ **[12,13]**
- Thiếu tích hợp luật chuyên gia (knowledge-based) với ML — Foresight bị chê vì thiếu nhúng tri thức y khoa trực tiếp **[13]**
- Hầu hết nghiên cứu dùng dữ liệu cắt ngang, chưa validate temporally trên cùng quần thể — chỉ 10% có external validation **[12]**
- Foundation model yêu cầu dữ liệu triệu bệnh nhân **[11]** — không khả thi với NCKH sinh viên

### 1.3 Đóng góp chính

1. Kiến trúc ba trụ cột (ML + Knowledge + Bayesian Fusion) với calibration isotonic — **khác biệt với [13,14] bằng cách nhúng trực tiếp tri thức y khoa**
2. Hàm tổng hợp Bayesian có trọng số tối ưu bằng CV 5-fold
3. Kiểm định temporally trên NHANES-LMF — mô phỏng quá trình ra quyết định thực tế, **trả lời cho [12] về thiếu external validation**
4. Phân tích giới hạn NCKH sinh viên và phương án xử lý — **thừa nhận [11,15] về rào cản dữ liệu**

### 1.4 Cấu trúc bài

---

## 2. Tổng quan liên quan (~1000 chữ)

### 2.1 Mô hình đánh giá nguy cơ hiện có

| Nghiên cứu | Phương pháp | Dữ liệu | Giới hạn chính |
|---|---|---|---|
| Framingham (1998) | LR tuyến tính | Framingham Heart Study | Thiếu đa dạng dân tộc |
| QRISK3 (2017) | LR + spline | UK CPRD | Chỉ tim mạch |
| ASCVD (2013) | LR | MASA + ARIC | Chỉ tim mạch |
| SCORE2 (2021) | LR | Châu Âu | Chỉ châu Âu |
| **[14] Delphi-2M** | Transformer tạo sinh (GPT-2, ~2.2M tham số) | UK Biobank + 1.93M Đan Mạch | **Thiếu giải thích; AUROC 0.76; xuyên quốc gia 0.67** |
| **[13] Foresight** | Generative Transformer (GPT-2) | 811.336 BN, 3 bệnh viện (UK) | **Precision@10 = 0.68–0.91; 97% phù hợp lâm sàng nhưng ưu tiên xác suất xuất hiện thay vì tính cấp bách** |
| **[11] CLMBR-T** | Foundation model (141M tham số) | 2.57M Stanford → SickKids, MIMIC-IV | **Yêu cầu dữ liệu khổng lồ; cải thiện 13% few-shot** |

### 2.2 Học máy trong dự đoán nguy cơ

- **[11]** GBM đạt hiệu suất tương đương foundation model với chi phí thấp hơn → nên dùng LightGBM/XGBoost làm baseline
- **[12]** LSTM/RNN phổ biến nhất cho EHR dọc, dự đoán tốt nhất: tiểu đường, thận, tim mạch → nhưng 90% thiếu external validation
- Tổng hợp 2020–2026: XGBoost/LightGBM thường đạt AUC 0.85–0.95 trên dữ liệu EHR
- Vấn đề: khả năng giải thích (XAI) và cơ chế kiểm định lâm sàng

### 2.3 Khoảng trống nghiên cứu (Research Gaps)

Từ tổng hợp [11]–[15], nổi lên **4 khoảng trống** mà đề tài đề xuất có thể chiếm lĩnh:

1. **Tính cá nhân hóa còn yếu [11,12,13,14]:** Hầu hết mô hình so sánh bệnh nhân với *quần thể chung*, bỏ qua "đường cơ sở sinh lý" riêng của từng cá thể
2. **Hộp đen [12,13,14]:** LSTM/RNN và transformer cho hiệu năng cao nhưng khó truy xuất *lý do* ra cảnh báo → khó được bác sĩ tin dùng
3. **Thiếu ngoại kiểm & minh bạch lâm sàng [12]:** Chỉ ~10% nghiên cứu có external validation; nhiều mô hình dừng ở prototype
4. **Phụ thuộc dữ liệu khổng lồ [11,15]:** Foundation model yêu cầu triệu bệnh nhân + hạ tầng lớn — không phù hợp môi trường nhỏ

### 2.4 Vị thế đề tài trong bản đồ nghiên cứu

> **[15]** EHR 25 năm: 3 thập kỷ POMR → Big Data/RWE → AI/NLP/Genomics; rào cản: liên chuẩn, privacy → đề tài thiết kế theo chuẩn tối giản (CSV), chạy được trên dữ liệu tối thiểu

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

- **Kế thừa [12]** (LSTM chứng minh giá trị dữ liệu dọc) và **[11]** (GBM đạt ngang foundation model)
- **Khác biệt [13,14]**: thay vì "đoán sự kiện tiếp theo", hệ thống **phân tầng nguy cơ giải thích được** dựa trên sai lệch cá nhân hóa và luật y khoa
- **Trả lời [15]**: thiết kế chạy được trên dữ liệu tối giản (chỉ số cơ thể dạng bảng), không đòi hỏi toàn bộ EHR

---

## 3. Kiến trúc hệ thống (~1500 chữ)

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
  • Total = stat×0.30 + knowledge×0.35 + ml×0.25 + trend×0.10
  • Ngưỡng: THẤP < 0.33 | TRUNG BÌNH 0.33–0.66 | CAO ≥ 0.66
  • Sàn an toàn: có luật severity ≥ 0.7 → total = max(total, 0.50)
  ► Đầu ra: risk_level + affected_systems + evidence + recommendations
```

> **Tham chiếu**: `docs/03_Thiet_ke_he_thong.md`, `docs/14_Kien_truc_he_thong_chi_tiet.md`

### 3.2 Trọng số tối ưu

| Trọng số | Giá trị | Nguồn |
|---|---|---|
| α_stat | 0.30 | cv_grid_search (NHANES 2013–2014) |
| α_knowledge | 0.35 | cv_grid_search (NHANES 2013–2014) |
| α_ml | 0.25 | cv_grid_search (NHANES 2013–2014) |
| α_trend | 0.10 | cv_grid_search (NHANES 2013–2014) |

- Chỉ số自信 `conf = |α_ml − α_stat| + (auc − 0.5) × 2`
- `INSUFFICIENT_DATA` khi `n_observations < 7`

> **Tham chiếu**: `src/config.py:25-29`

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

> **Tham chiếu**: `src/tier2_knowledge/knowledge_base.json`, `docs/08_Nguon_tri_thuc_va_xay_dung_luat.md`

### 3.4 Hiệu chỉnh isotonic

- Production calibrator: `load_ml_calibrator()` trong `src/core/pipeline.py`
- ECE 0.004 → 0.000 (isotonic nội bộ)

> **Tham chiếu**: `docs/16_Tra_loi_cac_van_de_con_lai.md`

### 3.5 Lưu trữ & Audit Trail

- Knowledge base JSON versioned + hash SHA-256
- Audit trail actors: `bs_an`, `bs_test`, `bs_truong`, `tester` (synthetic)
- Dual-write: disk JSONL + optional PostgreSQL

> **Tham chiếu**: `docs/15_Suy_nghi_va_lo_trinh_chuan_hoa.md` §7

---

## 4. Dữ liệu kiểm định (~1000 chữ)

### 4.1 Dữ liệu mô phỏng

- 10 hồ sơ demo bệnh nhân Việt Nam (đã fix伪sau commit)
- 35 kiểm tra e2e (`scripts/e2e_test.py` — 35/35 PASS)

### 4.2 Dữ liệu huấn luyện — NHANES 3 chu kỳ

- **Nguồn**: CDC NHANES 2015–2016, 2017–2018, 2021–2023
- **Gộp theo SEQN**: demographics + BPX/BPXO + lab BMX/GLU/GH/SCR + questionnaire thuốc
- **Lọc**: tuổi 20–80, không mang thai, đủ thông tin nhãn
- **Kết quả**: n = 16.314 dòng, positive 7.991 (49.0%)
- **Nhãn**: `htn | dm` (tăng huyết áp OR đái tháo đường) — quy tắc lâm sàng, không tự khai

> **Tham chiếu**: `scripts/build_nhanes_dataset.py`, `docs/14_Kien_truc_he_thong_chi_tiet.md` §3-4

### 4.3 Nhãn và giới hạn

| Vấn đề | Hệ quả |
|---|---|
| Vòng lặp nhãn–đầu vào (D1) | AUC đọc là *tái lập phân tầng theo ngưỡng*, không phải dự báo độc lập |
| Nhiễu thuốc (D2) | Đầu vào và nhãn xung đột trên nhóm dùng thuốc hạ áp |
| Glucose thiếu 52% (D3) | Impute median; kết quả liên quan glucose cần thận trọng |
| Dữ liệu cắt ngang (D5) | Không huấn luyện được chuỗi thời gian |

> **Tham chiếu**: `docs/14_Kien_truc_he_thong_chi_tiet.md` §4.1

### 4.4 Dữ liệu kiểm định temporally — NHANES-LMF

- **Nguồn**: CDC NHANES 2015–2018, Linkage to Mortality Files
- **Outcome**: tử vong toàn bộ ≤12 tháng (`UCOD == '0'` AND `MORTSTAT == '1'`)
- **Split**: train 2015-16 (n=5.048) / test 2017-18 (n=4.773)
- **Prevalence**: train 0.97%, test 1.40%

> **Tham chiếu**: `scripts/run_temporal_validation.py`, `experiments/EXP-TEMPORAL-LMF/summary.json`

---

## 5. Kết quả (~1500 chữ)

### 5.1 Benchmark 6 mô hình (nhanes_merged, n=16.314, 5 seed)

| Model | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|
| XGBoost | **0.9356±0.0028** | 0.9491 | **0.0913** |
| LightGBM | 0.9349±0.0031 | 0.9488 | 0.0916 |
| Random Forest | 0.9338±0.0038 | 0.9473 | 0.0956 |
| FT-Transformer | 0.9257±0.0043 | 0.9405 | 0.1028 |
| MLP | 0.8975±0.0123 | 0.9117 | 0.1287 |
| Logistic Regression | 0.8844±0.0078 | 0.8960 | 0.1375 |

> **Tham chiếu**: `experiments/summary.json`, `docs/14_Kien_truc_he_thong_chi_tiet.md` §6.1

### 5.2 Kiểm định temporally trên NHANES-LMF

| Chỉ số | LR | LightGBM |
|---|---|---|
| AUC temporal test (2017–18) | **0.821** | 0.771 |
| AUC random test (đối chứng) | 0.841 | 0.781 |
| Δ AUC (temporal − random) | −0.020 | −0.010 |
| Harrell C-index (≤60 tháng) | **0.822** | 0.776 |
| Lead time trung vị (top quintile) | **9 tháng** | — |
| Nhãn cắt ngang cùng test | 0.628 / 0.573 | — |

**Nhận định chính**:
- LR ổn định nhất: ΔAUC gần 0, calibration tốt hơn
- LightGBM AUC cao nhất trên CV nhưng Δ lớn hơn khi test temporally → overfitting dữ liệu tĩnh
- Nhãn cắt ngang (0.573–0.628) thấp hơn rõ so với tử vong (0.821) → tín hiệu mô hình không chỉ "học lại nhãn"

> **Tham chiếu**: `experiments/EXP-TEMPORAL-LMF/summary.json`, `docs/19_Bao_cao_tien_do_P2_da_giai_quyet_va_gioi_han.md`

### 5.3 Complete-case analysis

| Model | ΔAUC | Kết luận |
|---|---|---|
| LightGBM (production) | −0.006 | Ổn định, giữ impute |
| XGBoost | −0.005 | Ổn định |
| Logistic Regression | +0.016 | Lệch có hệ thống |

- Glucose fasting thiếu 52% → impute median chấp nhận được cho tree-based models

> **Tham chiếu**: `experiments/COMPLETE-CASE-CHECK/summary.json`

### 5.4 Hệ thống đầu-đầu

- Chat nhận câu tự nhiên/file → tích lũy theo ngày → đủ 7 ngày ra báo cáo cá nhân hóa
- Endpoint: `/api/chat`, `/api/assess`, `/api/kb/*`, `/api/benchmark`
- UI v2: dark mode, biểu đồ xu hướng SVG, xuất CSV, 6 chế độ chuyên khoa

> **Tham chiếu**: `src/api.py`, `src/chat/static/app.html`

### 5.5 [template — chưa có] Kiểm định trên MIMIC-IV

> [Chờ hoàn tất PhysioNet credentialing → tải MIMIC-IV → chạy temporal validation]

### 5.6 [template — chưa có] Hiệu suất lâm sàng

> [Chờ triển khai mô hình nhỏ trong phòng khám]

---

## 6. Thảo luận (~800 chữ)

### 6.1 Nhận định chính

- **[11]** GBM đạt ngang foundation model → LR + LightGBM trong đề tài phù hợp làm baseline với chi phí thấp
- LR ổn định nhất: ΔAUC gần 0, calibration tốt hơn → phù hợp làm baseline trong bệnh viện
- **[14]** Delphi-2M AUROC 0.76 (thấp hơn LR 0.821 trên tử vong) nhưng đa bệnh lý → đề tài tập trung hẹp hơn (NCDs) nên AUC cao hơn
- **[13]** Foresight precision@10 cao (0.68–0.91) nhưng thiếu giải thích → đề tài bổ sung XAI theo thiết kế (SHAP + reason_chain)
- Thiết kế ba trụ cột giúp tăng độ tin cậy khi một trụ cột gặp vấn đề — **trả lời [12] về thiếu minh bạch lâm sàng**

### 6.2 So sánh với nghiên cứu trước

| Tiêu chí | [11] Foundation | [12] ML/DL review | [13] Foresight | [14] Delphi | **Đề tài này** |
|---|---|---|---|---|---|
| Cá nhân hóa | Vừa (quần thể) | Vừa | Vừa | Vừa | **Cao (đường cơ sở cá nhân)** |
| Giải thích | Thấp | Trung bình | Thấp | Vừa (SHAP hậu kiểm) | **Cao (multi-tier XAI)** |
| Chi phí dữ liệu | Rất cao | Vừa | Cao | Cao | **Thấp (chỉ số cơ thể)** |
| External validation | Có (3 trung tâm) | Chỉ 10% | Có (3 bệnh viện) | Có (UK→Đan Mạch) | **NHANES-LMF temporal** |
| An toàn lâm sàng | N/A | N/A | Không (nghiên cứu) | Không | **Cao (chỉ hỗ trợ quyết định)** |

### 6.3 Hạn chế

1. Chỉ dùng NHANES công khai; MIMIC-IV chưa tải; KNHANES yêu cầu KDCA — **[15] cũng thừa nhận rào cản privacy/data access**
2. Outcome = tử vong, không phải thời điểm khởi phát bệnh — **[14] đo được onset vì dùng UK Biobank dọc**
3. FIB-4, calcium chưa triển khai (chưa có guideline PDF)
4. **[template]**: Chưa có kết quả trên bệnh nhân Việt Nam thật
5. Chưa có chuyên gia lâm sàng review/(nếu nhóm quyết định không làm expert review)

> **Tham chiếu**: `docs/19_Bao_cao_tien_do_P2_da_giai_quyet_va_gioi_han.md` §3

---

## 7. Giới hạn NCKH sinh viên và hướng xử lý (~600 chữ)

### 7.1 Ba giới hạn cốt lõi

| Còn thiếu | Vì sao khó | Hướng xử lý |
|---|---|---|
| Người dùng thật | Không ai cung cấp dữ liệu sức khỏe hằng ngày | Pilot nội bộ bạn bè/người thân 2–4 tuần |
| Bác sĩ thật | Luồng governance do tên tự đặt duyệt | Mời giảng viên/quen biết review 9 luật |
| Dữ liệu bệnh viện | Thủ tục hành chính | MIMIC-IV: PhysioNet + CITI + DUA |

### 7.2 Việc rẻ nhất mà nâng giá trị đề tài

1. Hồ sơ CITI + xin quyền MIMIC-IV
2. Checklist TRIPOD-AI
3. Mời 1 bác sĩ review 9 luật hiện có

> **Tham chiếu**: `docs/19_Bao_cao_tien_do_P2_da_giai_quyet_va_gioi_han.md` §7

---

## 8. Kết luận (~400 chữ)

- Hệ thống đã được thiết kế, triển khai, kiểm định trên dữ liệu công khai
- Kết quả temporally trên NHANES-LMF cho thấy LR có tiềm năng (AUC 0.821, C-index 0.822)
- **[11]** Xác nhận GBM là baseline mạnh với chi phí thấp
- **[12]** Trả lời cho khoảng trống external validation (chỉ 10% nghiên cứu có)
- **[13,14]** Khác biệt hóa bằng cách nhúng trực tiếp tri thức y khoa thay vì chỉ học từ dữ liệu
- **[15]** Thiết kế chạy được trên dữ liệu tối giản, vượt qua rào cản privacy/data access
- Hướng tiếp theo: MIMIC-IV, KNHANES, pilot study nhỏ

---

## 9. Tài liệu tham khảo (~60 mục)

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

- [ ] Bổ sung kết quả MIMIC-IV nếu hoàn tất trước hạn nộp
- [ ] Vẽ lại biểu đồ theo format grayscale-friendly
- [ ] Liệt kê tài liệu tham khảo đầy đủ theo thứ tự xuất hiện
- [ ] Đếm chữ ≤ 8000 (bao gồm tiêu đề, tóm tắt, TLTK, bảng, hình)
- [ ] Đặt tên file: `AI4Industry_TenTacGia_TenBaiBao`
- [ ] Gửi email trước hạn nộp

---

*Cập nhật lần cuối: 2026-09-03*
