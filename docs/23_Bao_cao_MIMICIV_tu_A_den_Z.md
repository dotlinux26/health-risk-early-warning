# 23. Báo cáo toàn bộ quy trình MIMIC-IV: từ Credentialing đến Kết quả (Tuân thủ DUA)

**Ngày cập nhật:** 2026-09-11  
**Trạng thái:** HOÀN TẤT credentialing core + temporal validation core files  
**Lưu ý:** Báo cáo này chỉ mô tả quy trình, KHÔNG chứa dữ liệu MIMIC-IV thực tế (tuân thủ DUA §4.2, §7.1)

---

## 1. Tổng quan

| Giai đoạn | Trạng thái | Ngày hoàn tất |
|---|---|---|
| PhysioNet account tạo | ✅ | 2026-08-15 |
| CITI Training (Data or Specimens Only Research) | ✅ | 2026-08-20 |
| CITI Certificate upload | ✅ | 2026-08-22 |
| Reference (GVHD) confirm | ✅ | 2026-08-25 |
| DUA Online ký (MIMIC-IV v3.1) | ✅ | 2026-09-03 |
| Credentialed Access granted | ✅ | 2026-09-03 |
| Tải core files (10 files, ~108 MB) | ✅ | 2026-09-04 |
| Tổ chức thư mục & .gitignore | ✅ | 2026-09-04 |
| Viết adapter `build_mimic_dataset.py` | ✅ | 2026-09-05 |
| Chạy build dataset | ✅ | 2026-09-05 |
| Chạy temporal validation | ✅ | 2026-09-05 |
| Cập nhật bài báo AI4Industry | ✅ | 2026-09-11 |

---

## 2. Credentialing chi tiết (Tuân thủ DUA §3, §4)

### 2.1 Yêu cầu bắt buộc của PhysioNet
MIMIC-IV Clinical Database v3.1 là **Credentialed Access** — yêu cầu:
1. Tài khoản PhysioNet verified email
2. **CITI Training**: "Data or Specimens Only Research" (Good Clinical Practice + HIPAA)
3. **Reference**: Người có thẩm quyền (GVHD) xác nhận trên PhysioNet
4. **Data Use Agreement (DUA)**: Ký online click-through

### 2.2 Hồ sơ đã nộp
| Hồ sơ | File | Trạng thái |
|---|---|---|
| CITI Certificate | `docs/CITI_Certificate.pdf` | ✅ Committed (private repo) |
| CITI Completion Report | `docs/CITI_Completion_Report.pdf` | ✅ Committed (private repo) |
| DUA signed | Online trên PhysioNet | ✅ `canhnguyen26` account |

> **TUÂN THỦ DUA**: Certificate/Report PDF **KHÔNG** push public. Chỉ lưu local/private repo. DUA yêu cầu: "Will not share access to PhysioNet restricted data with anyone else" (§4.2).

### 2.3 Mục đích nghiên cứu khai báo
> *"Student research (NCKH): temporal validation of multi-tier health risk assessment framework on MIMIC-IV vs NHANES-LMF external validation"*

---

## 3. Dữ liệu đã tải (Core files only — Open Access trong Credentialed)

### 3.1 Danh sách file đã tải (~108 MB nén)

| File | Kích thước | Mô tả | Dùng cho |
|---|---|---|---|
| `patients.csv.gz` | 2.7 MB | Demographics, anchor_age, gender, dod | Age, sex, mortality label |
| `admissions.csv.gz` | 19 MB | Admittime, dischtime, deathtime, hospital_expire_flag | Temporal split, labels |
| `diagnoses_icd.csv.gz` | 32 MB | ICD-10 codes per admission | 17 comorbidity flags (Charlson/Deyo) |
| `procedures_icd.csv.gz` | 7.4 MB | ICD-10 procedures | (Chưa dùng) |
| `omr.csv.gz` | 43 MB | Outpatient measurements (BP, BMI, Weight, Height, eGFR) | Vitals/anthropometrics |
| `icustays.csv.gz` | 3.2 MB | ICU stay IDs, intime | (Chưa dùng) |
| `d_labitems.csv.gz` | 12.9 KB | Lab item dictionary | (Chưa dùng — không có labevents) |
| `d_items.csv.gz` | 57.4 KB | Chart item dictionary | (Chưa dùng — không có chartevents) |
| `d_icd_diagnoses.csv.gz` | 855 KB | ICD-10 diagnosis descriptions | Mapping codes |
| `d_icd_procedures.csv.gz` | 575 KB | ICD-10 procedure descriptions | (Chưa dùng) |

### 3.2 File KHÔNG tải (cần restricted access / full DUA)
- `labevents.csv.gz` (~2.4 GB) — Lab results inpatient
- `chartevents.csv.gz` (~3.3 GB) — Vitals/labs inpatient  
- `emar.csv.gz`, `pharmacy.csv.gz`, `poe.csv.gz` — Medications
- `inputevents.csv.gz`, `outputevents.csv.gz` — ICU I/O

> **TUÂN THỦ DUA**: Chỉ tải file **core** (được phép với credentialed access cơ bản). Full dataset yêu cầu DUA mở rộng và Institutional Review Board (IRB) approval riêng.

---

## 4. Cấu trúc thư mục (An toàn DUA)

```bash
data/mimiciv/3.1/
├── hosp/
│   ├── patients.csv.gz
│   ├── admissions.csv.gz
│   ├── diagnoses_icd.csv.gz
│   ├── procedures_icd.csv.gz
│   ├── omr.csv.gz
│   ├── d_labitems.csv.gz
│   ├── d_items.csv.gz
│   ├── d_icd_diagnoses.csv.gz
│   └── d_icd_procedures.csv.gz
└── icu/
    └── icustays.csv.gz
```

### .gitignore (Committed: `abb5be8`)
```gitignore
# Dữ liệu MIMIC-IV — DUA PhysioNet cấm tái phân phối, KHÔNG BAO GIỜ commit
data/mimic/
data/mimiciv/
**/mimic*/
mimic*.zip
mimic*.csv*
mimic*.parquet
*_mimic*
bypass_*.txt
```

> **Kiểm tra**: `git status --ignored` xác nhận `data/mimiciv/` bị ignored. **ZERO MIMIC data trong git history**.

---

## 5. Adapter: `scripts/build_mimic_dataset.py`

### 5.1 Đặc trưng trích xuất (từ OMR + ICD-10)

| Feature | Nguồn OMR | Xử lý |
|---|---|---|
| `systolic_bp` | `Blood Pressure` (format "120/80") | Parse → systolic |
| `diastolic_bp` | `Blood Pressure` | Parse → diastolic |
| `bmi` | `BMI (kg/m2)`, `BMI` | Numeric |
| `weight` | `Weight (Lbs)` | Lbs → kg (×0.453592) |
| `height` | `Height (Inches)` | Inches → cm (×2.54) |
| `egfr` | `eGFR` | Numeric (chỉ 239 rows) |

**Comorbidity flags (17) từ ICD-10 (Charlson/Deyo):**
`mi`, `chf`, `pvd`, `cerebrovascular`, `dementia`, `copd`, `rheumatic`, `pud`, `mild_liver`, `dm`, `dm_complications`, `paraplegia`, `renal`, `cancer`, `metastatic`, `severe_liver`, `hiv`

### 5.2 Temporal Split (Shifted Years)
MIMIC-IV dùng shifted years **2105–2214**:
- **Train**: shifted years ≤2115 (n=22.552 admissions)
- **Test**: shifted years ≥2116 (n=523.476 admissions)

> **No data leakage**: Chỉ dùng OMR measurements **trước admittime** (merge_asof direction="backward")

### 5.3 Labels
| Label | Định nghĩa |
|---|---|
| `mortality_30d` | Tử vong trong 30 ngày sau admittime |
| `in_hospital_mortality` | `hospital_expire_flag` từ admissions |

---

## 6. Dataset tạo ra

| File | Kích thước | Mô tả |
|---|---|---|
| `data/datasets/mimiciv_train.csv` | ~2.5 MB | 22.552 rows, 25 features + 2 labels |
| `data/datasets/mimiciv_test.csv` | ~55 MB | 523.476 rows |
| `data/datasets/mimiciv_merged.csv` | ~57 MB | Full dataset |
| `data/datasets/mimiciv_metadata.json` | ~3 KB | Metadata, missingness, prevalence |

### 6.1 Missingness (Train set)

| Feature | Missing % | Ghi chú |
|---|---|---|
| systolic_bp / diastolic_bp | 61.04% | OMR outpatient only |
| bmi | 57.81% | |
| weight | 54.62% | |
| height | 61.05% | |
| egfr | 99.98% | Chỉ 239 rows có eGFR |
| 17 comorbidities | 0.00% | Từ ICD-10 — complete |

**Prevalence:**
- Train mortality_30d: 2.56%
- Test mortality_30d: 2.00%

---

## 7. Temporal Validation: `scripts/run_temporal_mimiciv.py`

### 7.1 Models
- **Logistic Regression**: Pipeline (Imputer median + StandardScaler + LR balanced)
- **LightGBM**: 500 estimators, early stopping, class_weight balanced

### 7.2 Kết quả (Test set n=523.476)

| Chỉ số | Logistic Regression | LightGBM |
|---|---|---|
| **AUC Temporal** | **0.752** | 0.751 |
| AUC Random (70/30) | 0.761 | 0.784 |
| **Δ AUC (Temporal − Random)** | **−0.010** | **−0.033** |
| AUPRC Temporal | 0.0586 | 0.0609 |
| Brier Score | 0.2069 | 0.1646 |
| ECE (10-bin) | 0.3843 | 0.3513 |
| Capture Rate Top Quintile | 52.8% | 54.3% |

### 7.3 Nhận định
1. **LR ổn định hơn** (ΔAUC −0.010) — phù hợp baseline triển khai
2. **LGBM overfitting** dữ liệu tĩnh (ΔAUC −0.033)
3. **AUC thấp hơn NHANES-LMF** (0.75 vs 0.82) do:
   - Missingness vitals ~60% (OMR outpatient ≠ inpatient)
   - Driver chính = comorbidity flags (ICD-10 complete)
   - Outcome 30d mortality ICU pattern khác tử vong dân cư
4. **Capture rate ~53–54%**: Phát hiện >50% ca tử vong 30d ở top quintile

### 7.4 Evidence Package
```
experiments/EXP-TEMPORAL-MIMICIV/
├── summary.json    # Full metrics
├── summary.md      # Markdown report
```

---

## 8. Cập nhật bài báo AI4Industry 2026

**Files:** `docs/22_Bai_bao_AI4Industry_2026_full.md` + `.tex`

### Sections cập nhật:
- **Abstract**: Thêm MIMIC-IV results (AUC 0.752, capture 53%)
- **§4.5**: Mô tả chi tiết MIMIC-IV data (OMR, ICD-10, missingness, split)
- **§5.5**: Bảng kết quả MIMIC-IV + nhận định
- **§6.3**: Hạn chế cập nhật (missingness 60%, outcome difference)
- **§8**: Kết luận so sánh 2 dataset temporal độc lập
- **Checklist**: ✅ MIMIC-IV temporal validation done

---

## 9. Git Commits liên quan

| Commit | Ngày | Nội dung |
|---|---|---|
| `e798a02` | 2026-08-28 | docs/18: CITI hoàn tất + kế hoạch MIMIC-IV |
| `1c0d4d0` | 2026-08-30 | docs/19 §7: Giới hạn NCKH + workaround |
| `6d44e77` | 2026-09-01 | CITI PDFs committed (private) |
| `abb5be8` | 2026-09-03 | .gitignore explicit `data/mimiciv/` block |
| `c5384cd` | 2026-09-03 | Chuẩn hóa report (ký tự Trung, giãn dòng tham chiếu) |
| `315e1c5` | 2026-09-05 | Bài báo full MD + LaTeX (template MIMIC-IV) |
| `6ef0351` | 2026-09-11 | Cập nhật bài báo với kết quả MIMIC-IV thực tế |

---

## 10. Kiểm tra tuân thủ DUA (Checklist)

| DUA Clause | Yêu cầu | Trạng thái |
|---|---|---|
| §3.1 | Chỉ dùng cho nghiên cứu khoa học hợp pháp | ✅ NCKH sinh viên, temporal validation |
| §3.2 | Không cố định danh tính cá nhân/cơ quan | ✅ Chỉ dùng aggregated features, không PHI |
| §3.3 | Không chia sẻ truy cập dữ liệu hạn chế | ✅ Chỉ tài khoản `canhnguyen26` |
| §3.4 | Bảo vệ an ninh vật lý/điện tử | ✅ Local encrypted disk, .gitignore block |
| §3.5 | Báo cáo PHI nếu phát hiện | ✅ Không có PHI trong core files |
| §3.6 | Đóng góp code về cộng đồng | ✅ Repo public: `github.com/dotlinux26/health-risk-early-warning` |
| §4.2 | Không tái phân phối dữ liệu | ✅ .gitignore chặn toàn bộ `data/mimiciv/` |
| §7.1 | DUA có thể chấm dứt bất cứ lúc nào | ✅ Hiểu rõ,義务 tiếp tục sau chấm dứt |

---

## 11. Hạn chế hiện tại & Next Steps

### 11.1 Hạn chế (đã công bố trong bài báo §6.3)
1. **Missingness vitals ~60%**: OMR outpatient only, thiếu inpatient vitals/labs
2. **Outcome = all-cause mortality**: Không phải disease onset
3. **Chưa có creatinine, glucose, HbA1c, heart_rate**: Cần labevents/chartevents
4. **Shifted years**: Không map trực tiếp calendar year

### 11.2 Next Steps (cần full DUA/restricted access)
| Bước | Yêu cầu | Ưu tiên |
|---|---|---|
| Tải `labevents.csv.gz`, `chartevents.csv.gz` | DUA mở rộng + IRB approval | Cao |
| Trích xuất glucose, creatinine, HbA1c, heart_rate từ labs/vitals inpatient | Cần full data | Cao |
| Temporal validation với features đầy đủ | So sánh fair với NHANES-LMF | Cao |
| KNHANES external validation | KDCA RDC proposal | Trung bình |

---

## 12. Kết luận

**Đã hoàn tất end-to-end với MIMIC-IV core files:**
- ✅ Credentialing full (CITI + Reference + DUA)
- ✅ Tải 10 core files (~108 MB) — tuân thủ DUA
- ✅ Adapter trích xuất features từ OMR + ICD-10
- ✅ Temporal validation trên 546k admissions (shifted years)
- ✅ Kết quả: LR AUC 0.752 (ổn định), LGBM AUC 0.751 (overfitting)
- ✅ Cập nhật bài báo AI4Industry với 2 dataset temporal độc lập
- ✅ **ZERO MIMIC data trong repo/git history** — tuân thủ DUA §4.2

**Sẵn sàng cho:** Full data access khi có DUA mở rộng, KNHANES, pilot study VN.

---

*Báo cáo này chỉ mô tả quy trình và metadata. Không chứa dữ liệu bệnh nhân MIMIC-IV thực tế.*