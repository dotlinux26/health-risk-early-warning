# docs/18 — Giai đoạn P2: dữ liệu dọc theo thời gian

*Ngày: 24/08/2026 · Trạng thái: P2 khởi động, mục P2.1 (mới) đã có kết quả đầu tiên*

Tài liệu này thay thế phần "nguồn khả thi" còn mở của docs/16 (§3.1, §3.4) bằng
khảo sát cụ thể các bộ dữ liệu dọc và báo cáo thí nghiệm temporal validation
đầu tiên chạy trên dữ liệu thật công khai.

---

## 1. Mục tiêu giai đoạn

Các vấn đề còn bỏ ngỏ sau P1 đều chung một gốc: **thiếu dữ liệu có trục thời gian
và biến cố tương lai**:

| Vấn đề (docs/16) | Cần gì từ dữ liệu |
|---|---|
| T7 — validation theo thời gian + lead time | biến cố ghi nhận theo thời gian sau kỳ đo |
| T8 — vòng lặp nhãn–đầu vào | nhãn sinh từ biến cố tương lai độc lập phép đo |
| T9 — mô hình chuỗi thời gian | ≥500 người × ≥60 quan sát liên tục |
| T10 — xác nhận ngoài | quần thể khác cùng schema |

## 2. Kho dữ liệu dọc khả dụng (khảo sát 24/08/2026)

| Dataset | Trục thời gian | Biến cố | Truy cập | Quyết định |
|---|---|---|---|---|
| **NHANES Public-Use Linked Mortality File 2019** (1999–2018) | tháng kể từ kỳ khám MEC đến tử vong / 31-12-2019 | tử vong toàn bộ + nguyên nhân chính | **công khai**, FTP CDC, không cần credential | ✅ **Đã tích hợp** |
| NHANES III LMF | như trên, cohort 1988–1994 | như trên | công khai | dự phòng (quá cũ) |
| MIMIC-IV (v3.x, PhysioNet) | EHR ICU nhiều lần nhập viện | nhập viện/tử vong trong bệnh viện | PhysioNet account + chứng nhận CITI + DUA | kế hoạch P2.7 (xin quyền) |
| eICU-CRD, HiRID, AmsterdamUMCdb | ICU đa trung tâm | như MIMIC | tương tự MIMIC | không ưu tiên (chỉ ICU) |
| EHRSHOT / INSPECT / MedAlign (Stanford STARR, OMOP) | EHR ngoại + nội trú dài hạn, ~26k bệnh nhân | 15 tác vụ few-shot, tử vong, tái nhập viện | Redivis DUA + CITI | ứng viên P2.7+ |
| CardioEHR (Bệnh viện Union Vũ Hán, 73k BN tim mạch 2010–2024) | lần khám/nhập viện, lab theo thời gian | chẩn đoán tim mạch, tử vong | controlled access OMIX012906 + DUA | ứng viên dài hạn (gần schema VN hơn) |
| UK Biobank | khảo sát + follow-up quốc gia | rộng | đơn vị nghiên cứu, phí | ngoài tầm hiện tại |
| KNHANES–Cause of Death linkage (Hàn Quốc) | tháng từ kỳ khám đến tử vong (2007→2019+, follow-up TB 8.4 năm) | tử vong toàn bộ + nguyên nhân theo KCD | **đã xác nhận tồn tại** ('07–'22: 69 855 người, linkage 97.5%) nhưng chỉ phân tích tại **Research Data Center của KDCA** sau khi đề cương được duyệt — không tải về | ứng viên P2.3 (cần nộp đề cương); nguồn: kdca.go.kr, e-epih.e2022021 |
| Kênh nhập hệ thống (P2.4 docs/16) | ngày, tự tích lũy | tự định nghĩa qua governance | nội bộ | đang chạy |

Lý do chọn NHANES-LMF làm bước đầu: (i) **khớp SEQN trực tiếp** với
`nhanes_merged.csv` sẵn có — chi phí tích hợp gần bằng 0; (ii) outcome tử vong
**độc lập tuyệt đối** với các chỉ số đầu vào, giải quyết đúng con đường (a) của
T8; (iii) không rào cản pháp lý, kết quả tái lập được bởi bất kỳ ai.

## 3. Đã tích hợp: NHANES Linked Mortality File

Pipeline: `scripts/fetch_nhanes_mortality.py` → `data/datasets/nhanes_mortality.csv`

- Tải file fixed-width `.dat` của 10 chu kỳ 1999–2018 từ
  `ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/`,
  parse theo bố cục chính thức `R_ReadInProgramAllSurveys.R` (SEQN 1–6,
  ELIGSTAT 15, MORTSTAT 16, UCOD_LEADING 17–19, PERMTH_EXM 46–48).
- Ghép với `data/datasets/nhanes_mortality.csv`: 16 314 dòng, khớp LMF 10 065
  (chu kỳ 2015-16: 99,7%; 2017-18: 99,5%; 2021-23 chưa có linkage công khai).
- Sinh nhãn: `death_all`, `death_cvd` (bệnh tim), `death_5y`, `cvd_death_5y`,
  `followup_months`.

Cảnh báo về bản public-use (ghi đúng theo codebook NCHS): một số bản ghi bị
**perturbation** — follow-up time hoặc UCOD được thay bằng giá trị tổng hợp;
**tình trạng sống/chết (MORTSTAT) không bị thay**. Kiểm chứng thực tế: chu kỳ
2015–2018 chỉ còn UCOD 1 (bệnh tim), 2 (ung thư), 10 (khác) → đột quỵ/ĐTĐ/bệnh
thận không tách được khỏi "nguyên nhân khác".

## 4. Thí nghiệm EXP-TEMPORAL-LMF — temporal validation đầu tiên

Chạy: `python3 scripts/run_temporal_validation.py` →
`experiments/EXP-TEMPORAL-LMF/{summary.json, summary.md}`

Thiết kế (chốt trước khi nhìn kết quả — theo tinh thần §3.1 docs/16):

1. Cohort: người lớn ≥20 tuổi, linkage đủ điều kiện, chu kỳ **2015-2016
   (train, n=5048)** và **2017-2018 (test, n=4773)** — split **theo mốc thời gian**.
2. Outcome chính: tử vong toàn bộ **≤12 tháng** từ ngày khám MEC — cửa sổ duy
   nhất hợp lệ cho cả hai chu kỳ (chu kỳ test có follow-up tối đa ~37 tháng do
   kỳ khám muộn). Prevalence: train 0,97%, test 1,40%.
3. Đặc trưng: 7 chỉ số chuẩn hệ thống + tuổi + giới tính, impute median
   fit-train, không có thông tin nào từ tương lai.
4. So sánh split temporal vs random cùng tỉ lệ; hiệu chỉnh isotonic fit-train.

### Kết quả chính

| Model | Split | ROC-AUC | AUPRC | Brier | ECE(10) |
|---|---|---|---|---|---|
| LR | temporal test | **0.8209** | 0.0794 | 0.0136 | — |
| LR | random test | 0.8411 | 0.0839 | 0.0122 | — |
| LGBM | temporal test | 0.7709 | 0.0459 | 0.0142 | — |
| LGBM | random test | 0.7810 | 0.0662 | 0.0125 | — |
| LR + isotonic | temporal test | 0.7999 | — | 0.0138 | 0.0040 |

- Harrell's C-index toàn follow-up (≤60 tháng, censoring đúng protocol):
  LR **0.8217**, LGBM 0.7763.
- Lead time: top-quintile nguy cơ bắt **53/94** ca tử vong ≤24 tháng của tập
  test, trung vị **9 tháng** trước sự kiện (nhóm còn lại 8 tháng).

### Diễn giải

1. **Lần đầu tiên đề tài có bằng chứng prospective thật**: đặc trưng một kỳ khám
   dự báo tử vong 12 tháng trên chu kỳ dữ liệu *sau* đó, AUC ≈ 0.82. Không còn
   vòng lặp nhãn — outcome do NDI ghi nhận độc lập với mọi đầu vào.
2. **Random split chỉ lạc quan hơn 0.01–0.02 AUC** so với temporal split → các
   kết quả benchmark trước đây không bị thổi phồng nghiêm trọng bởi dataset
   shift giữa chu kỳ; mức độ này nằm trong biên độ seed (K2: ±0.009).
3. **LR thắng LGBM** vì chỉ 49 kiện ở train — nhất quán với nguyên tắc "ít kiện
   thì mô hình đơn giản tổng quát tốt hơn" (đã nêu trong docs/15 §5E); LightGBM
   overfit (train AUC = 1.0).
4. **Đối chứng trực tiếp vấn đề T8**: trên đúng tập test, mô hình học từ nhãn
   cắt ngang cũ chỉ đạt AUC 0.628 (`label_htn`) / 0.573 (`label_dm`) khi dự báo
   tử vong thật, trong khi mô hình học từ biến cố tương lai đạt 0.821 — nhãn
   vòng lặp vẫn mang thông tin nguy cơ nhưng yếu hơn hẳn nhãn biến cố.
5. Lead time trung vị 9 tháng > 0: đạt tiêu chí thành công tối thiểu của §3.1
   docs/16 **ở cấp cohort, horizon tháng**. Tiêu chí gốc viết cho cửa sổ
   30/90/180 ngày với dữ liệu dọc theo ngày — vẫn phải chờ P2.4/P2.5.

## 5. Tác động lên các vấn đề còn lại

- **T7**: giao thức đã chạy end-to-end trên dữ liệu thật; lead time > 0 đạt ở
  cấp cohort Mỹ. Chữ "cảnh báo sớm" giờ có nghĩa định lượng được ("top-20% nguy
  cơ bao phủ 56% tử vong ≤24 tháng, trung vị 9 tháng") nhưng vẫn giữ nguyên quy
  tắc docs/16 §5: chưa dùng trong sản phẩm cho đến khi lặp lại được trên dữ liệu
  dọc theo ngày.
- **T8**: con đường khắc phục căn bản (a) đã chứng minh khả thi và cho điểm số
  mạnh hơn hẳn; sản xuất vẫn dùng nhãn cắt ngang cho đến khi chuyển huấn luyện
  sang outcome biến cố (P2.7).
- **T10**: hold-out theo thời gian 2017-18 là external-by-time đầu tiên đạt
  tiêu chí (AUC drop 0.02 ≤ 0.05). Xác nhận ngoài theo địa lý/quần thể
  (KNHANES, MIMIC-IV, cơ sở VN) vẫn mở.
- **T9**: chưa đổi — cần chuỗi ≥60 điểm/người, mortality data không thay thế.

## 6. Hạn chế bắt buộc ghi nhận

1. Public-use LMF bị perturbation (một số follow-up/UCOD tổng hợp) — ước lượng
   có nhiễu nhỏ, tình trạng sống/chết không đổi.
2. Tử vong ≠ khởi phát tăng HA/ĐTĐ; mô hình trả lời câu hỏi "ai có nguy cơ chết
   sớm", không phải "ai sắp mắc bệnh".
3. Horizon 12 tháng ít kiện (67 events test) — khoảng tin cậy AUC rộng (~±0.04);
   nên coi 0.82–0.84 là một khoảng, không phải một con số.
4. Quần thể Hoa Kỳ ≠ Việt Nam; 2021–2023 chưa có mortality linkage công khai.
5. Perturbation làm mất phân loại UCOD chi tiết → phân tích nguyên nhân tim mạch
   chỉ còn "tử vong do bệnh tim".

## 7. Bước tiếp theo của P2

| Bước | Công việc | Ghi chú |
|---|---|---|
| ~~P2.1 (mới)~~ | ✅ Temporal validation trên NHANES-LMF | mục này |
| P2.2 | ✅ Complete-case check glucose 52% | `experiments/COMPLETE-CASE-CHECK/` — cây ổn định (<0.01), LR lệch +0.016; giữ impute production (docs/16 §3.5) |
| P2.3 | Xác nhận ngoài theo địa lý | KNHANES linkage **đã xác nhận tồn tại** nhưng chỉ chạy tại RDC của KDCA (nộp đề cương); thực tế hơn: MIMIC-IV sau khi có quyền (P2.7) |
| P2.4 | Dữ liệu dọc qua kênh nhập hệ thống | ≥50 người × ≥30 ngày là mốc giữa |
| P2.5 | Lead time theo ngày khi có P2.4 | áp đúng tiêu chí §3.1 docs/16 |
| P2.6 | Chuỗi thời gian khi đủ dữ liệu | LightGBM+lag là baseline bắt buộc |
| **P2.7 (mới)** | Xin quyền MIMIC-IV (CITI + DUA) làm nguồn biến cố nhập viện; đồng thời đánh giá EHRSHOT/CardioEHR | chuyển huấn luyện sang outcome tương lai |

### Hồ sơ nghiên cứu đã đồng bộ (24/08/2026)

Cùng ngày, docs 01–07 được rà soát theo trạng thái hiện tại: docs/06 cập nhật
model sản xuất (n=16 314, AUC 0.9356), thêm §2.4 calibration production, đánh
dấu các giới hạn đã giải quyết (temporal split, imputer, calibration), bảng
kết quả demo đo lại sau isotonic; docs/03 bổ sung mô tả governance vòng đời
luật; docs/02/04/07 chỉnh định vị thuật ngữ "cảnh báo sớm" theo docs/15 §14;
`experiments/README.md` liệt kê đủ các evidence package K1–K4 + P2.

**Bổ sung cuối ngày 24/08/2026:** UI v2 (dark mode, biểu đồ xu hướng SVG,
xuất CSV bản ghi + audit trail — khắc phục hạn chế docs/17 §6 mục 3) và bộ
kiểm thử end-to-end chính thức `scripts/e2e_test.py` — 35 PASS / 0 FAIL,
gồm cả check cho payload `temporal_validation` và `complete_case` của
`/api/benchmark/research`. Báo cáo tổng kết đã giải quyết / giới hạn còn lại:
**docs/19**.
