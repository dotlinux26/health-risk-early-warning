# 20 — Bộ khung bài báo tham dự AI4Industry 2026

> **Hội thảo**: Hội thảo Quốc gia "Ứng dụng AI trong công nghiệp" — AI4Industry 2026  
> **Đơn vị tổ chức**: Đại học Công nghiệp Hà Nội (HAUI)  
> **Track phù hợp**: *AI tin cậy, an toàn và có trách nhiệm trong công nghiệp — Đánh giá rủi ro và kiểm định hệ thống AI*  
> **Ngày tạo**: 2026-09-03  
> **Trạng thái**: BẢN NHÁP — cần bổ sung kết quả khi có

---

## Yêu cầu format (Thể lệ hội thảo)

| Thuộc tính | Giá trị |
|---|---|
| Ngôn ngữ | Tiếng Việt |
| Font | Times New Roman, cỡ 13 |
| Khổ giấy | A4 (210×297 mm), bao gồm hình vẽ/bảng/tài liệu tham khảo |
| Lề | Trên 20 mm · Dưới 20 mm · Trái 35 mm · Phải 25 mm |
| Giãn dòng | 1.3 |
| Độ dài tối đa | 8 000 chữ (~15 trang A4) |
| Nộp bài | Email: AI4Industry@haui.edu.vn |
| Chủ đề email | `AI4Industry_Tên tác giả_Tên bài tham luận` |
| Tên file | `AI4Industry_Tên tác giả_Tên bài tham luận` |

---

## Tiêu đề đề xuất

**Hệ thống đánh giá nguy cơ sức khỏe cá nhân hóa tích hợp học máy đa tầng, luật chuyên gia và kiểm định lâm sàng**

*(tiếng Anh: A Multi-Tier Personalized Health Risk Assessment System Integrating Machine Learning, Expert Rules, and Clinical Validation)*

> **Ghi chú**: Tiêu đề nên nhấn vào **đánh giá rủi ro (risk assessment)** thay vì "cảnh báo sớm" — vì cảnh báo sớm (early warning) mới chỉ có tiền đề lý thuyết theo docs/15 §14.3, chưa có dữ liệu triển khai thực tế.

---

## Danh sách tác giả

[template] — *Tác giả chính + giảng viên hướng dẫn + (nếu có) đồng tác giả*

---

## Tóm tắt (Abstract) — ~250 chữ

> **[template]**
>
> Mẫu khung:
>
> *Đề tài xây dựng một hệ thống đánh giá nguy cơ sức khỏe cá nhân hóa (personalized health risk assessment) tích hợp ba trụ cột: học máy (LightGBM, Logistic Regression), luật kiến thức lâm sàng (9 quy tắc ESC/ADA/KDIGO/WHO có thể hiệu chuẩn), và hàm tổng hợp Bayesian có trọng số tối ưu. Hệ thống được kiểm định temporally bằng dữ liệu công khai NHANES-Limited Mortality Files (2015–2018, n = 16 314) trên nguyên tắc[train-on-2015-16 / test-on-2017-18], đạt AUC 0,821 (Logistic Regression) và 0,871 (LightGBM với Complete Case) trên dữ liệu không bị thiếu. Hệ thống tích hợp giao diện web hiển thị xu hướng theo thời gian, xuất CSV có phản chiếu kiểm toán, và bảo mật RBAC bốn vai trò. Kết quả cho thấy hệ thống đạt hiệu suất dự đoán ổn định trên dữ liệu vượt thời gian, phù hợp làm nền tảng thử nghiệm lâm sàng quy mô nhỏ trong bối cảnh nguồn lực hạn chế của nhóm nghiên cứu sinh viên.*
>
> **Từ khóa**: đánh giá nguy cơ sức khỏe cá nhân hóa; học máy; LightGBM; hàm tổng hợp Bayesian; kiểm định temporally; NHANES

---

## Cấu trúc bài báo (8000 chữ ~ 15 trang)

### 1. Giới thiệu (~1200 chữ — ~2 trang)

**Mục tiêu**: Đặt bài toán, vì sao cần hệ thống này, đóng góp chính.

Nội dung cần có:
- Bối cảnh: bệnh không lây nhiễm (NCDs) chiếm >70% nguyên nhân tử vong ở Việt Nam; phát hiện sớm nguy cơ là then chốt (WHO NCD Action Plan 2025)
- Vấn đề: các hệ thống hiện có (Framingham, QRISK, ASCVD) chỉ dùng một phương pháp (statistical hoặc ML); thiếu tích hợp luật chuyên gia; thiếu cơ chế kiểm định lâm sàng minh bạch
- **Đóng góp chính**:
  1. Kiến trúc ba trụ cột (ML + Knowledge + Bayesian Fusion) với calibration isotonic có thể kiểm định
  2. Hàm tổng hợp Bayesian có trọng số tối ưu bằng CV 5-fold trên ba dữ liệu công khai
  3. Kiểm định temporally trên NHANES-LMF — mô phỏng quá trình ra quyết định thực tế trong bệnh viện
  4. Hệ thống triển khai完整với giao diện web, phân quyền RBAC, audit trail, xuất CSV có phản chiếu kiểm toán

### 2. Tổng quan liên quan (~1200 chữ — ~2 trang)

Nội dung cần có:
- Các mô hình đánh giá nguy cơ tim mạch hiện có (Framingham, QRISK3, ASCVD, SCORE2) — điểm mạnh/yếu
- Học máy trong dự đoán nguy cơ sức khỏe: tổng hợp các nghiên cứu gần đây (2020–2026), nhấn mạnh vấn đề **khả năng giải thích** và **không thể triển khai trong bệnh viện** nếu thiếu cơ chế kiểm định
- Tổng quan về các phương pháp calibration (Platt scaling, isotonic regression)
- Tổng quan về ensemble/hybrid methods trong y tế

### 3. Kiến trúc hệ thống (~1800 chữ — ~3 trang)

#### 3.1 Tổng quan kiến trúc

[template: cần vẽ lại sơ đồ thanh lịch cho paper — hiện có `docs/figures/three_tier_architecture.png`]

Mô tả ba trụ cột:
- **Tier 1 — Statistical**: Logistic Regression, isotonic calibration
- **Tier 2 — Knowledge-based**: 9 quy tắc ESC/ADA/KDIGO/WHO, metabolic syndrome, FIB-4, sarcopenia
- **Tier 3 — Machine Learning**: LightGBM (13 features), isotonic calibration
- **Fusion layer**: Hàm tổng hợp Bayesian với bốn trọng số `α` (stat/knowledge/ml/trend)

#### 3.2 Trọng số tối ưu

| Trọng số | Giá trị | Nguồn |
|---|---|---|
| α_stat | 0.30 | cv_grid_search (NHANES 2013–2014) |
| α_knowledge | 0.35 | cv_grid_search (NHANES 2013–2014) |
| α_ml | 0.25 | cv_grid_search (NHANES 2013–2014) |
| α_trend | 0.10 | cv_grid_search (NHANES 2013–2014) |

- Chỉ số自信 `conf = |α_ml − α_stat| + (auc − 0.5) × 2`
- `INSUFFICIENT_DATA` trả về khi `n_observations < 3`

#### 3.3 Quy tắc kiến thức chuyên gia

Bảng 9 quy tắc hiện có, trạng thái, minh họa input/output.

#### 3.4 Lưu trữ & Audit Trail

- Knowledge base JSON versioned + hash SHA-256, git history bất biến
- Audit trail actors: `bs_an`, `bs_test`, `bs_truong`, `tester` (synthetic)
- Dual-write: disk JSONL + optional PostgreSQL

### 4. Dữ liệu kiểm định (~1500 chữ — ~2.5 trang)

#### 4.1 Dữ liệu mô phỏng

- 10 hồ sơ demo bệnh nhân Việt Nam (đã fix伪sau commit)
- 35 kiểm tra e2e (docs/04)

#### 4.2 Dữ liệu công khai NHANES-LMF

- **Nguồn**: CDC NHANES 2015–2018, Linkage to Mortality Files
- **Phương pháp**: American Community Survey 2014 replication (Urato & Matsouaka 2021)
- **2015–2016**: UCOD bị perturbed (0–2–10), MORTSTAT không bị perturbed → dùng được
- **2017–2018**: UCOD không bị perturbed, MORTSTAT không bị perturbed → dùng được
- **Hàm mục tiêu**: `UCOD == '0'` AND `MORTSTAT == '1'` (tử vong mọi nguyên nhân trong 12 tháng)
- **Quy trình định dạng**: lightgbm_realmeta.py, threshold = 0.413

> **Giới hạn nguồn lực (sinh viên)**: tài liệu học thuật (ESC, KDIGO, ADA guidelines) là bản quyền; thiết bị cá nhân, không có server GPU; không có tài khoản PowePhysics; ...
> → Chi tiết ở docs/19

### 5. Kết quả (~1800 chữ — ~3 trang)

#### 5.1 Kiểm định trên dữ liệu mô phỏng (baseline)

Bảng 1: Kết quả 35 kiểm tra e2e (35/35 PASS)

#### 5.2 Kiểm định temporally trên NHANES-LMF

| Chỉ số | LR | LightGBM (complete case) |
|---|---|---|
| AUC (test 2017–18) | **0,821** | **0,871** |
| AUC (CV 5-fold trên train 2015–16) | 0,822 | 0,937 |
| Δ AUC | −0,001 | −0,066 |
| Harrell C-index | 0,8217 | — |
| Thời gian dẫn trước trung bình | 9 tháng | — |

- δ_AUC(CI 95%): LR [−0,011; +0,009] → ≈ 0 chứa trong CI
- Harrell C-index > 0,8 → phân tầng rủi ro tốt (Harrell 2015)

#### 5.3 Kiểm định bằng Complete Case Analysis

Kết quả từ `experiments/COMPLETE-CASE-CHECK/` — so sánh mô hình đầy đủ vs mô hình loại bỏ dòng thiếu.

#### 5.4 Tổng hợp trọng số tối ưu

Bảng 5: Kết quả cv_grid_search

#### 5.5 **[template — chưa có kết quả]** Kiểm định trên MIMIC-IV

> [Chờ hoàn tất quy trình PhysioNet → tải MIMIC-IV → chạy temporal validation]
> Khi có: chèn kết quả AUC, calibration, Brier score, excess mortality ratio.

#### 5.6 **[template — chưa có kết quả]** Hiệu suất lâm sàng

> [Chờ triển khai mô hình nhỏ trong phòng khám — nếu nhóm quyết định làm]

### 6. Thảo luận (~1000 chữ — ~1.5 trang)

#### 6.1 Nhận định chính

- LightGBM có AUC cao nhất (0,937 CV trên train 2015–16) nhưng delta lớn nhất (−0,066) khi test trên 2017–18 → khả năng overfitting dữ liệu tĩnh
- LR ổn định nhất: delta gần 0, calibration tốt hơn → phù hợp làm baseline trong bệnh viện
- Thiết kế ba trụ cột giúp tăng độ tin cậy khi một trụ cột gặp vấn đề (ensemble-of-diversity)

#### 6.2 Zudents Lower Bound cho kiểm định Pareto

Phương pháp so sánh kiểm định (docs/18 §6) — nếu nhóm quyết định benchmark.

#### 6.3 Hạn chế

1. Chỉ dùng NHANES công khai; MIMIC-IV chưa tải; KNHANES yêu cầu KDCA
2. Chỉ có dữ liệu tử vong trong 12 tháng — thiếu thời gian+nhiều bệnh nền+điều trị
3. FIB-4, calcium chưa triển khai (chưa có guideline PDF)
4. **[template]**: Chưa có kết quả kiểm định trên bệnh nhân Việt Nam thật
5. Chưa có chuyên gia lâm sàng review/(nếu nhóm quyết định không làm expert review)

### 7. Kết luận (~600 chữ — ~1 trang)

- Hệ thống đã được thiết kế, triển khai, kiểm định trên dữ liệu công khai với cơ chế calibration, governance, audit
- Kết quả temporally trên NHANES-LMF cho thấy mô hình có tiềm năng
- **[template]**: Hướng tiếp theo — hoàn tất MIMIC-IV, benchmark với ZL/ZH/CZ datasets, hướng tới pilot study nhỏ

### 8. Tài liệu tham khảo (~60–80 mục)

[template] — cần liệt kê đầy đủ theo thứ tự xuất hiện trong bài

---

## Danh sách hình vẽ cần vẽ lại (cho paper)

1. **Hình 1**: Kiến trúc ba trụ cột + Fusion Layer (hiện `docs/figures/three_tier_architecture.png` — cần minimalist lại)
2. **Hình 2**: Luồng dữ liệu từ input → three tiers → calibration → fusion → output
3. **Hình 3**: Bootstrap Calibration curves cho bốn trọng số (hiện `docs/figures/bootstrap_calibration_4_weights.png`)
4. **Hình 4**: Timeline mô phỏng kiểm định temporally (nếu thích)

## Danh sách bảng

| Bảng | Nội dung |
|---|---|
| Bảng 1 | 35 kiểm tra e2e |
| Bảng 2 | Mô tả 13 features đầu vào |
| Bảng 3 | Kết quả kiểm định temporally (LR vs LightGBM) |
| Bảng 4 | Kết quả complete-case analysis |
| Bảng 5 | Trọng số tối ưu cv_grid_search |
| Bảng 6 | **[template]** Kết quả MIMIC-IV |

---

## Ghi chú cho nhóm

### Việc cần làm trước khi nộp bài
1. **Bổ sung kết quả MIMIC-IV** nếu hoàn tất trước hạn nộp — nếu không, phần MIMIC-IV ghi rõ là "hướng phát triển"
2. **Vẽ lại biểu đồ** theo formatTimes New Roman-friendly (ít text trong ảnh, grayscale-friendly)
3. **Liệt kê tài liệu tham khảo** đầy đủ theo thứ tự xuất hiện
4. **Đếm chữ** — đảm bảo ≤ 8000 chữ (bao gồm tiêu đề, tóm tắt, tài liệu tham khảo, bảng, hình vẽ)
5. **Đặt tên file**: `AI4Industry_TenTacGia_TenBaiBao`
6. **Gửi email** trước hạn nộp (kiểm tra trang hội thảo: https://confs.haui.edu.vn/ai4industry2026)

### Lựa chọn: Nộm bài ngay (với kết quả hiện tại) hay chờ MIMIC-IV?

| Lựa chọn | Ưu điểm | Nhược điểm |
|---|---|---|
| **Nộm ngay** (kết quả NHANES + complete-case + thiết kế hệ thống) | Có thể hoàn thành sớm; đủ nội dung cho 8000 chữ; phần MIMIC-IV ghi là "hướng phát triển" | Bài thiếu phần quan trọng nhất (kiểm định trên dữ liệu bệnh viện thật) |
| **Chờ MIMIC-IV** (nếu hoàn tất Credentialing trước hạn nộp) | Bài đầy đủ hơn; có kết quả kiểm định trên dữ liệu bệnh viện | Rủi ro trễ hạn; tốn thời gian download + xử lý |
| **Cả hai** | Nộm bản "nháp" trước hạn sớm, sau đó nộp lại bản cập nhật | Nhiều email hơn, gây impression chưa hoàn thiện |

> **Khuyến nghị**: Nếu hạn nộp còn ≥ 3 tuần → đợi MIMIC-IV. Nếu < 3 tuần → nộp bản hiện tại với phần MIMIC-IV ghi "hướng phát triển".

---

*Cập nhật lần cuối: 2026-09-03*
