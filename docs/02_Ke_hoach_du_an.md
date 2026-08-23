# 02. Kế hoạch dự án (Project Plan)

**Đề tài:** *Hệ thống đánh giá nguy cơ sức khỏe cá nhân hóa tích hợp học máy và hỗ trợ quyết định lâm sàng.* (Tên chính thức cập nhật ngày 23/08/2026; tên cũ và lý do đổi xem `docs/15` mục 14.)

---

## 1. Mục tiêu & Phạm vi (SMART)

### Mục tiêu tổng quát
Xây dựng hệ thống **đa tầng, minh bạch** phân tích chuỗi chỉ số cơ thể (theo thời gian) của từng cá nhân để cảnh báo sớm nguy cơ bệnh, kèm giải thích truy xuất được.

### Mục tiêu cụ thể (SMART)
- **S1:** Thiết kế bộ phát hiện bất thường cá nhân hóa (Z-Score, Isolation Forest, xu hướng EWMA/STL) đạt F1 ≥ 0,8 trên dữ liệu thử nghiệm.
- **S2:** Xây dựng cơ sở tri thức y khoa dạng luật cho ≥ 4 hệ cơ quan (tim mạch, nội tiết, thận, hô hấp) đối chiếu từ hướng dẫn lâm sàng.
- **S3:** Xây dựng mô hình điểm rủi ro (baseline LightGBM + đối chứng LSTM) với AUROC ≥ 0,75 trên tập nội kiểm.
- **S4:** Đầu ra chuẩn hóa 3 thành phần bắt buộc: *phân loại rủi ro*, *hệ cơ quan ảnh hưởng*, *giải thích thông số + khuyến nghị chuyên khoa*.
- **S5:** Cung cấp API (FastAPI) + báo cáo PDF/markdown để bác sĩ xem xét (chỉ hỗ trợ quyết định).

### Ngoài phạm vi (Out of scope)
- Không chẩn đoán thay bác sĩ; không kê đơn; không thay thế thiết bị y tế chuyên dụng.
- Không xử lý văn bản lâm sàng phi cấu trúc (giao lại cho hướng NLP/Foresight nếu mở rộng sau).

---

## 2. Phân rã công việc (WBS) & Lộ trình

Dự án ước tính **16–18 tuần** (4 tháng), chia 5 giai đoạn:

### Giai đoạn 0 — Khởi động & Rà soát tri thức (Tuần 1–2)
- [ ] Hoàn thiện đề cương (đã có bản tổng hợp `01_Tong_hop_nghien_cuu.md`).
- [ ] Khảo sát dữ liệu có sẵn: số liệu tự thu (thủ công), dataset công khai (MIMIC-IV, PhysioNet, NHANES — lưu ý giấy phép/đạo đức).
- [ ] Chốt chuẩn dữ liệu đầu vào (schema) & quy trình đạo đức nghiên cứu (IRB nếu dùng dữ liệu thực).

**Sản phẩm:** Bản đặc tả dữ liệu (Data Specification) v1.

### Giai đoạn 1 — Dữ liệu & Tiền xử lý (Tuần 3–5)
- [ ] Viết pipeline nạp/làm sạch/khớp thời gian (resample theo ngày/tuần).
- [ ] Xử lý thiếu dữ liệu: nội suy tuyến tính, LOCF, chỉ báo "missing".
- [ ] Xây dựng **simulator sinh dữ liệu tổng hợp** (để phát triển khi chưa có dữ liệu thực — bài học từ Delphi).
- [ ] Đặc trưng hóa: rolling mean/std, delta, % thay đổi, Z-Score lịch sử.

**Sản phẩm:** Bộ dữ liệu chuẩn hóa + script `src/data/`.

### Giai đoạn 2 — Tầng 1 & Tầng 2 (Tuần 6–9)
- [ ] Tầng 1: Z-Score cá nhân hóa, Isolation Forest đa chiều, phát hiện xu hướng (EWMA/STL).
- [ ] Tầng 2: Mô hình hóa cơ sở tri thức (JSON), viết engine đánh giá luật, bản đồ chỉ số → hệ cơ quan.
- [ ] Đánh giá tầng 1 (precision/recall bất thường) trên dữ liệu tổng hợp + dữ liệu thực (nếu có).

**Sản phẩm:** `src/tier1_anomaly/`, `src/tier2_knowledge/` chạy được.

### Giai đoạn 3 — Tầng 3 & Đối chứng mô hình (Tuần 10–14)
- [ ] Điểm rủi ro: LightGBM + SHAP (chính); LSTM/GRU (đối chứng).
- [ ] Thiết lập thí nghiệm so sánh (CV, metrics AUROC/AUPRC/F1), kiểm định ý nghĩa thống kê.
- [ ] Tích hợp bộ giải thích: SHAP global/local → văn bản giải thích tiếng Việt.
- [ ] Kiểm chứng chéo với luật y khoa (Tầng 2) — mô hình & luật đồng thuận thì tăng độ tin cậy.

**Sản phẩm:** Báo cáo benchmark (so sánh các mô hình), `src/models/`.

### Giai đoạn 4 — Hệ thống & Giao diện (Tuần 15–16)
- [ ] API FastAPI (`/assess`, `/health`, `/explain`).
- [ ] Báo cáo đầu ra cấu trúc (markdown/PDF).
- [ ] Đóng gói, tài liệu sử dụng, test E2E.
- [ ] (Tùy chọn) Dashboard demo bằng Streamlit.

**Sản phẩm:** Hệ thống chạy end-to-end + tài liệu.

### Giai đoạn 5 — Báo cáo & Công bố (Tuần 17–18)
- [ ] Viết báo cáo khoa học (cấu trúc: tổng quan, phương pháp, kết quả, thảo luận, hạn chế).
- [ ] Chuẩn bị demo & slide bảo vệ.
- [ ] Xem xét công bố hội thảo trong nước/quốc tế (dựa trên benchmark).

---

## 3. Phân công vai trò (đội ngũ)

| Thành viên | Vai trò | Trách nhiệm | Giai đoạn |
|------------|---------|-------------|-----------|
| **Nguyễn Đức Cảnh** | Trưởng nhóm / Chủ trì code | Chịu trách nhiệm toàn bộ code: pipeline dữ liệu, Tầng 1–3, mô hình, benchmark, API; định hướng phương pháp; phê duyệt kết quả | Toàn bộ |
| **Nguyễn Khắc Nam Khánh** | Nghiên cứu & Báo cáo | Tìm hiểu cơ sở lý thuyết, tổng hợp tài liệu (5 bài báo), viết báo cáo khoa học & đề cương | 0–1, 5 |
| **Vũ Đình An** | Nghiên cứu & Báo cáo | Tìm hiểu cơ sở lý thuyết, tổng hợp tài liệu, hỗ trợ viết báo cáo, xây dựng phần luật y khoa Tầng 2 (tham vấn) | 0–1, 2, 5 |

> Quy tắc chung: mọi khối lượng **code** tập trung về Nguyễn Đức Cảnh; Khánh & An tập trung **nghiên cứu lý thuyết, dữ liệu, tài liệu và viết báo cáo**, đảm bảo không trùng lặp công việc.

---

## 4. Nguồn dữ liệu khả thi

| Nguồn | Loại | Cân nhắc |
|-------|------|----------|
| **Dữ liệu tự thu thập** | Chỉ số cơ thể định kỳ (huyết áp, nhịp tim, đường huyết, cân nặng...) | Đơn giản, cá nhân hóa thật; mẫu nhỏ |
| **MIMIC-IV / MIMIC-III** | Vital signs, lab (ICU) | Phổ biến, cần đăng ký PhysioNet, đạo đức |
| **NHANES** | Chuỗi dọc sức khỏe cộng đồng | Public, nhưng ít dạng chuỗi dọc chi tiết |
| **UK Biobank** | Đủ loại, chuẩn OMOP | Chỉ giả lập/tham khảo, khó truy cập |
| **Dữ liệu tổng hợp (simulator)** | AI sinh chỉ số theo mô hình sinh lý | **Khuyến nghị** cho giai đoạn phát triển ban đầu |

---

## 5. Rủi ro & Giảm thiểu

| Rủi ro | Mức độ | Giải pháp |
|--------|--------|-----------|
| Thiếu dữ liệu thực, mẫu nhỏ | Cao | Simulator sinh dữ liệu (bài học Delphi: AUROC 0,74 trên dữ liệu sinh); data augmentation |
| Chuỗi không đều thời gian, nhiều missing | Cao | Nội suy + chỉ báo missing; chọn cửa sổ (7/30/90 ngày) phù hợp |
| Luật y khoa thiếu/thay đổi | Trung bình | Cấu trúc tri thức JSON dễ chỉnh sửa; tham vấn bác sĩ; versioning |
| Quá khớp mô hình ML trên mẫu nhỏ | Trung bình | Regularization, CV, ưu tiên mô hình đơn giản + luật |
| Vấn đề đạo đức/quyền riêng tư | Trung bình | Ẩn danh, mã hóa, tuân thủ GDPR/HIPAA, không lưu nhận dạng |
| Hệ thống bị coi là "chẩn đoán" | Cao (pháp lý) | Dán nhãn rõ "hỗ trợ quyết định", disclaimer, giới hạn phạm vi |

---

## 6. Tiêu chí nghiệm thu (Deliverables & Acceptance)

1. Pipeline dữ liệu chạy tự động: `make pipeline` hoặc `python -m src.main --input data.csv --output report/`.
2. Báo cáo đánh giá 3 tầng + benchmark mô hình (`docs/05_benchmark.md` — sẽ sinh ra).
3. API hoạt động, trả JSON đúng schema 3 thành phần đầu ra.
4. Báo cáo cuối kỳ đạt yêu cầu đề cương, có đối chiếu với 5 bài báo đã tổng hợp.

---

## 7. Công nghệ đề xuất

- **Ngôn ngữ:** Python 3.11+
- **Xử lý dữ liệu:** pandas, numpy, scipy
- **ML:** scikit-learn (Isolation Forest, SHAP), lightgbm, xgboost
- **Deep learning (đối chứng):** PyTorch + LSTM/GRU
- **API/Báo cáo:** FastAPI, jinja2, reportlab/markdown
- **Quản lý dự án:** Git, requirements.txt/pyproject.toml
