# 19. Báo cáo tiến độ giai đoạn P2 — Đã giải quyết và giới hạn còn lại

> Ngày cập nhật: 24/08/2026 · Trạng thái: P2.1 ✅, P2.2 ✅, P2 kickoff (LMF + temporal validation) ✅
> Tài liệu này là bản tổng kết một trang: những gì **đã giải quyết xong**, kết quả
> định lượng, và **giới hạn còn đứng lại** kèm điều kiện khắc phục.
> Chi tiết kỹ thuật từng việc nằm ở docs/15–docs/18.

---

## 1. Bức tranh tổng thể

Hệ thống đi qua ba giai đoạn chuẩn hóa:

| Giai đoạn | Nội dung chính | Kết quả |
|---|---|---|
| P0 | Sửa lỗi nền tảng (KB contract, pipeline, UI chat) | ✅ Hoàn tất (docs/15) |
| P1 | Calibration isotonic vào production, benchmark research, governance, seed demo, e2e | ✅ Hoàn tất (docs/16, docs/17) |
| P2 | Dữ liệu dọc thời gian: NHANES-LMF, validation theo thời gian, kiểm định khuyết dữ liệu, đồng bộ hồ sơ, UI v2 | ✅ Phần làm được bằng máy đã xong; phần còn lại chờ quyền truy cập dữ liệu / tích lũy dữ liệu thực |

## 2. Những vấn đề ĐÃ GIẢI QUYẾT XONG

### 2.1 Danh mục theo docs/16

| Vấn đề | Kết quả cuối | Bằng chứng |
|---|---|---|
| T1 — Hiệu chỉnh xác suất | Isotonic nội bộ ECE 0.004→0.000; đã cài vào production fusion (`load_ml_calibrator`) | K1, `/api/evidence/ml` |
| T2/T5/T6 — Độ bền vững baseline/trọng số | ΔAUC ≤ 0.006 mọi biến cố; nhãn thay thế không phải nguyên nhân duy nhất | K2–K4 |
| T7 — Chưa có outcome tương lai | **Đã có:** train chu kỳ 2015-16 → test chu kỳ 2017-18 trên 9.821 người liên kết tử vong; LR AUC 0.8209 (temporal) vs 0.8411 (random); Harrell C-index 0.8217; lead time trung vị 9 tháng cho nhóm nguy cơ cao nhất | EXP-TEMPORAL-LMF |
| T8 — Nhãn cắt ngang tự chứng minh | Baseline nhãn cắt ngang cùng tập test chỉ đạt AUC 0.573–0.628, thấp hơn rõ so với 0.8209 của nhánh tử vong → tín hiệu mô hình không chỉ "học lại nhãn" | EXP-TEMPORAL-LMF |
| T10 — Chưa xác nhận ngoài | Ở mức **cohort-level**: temporal split = hai cuộc khảo sát độc lập (khác dân mẫu, khác thời điểm). Vẫn còn thiếu external-by-geography (xem §3) | EXP-TEMPORAL-LMF |
| T11 — Khuyết glucose 52% | Kiểm định trực tiếp: LightGBM/XGBoost ổn định (\|ΔAUC\| ≈ 0.005); LR lệch +0.0158 về phía complete-case → giữ impute median cho production, ghi chú khi trích baseline tuyến tính | COMPLETE-CASE-CHECK |
| Governance Tier-2 | Luồng draft→review→approved→active đầy đủ trên UI, audit trail, luật draft không chấm production | e2e mục 5 |
| Xuất dữ liệu | Bản ghi cá nhân và audit trail xuất CSV ngay trên UI (BOM UTF-8 mở Excel đúng tiếng Việt) | UI v2 24/08 |

### 2.2 Kết quả định lượng then chốt của P2

**EXP-TEMPORAL-LMF** (train 2015-16 n=5.048 / test 2017-18 n=4.773; tử vong ≤12 tháng):

| Đại lượng | LR | LGBM |
|---|---|---|
| AUC temporal test | **0.8209** | 0.7709 |
| AUC random test (đối chứng) | 0.8411 | 0.7810 |
| C-index ≤60 tháng | **0.8217** | 0.7763 |
| Lead time trung vị (top quintile bắt tử vong ≤24 tháng) | 53/94 ca, trung vị **9 tháng** | — |
| Nhãn cắt ngang (label_htn / label_dm) cùng test | 0.6281 / 0.5731 | — |

**COMPLETE-CASE-CHECK** (cùng split + seed, so impute vs bỏ mẫu thiếu):

| Model | ΔAUC | Kết luận |
|---|---|---|
| LightGBM (production) | −0.0059 | Ổn định, giữ impute |
| XGBoost | −0.0045 | Ổn định |
| Logistic Regression | +0.0158 | Lệch có hệ thống — ghi chú phương sai |

## 3. Giới hạn CÒN ĐỨNG LẠI (và điều kiện khắc phục)

| # | Giới hạn | Trạng thái | Điều kiện khắc phục |
|---|---|---|---|
| G1 | External-by-geography chưa có (T10 chỉ xong cohort-level) | ◐ Chờ thủ tục | KNHANES: nộp đề cương KDCA RDC; MIMIC-IV: tài khoản PhysioNet + chứng chỉ CITI + ký DUA (P2.3/P2.7) |
| G2 | Public mortality file bị nhiễu loạn: UCOD chỉ còn 1/2/10 → tử vong CV = tim mạch thuần, không tách đột quỵ/ĐTDĐ | ● Đặc tính nguồn, không sửa được | Chỉ giải quyết bằng bản restricted access của NCHS |
| G3 | Outcome = tử vong, không phải thời điểm khởi phát bệnh | ● Đặc tính nguồn | Cần EHR dọc (MIMIC-IV/EHRSHOT) mới đo được onset |
| G4 | Mẫu test temporal chỉ 67 biến cố → khoảng tin cậy AUC rộng (~±0.02) | ◐ | Gom thêm chu kỳ 1999–2014 (script đã hỗ trợ) hoặc dataset lớn hơn |
| G5 | Dân mẫu NHANES là người Mỹ ≠ Việt Nam | ◐ | Trùng G1: cần KNHANES hoặc dữ liệu bệnh viện VN |
| G6 | Lead time theo ngày chưa chứng minh được (P2.4/P2.5) | ⏳ Chờ dữ liệu thật | ≥50 người dùng nhập chỉ số × ≥30 ngày qua kênh nhập hệ thống |
| G7 | Mô hình chuỗi thời gian (LSTM/Transformer) bị hoãn (P2.6) | ⏳ Chờ dữ liệu thật | Ngưỡng ≥500 người × ≥60 ngày (docs/16 §3.3) |
| G8 | Production model huấn luyện trên nhãn cắt ngang NHANES (AUC 0.9356) — hợp lệ cho phân tầng hiện tại, KHÔNG tuyên bố dự báo | ● Thiết kế hiện tại | Khi có dữ liệu dọc sẽ tinh chỉnh lại với nhãn tương lai |
| G9 | TRIPOD-AI checklist chưa lập hồ sơ đầy đủ | ◐ Việc viết | Không phụ thuộc dữ liệu — có thể làm bất cứ lúc nào |

## 4. UI v2 (24/08/2026)

- **Dark mode**: nút 🌙/☀️ ở header, toàn bộ màu chuyển sang CSS variables, lưu localStorage.
- **Biểu đồ xu hướng SVG** trong tab Bản ghi: mỗi chỉ số một đường (chuẩn hóa min–max riêng), tooltip legend hiển thị giá trị cuối + khoảng dao động, tự co giãn mobile.
- **Xuất CSV**: bảng bản ghi (định dạng dài patient_id,date,metric,value,unit) và audit trail (≤1000 dòng gần nhất).
- **Chế độ chuyên khoa đầy đủ** ở tab Đánh giá: htn/dm/cv/ckd/met/resp.
- Trang Benchmark thêm panel **Complete-case check** bên cạnh panel Validation theo thời gian; cả hai lấy từ `/api/benchmark/research` (trường `temporal_validation`, `complete_case`).

## 5. Kiểm thử end-to-end

Bộ kiểm thử chính thức chuyển vào repo: `scripts/e2e_test.py` — 35 kiểm tra gồm:
trang tĩnh (app/benchmark/rules), đánh giá 5 bệnh nhân seeded, CRUD bản ghi,
toàn bộ luồng governance (chặn nhảy cóc trạng thái, version bump, draft không
chấm production, audit trail), markdown nl2br, evidence ML raw+isotonic,
research payload (K2–K4 + temporal_validation + complete_case).

```
python3 -m uvicorn src.api:app --port 8000   # terminal 1
python3 scripts/e2e_test.py                  # terminal 2 → 35 PASS / 0 FAIL (24/08/2026)
```

## 6. Việc còn lại trong P2 (tất cả chờ yếu tố ngoài)

1. **P2.3** — Nộp đề cương KDCA RDC (KNHANES) hoặc chuẩn bị hồ sơ PhysioNet/CITI/DUA (MIMIC-IV, P2.7): thủ tục do con người thực hiện.
2. **P2.4–P2.6** — Chờ tích lũy dữ liệu dọc theo ngày từ người dùng thật qua hệ thống.
3. Có thể làm ngay nếu muốn: lập checklist TRIPOD-AI (G9).

## 7. Giới hạn cốt lõi của NCKH sinh viên và hướng xử lý khả thi

Sau khi phần kỹ thuật trong tầm tay đã hoàn tất (mục 2), ba giới hạn còn lại
của đề tài đều thuộc loại **không tạo ra được bằng code**: người dùng thật,
chuyên môn y khoa thật, và quyền truy cập dữ liệu thể chế. Đây là ranh giới
tài nguyên, không phải ranh giới phương pháp — và đã được khai báo trung thực
ở bảng G1–G9 thay vì giấu. Mỗi giới hạn có phương án xử lý khả thi trong phạm
vi sinh viên:

| Còn thiếu | Vì sao khó với NCKH sinh viên | Hướng xử lý khả thi |
|---|---|---|
| Người dùng thật (P2.4–2.6 cần ≥50 người × ≥30 ngày) | Không ai cung cấp dữ liệu sức khỏe hằng ngày cho một đồ án | Pilot nội bộ: bạn bè/người thân tự nhập chỉ số của chính mình 2–4 tuần qua kênh nhập sẵn có; công bố rõ là pilot phi lâm sàng, đủ để minh họa lead time nhỏ |
| Bác sĩ thật (duyệt luật, ký governance) | Luồng draft→review→approved hiện do tên tự đặt duyệt | Mời một bác sĩ (giảng viên hướng dẫn, hoặc quen biết cá nhân) làm reviewer cho 9 luật hiện có; một lần phê duyệt thật đưa governance từ "mô phỏng" thành "có chuyên môn tham gia" |
| Dữ liệu bệnh viện/EHR (MIMIC-IV, KNHANES RDC) | Thủ tục hành chính; RDC cần đơn vị bảo lãnh | MIMIC-IV **không yêu cầu bác sĩ**: tự đăng ký PhysioNet + hoàn thành khóa CITI online miễn phí (~3–5 giờ) + ký DUA là nhận dữ liệu ~40 nghìn bệnh nhân ICU |

Việc rẻ nhất mà nâng giá trị đề tài ngay, không phụ thuộc ai:
**(1)** hồ sơ CITI + xin quyền MIMIC-IV (G1/P2.7 — hướng dẫn từng bước đã
kiểm minh: docs/18 §7), **(2)** checklist TRIPOD-AI (G9).
Cả hai chỉ tiêu thời gian cá nhân.

---

*Báo cáo tổng kết phiên làm việc ngày 24/08/2026. Mọi con số truy ngược được tới
evidence package tương ứng trong thư mục `experiments/`. Hệ thống hỗ trợ phân tầng
nguy cơ và luận giải bằng chứng; kết luận lâm sàng cuối cùng thuộc về bác sĩ.*
