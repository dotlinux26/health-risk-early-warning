# 13. Nhận xét của giảng viên hướng dẫn và đối sách của nhóm nghiên cứu

> Tài liệu ghi lại các nhận xét của giảng viên hướng dẫn sau buổi báo cáo tiến độ,
> đối chiếu từng nhận xét với hiện trạng mã nguồn của hệ thống, và liệt kê giải
> pháp tương ứng: phần đã có trong hệ thống, phần đang triển khai, phần còn phải
> làm. Mỗi mục truy xuất được tới file code cụ thể.

---

## 1. Tóm tắt nhận xét của giảng viên

### 1.1 Hạn chế của mô hình đề xuất

| # | Nhận xét |
|---|---|
| L1 | Khái niệm **cảnh báo sớm** cần được giới hạn phù hợp với bằng chứng thực nghiệm. Nếu dữ liệu chủ yếu là dữ liệu cắt ngang, hệ thống có cơ sở cho **phân tầng nguy cơ** nhưng chưa đủ chứng minh **dự báo biến cố tương lai** |
| L2 | Tầng 1 phụ thuộc vào việc có đủ **dữ liệu lịch sử cá nhân**. Khi số lượng quan sát ít, baseline cá nhân chưa ổn định → giảm độ tin cậy phát hiện bất thường |
| L3 | Tầng 2 rule engine phụ thuộc vào **chất lượng nguồn tri thức**, quy trình chuyển đổi hướng dẫn y khoa thành luật và **việc xác nhận chuyên gia** |
| L4 | Cơ chế tổng hợp điểm nguy cơ cần chứng minh rõ hơn về **trọng số**, ngưỡng phân loại Low/Medium/High và **hiệu chỉnh xác suất** |
| L5 | LightGBM phù hợp dữ liệu bảng nhưng **không mô hình hóa trực tiếp quan hệ thời gian dài** như các mô hình chuỗi thời gian |
| L6 | Cần kiểm soát nguy cơ **rò rỉ dữ liệu** và **rò rỉ nhãn** trong xây dựng đặc trưng và đánh giá mô hình |

### 1.2 Giải pháp nâng cao chất lượng được đề xuất

| # | Giải pháp |
|---|---|
| S1 | Bổ sung **dữ liệu dọc theo thời gian** để đánh giá khả năng cảnh báo trước biến cố |
| S2 | Chuẩn hóa pipeline dữ liệu: kiểm soát missing values, outlier, chuẩn hóa đặc trưng, **tách dữ liệu theo bệnh nhân** |
| S3 | Hoàn thiện validation: **xác nhận nội bộ → xác nhận theo thời gian → xác nhận ngoài** (nếu có điều kiện) |
| S4 | Bổ sung đánh giá **calibration, Brier score, false positive, false negative, lead time** cảnh báo |
| S5 | Chuẩn hóa cơ sở tri thức: **quản lý phiên bản luật, truy xuất nguồn gốc, chuyên gia phê duyệt, audit trail** |
| S6 | Duy trì **kiến trúc hybrid** giữa tri thức y khoa và ML thay vì thay thế hoàn toàn bằng mô hình hộp đen |

### 1.3 Hạn chế của bộ dữ liệu thực nghiệm

- NHANES, Cleveland Heart Disease, Pima Diabetes là dữ liệu y tế **dạng bảng**:
  hữu ích cho đánh giá ML, nhưng hạn chế khi chứng minh bài toán cảnh báo sớm vì
  thiếu chuỗi thời gian liên tục, thời điểm phát sinh biến cố, và quan hệ nhân quả
  giữa dự báo với kết quả lâm sàng.
- Nguy cơ hạn chế khả năng **khái quát** sang quần thể bệnh nhân thực tế khác.
- Cần đánh giá kỹ **tính mất cân bằng lớp** của các nhóm nguy cơ.
- Cần làm rõ **phương pháp tạo nhãn** và tính độc lập giữa nhãn với biến đầu vào.
- Chưa thay thế được **xác nhận lâm sàng**.

### 1.4 Đối sánh với các mô hình liên quan

| Nhóm mô hình | Ưu thế | Chi phí / hạn chế |
|---|---|---|
| LightGBM (boosting) | Dữ liệu bảng, huấn luyện rẻ, giải thích tốt qua SHAP | Không mô hình hóa quan hệ thời gian dài |
| LSTM/GRU | Chuỗi thời gian khi dữ liệu đủ lớn | Cần nhiều dữ liệu, khó giải thích hơn |
| Transformer thời gian / foundation model EHR | Mô hình hóa phức tạp | Yêu cầu hạ tầng và dữ liệu lớn |

---

## 2. Đối sách từng nhận xét — phần đã có trong hệ thống

### 2.1 Baseline cá nhân, không so quần thể (L2, S1)

Tầng 1 được thiết kế đúng nguyên tắc **so sánh người dùng với lịch sử của chính họ**:

- `build_baseline` (`src/data/preprocess.py:61`) tính đường cơ sở μ, σ bằng **cửa
  sổ trượt 90 ngày hướng về quá khứ** (`min_periods=5`) — không nhìn tương lai,
  mỗi bệnh nhân một đường cơ sở riêng.
- Z-Score (`src/tier1_anomaly/zscore.py`) so giá trị hiện tại với `{metric}_base_mean`
  và `{metric}_base_std` **cá nhân** (dòng 27–34), đầu ra gồm: độ lệch chuẩn hoá
  (z), mức tăng/giảm tuyệt đối, xu hướng nửa đầu/nửa sau chuỗi (dòng 46–47),
  `baseline_mean` — tức **bản ghi bất thường kèm độ lệch bao nhiêu và biến đổi ra
  sao theo thời gian**, đúng cấu trúc giảng viên nêu.
- Khi quan sát ít hơn `min_periods=5`, baseline là NaN → z-score **không được
  tính**, thay vào đó hệ thống đánh dấu thiếu dữ liệu; chế độ chat yêu cầu tối
  thiểu `min_points=7` ngày đo mới đưa báo cáo đầy đủ (`src/config.py`,
  `src/chat/store.py`). Hệ thống **từ chối kết luận khi baseline chưa ổn định**
  thay vì đưa ra cảnh báo nhiễu.

### 2.2 Kiểm soát rò rỉ dữ liệu / rò rỉ nhãn (L6, S2)

- Imputer median **fit trên tập train** rồi transform sang val/test — tránh leakage
  tiền xử lý (`src/experiments/protocol.py`, `experiments/README.md`).
- Test set **khóa lại**: 70/15/15, 5 seed, mọi model cùng split để so paired.
- Dataset NHANES mỗi dòng là một người → tách theo bệnh nhân mặc nhiên, không có
  hai dòng cùng người ở hai tập.
- **Rò rỉ nhãn qua thuốc** đã được nhận diện: nhãn NHANES tạo từ `(has_bp & has_lab)
  | has_meds` (`scripts/build_nhanes_dataset.py:193`); người đang dùng thuốc hạ áp
  có thể có huyết áp đo thấp → biến đầu vào chịu ảnh hưởng của tình trạng bệnh.
  Đây là giới hạn được ghi nhận và sẽ phân tích độ nhạy (mục 4).

### 2.3 Minh bạch trọng số và ngưỡng (L4)

Công thức tổng hợp điểm được công khai ngay trong báo cáo (mục "Cách tính điểm"):

```
total = stat×0.30 + knowledge×0.35 + ml×0.25 + trend×0.10   (src/config.py:25-29)
```

- Ngưỡng xếp loại: THẤP < 0.33 · TRUNG BÌNH 0.33–0.66 · CAO ≥ 0.66.
- An toàn lâm sàng: có luật severity ≥ 0.7 → điểm sàn nâng lên 0.50
  (`src/tier3_risk/scoring.py:295-301`) — mô hình không thể "gọi là an toàn"
  khi luật nghiêm trọng kích hoạt.
- **Calibration và Brier score đã có sẵn** trong benchmark: `brier_score_loss`
  (`protocol.py:96`) và đường calibration 10 bins cho từng mô hình
  (`runner.py:117`, xuất `calibration_curve.png`). Hiệu chỉnh hậu kỳ (isotonic/
  Platt) chưa có — nằm ở mục 4.

### 2.4 Chất lượng nguồn tri thức và quy trình luật (L3)

- Mỗi luật **bắt buộc** có `source_url`, `severity` (0–1), `specialty`, `modes`;
  validate chặt khi lưu (sai cấu trúc/toán tử/độ nặng/link đều bị từ chối —
  `src/tier2_knowledge/rules.py:71-122`).
- Trang `/rules` cho bác sĩ xem/thêm/sửa/xóa luật và chỉ số **không cần sửa code** —
  đây chính là điểm cắm cho quy trình chuyên gia phê duyệt.
- Hiện trạng: 9 luật, 10 chỉ số, nguồn tham chiếu là hướng dẫn ESH 2018, ADA,
  KDIGO… kèm link gốc. **Chưa có**: phiên bản luật, audit trail (mục 4).

### 2.5 Kiến trúc hybrid (S6)

Hệ thống giữ luật + thống kê làm lõi quyết định, ML chỉ là một nguồn điểm bổ sung:
luật luôn chạy được kể cả khi chưa có dữ liệu huấn luyện; mọi suy luận ML đều gắn
disclaimer "tham khảo bổ sung, kết luận cuối do bác sĩ xác nhận". Định vị này khớp
khuyến nghị của giảng viên và đã được lập luận trong `docs/07`.

### 2.6 Giới hạn phạm vi "cảnh báo sớm" (L1)

Đề tài định vị là **khung hỗ trợ quyết định lâm sàng** thực hiện phân tầng nguy cơ
trên dữ liệu sẵn có, **không tuyên bố dự báo biến cố tương lai** (`docs/07`). Với
bộ dữ liệu cắt ngang, mọi kết quả AUC/Brier được diễn giải là năng lực **phân tầng**,
không phải lead-time dự báo. Việc mở rộng sang dự báo dọc thời gian phụ thuộc dữ
liệu mới (mục 3, 4).

---

## 3. Đối sách — phần đang triển khai

| Giải pháp của giảng viên | Hiện trạng triển khai |
|---|---|
| S1 — Dữ liệu dọc theo thời gian | Kênh chat tích lũy nhật ký **theo ngày cho từng bệnh nhân** (`data/chat/*.jsonl`, dedup theo cặp chỉ số + ngày); chọn ngày đo + kéo-thả file giúp nhập dữ liệu quá khứ đúng ngày (`docs/12` mục 6); ca demo P1000 7 ngày chạy được phân tích cá nhân hóa đầy đủ |
| S4 — Calibration/Brier/FP/FN | Brier + calibration curve + precision/recall/specificity đã có trong evidence package từng lần chạy; FP/FN chi tiết theo ngưỡng vận hành sẽ được bổ sung vào báo cáo |
| S2 — Pipeline chuẩn hóa | Missing values: impute median fit-train + flag mức thiếu (`MISSING_CAO` 52% glucose_fasting được công bố rõ trong summary); outlier: tầng 1 phát hiện bất thường; chuẩn hóa đặc trưng: theo protocol từng model |

---

## 4. Kế hoạch còn phải làm (ưu tiên)

1. **Hiệu chỉnh xác suất (S4)**: thêm isotonic/Platt scaling fit trên val set,
   báo cáo Brier trước/sau hiệu chỉnh cho từng model — chi phí thấp, làm được ngay
   trên dữ liệu hiện có.
2. **Phân tích nhãn và độ nhạy (L6)**: tài liệu hóa quy tắc tạo nhãn NHANES; chạy
   lại benchmark với nhãn bỏ thành phần `has_meds` để đo mức thay đổi AUC — trả lời
   trực tiếp câu hỏi độc lập nhãn/đầu vào.
3. **Mất cân bằng lớp**: công bố positive rate từng dataset (NHANES merged ~49%,
   Pima ~35%, Cleveland ~46%) và PR-AUC song song ROC-AUC (đã có PR-AUC trong summary).
4. **Validation theo thời gian (S3)**: khi có bộ dữ liệu dọc — tách train/test
   **theo mốc thời gian** (train quá khứ, test tương lai) và đo **lead time** giữa
   lúc hệ thống cảnh báo và thời điểm biến cố. Đây là điều kiện cần để chứng minh
   "cảnh báo sớm"; hiện chưa đủ dữ liệu nên chỉ đặt lộ trình.
5. **Quản trị tri thức (S5)**: thêm trường `version` + lịch sử thay đổi luật
   (audit trail: ai sửa, khi nào, nội dung cũ/mới), trạng thái "chờ duyệt / đã
   duyệt" cho từng luật trên `/rules`.
6. **Mô hình chuỗi thời gian (L5)**: giữ LightGBM làm model sản xuất trên dữ liệu
   bảng; đánh giá LSTM/GRU hoặc foundation model EHR (tầng 1 đã chừa sẵn điểm cắm
   Chronos/TimesFM) khi có dữ liệu dọc đủ lớn — theo đúng đối sánh ở mục 1.4.
7. **Xác nhận ngoài (S3)**: hợp tác cơ sở y tế nếu có điều kiện; chưa đặt deadline.

## 5. Công việc tài liệu phân công sau buổi báo cáo

- Chương 2 (lý thuyết): phụ trách Khánh — tổng quan học máy trên dữ liệu y tế dạng
  bảng, boosting, SHAP, calibration.
- Chương 3 (kiến trúc): phụ trách An — kiến trúc 3 tầng, rule engine, cơ chế tổng
  hợp điểm, luồng dữ liệu.
- Tra cứu bổ sung ≥ 7 bài báo liên quan đến hệ thống (nhóm foundation model EHR:
  Foresight, Delphi; nhóm chuỗi thời gian y tế; nhóm cảnh báo sớm trên EHR) để
  làm nền tham chiếu cho hai chương trên.

---

*Tài liệu ghi nhận phản hồi và đối sách. Các con số truy xuất từ `experiments/`,
quyết định kỹ thuật truy xuất từ mã nguồn nêu trong từng mục. Không thay thế
chẩn đoán của bác sĩ.*
