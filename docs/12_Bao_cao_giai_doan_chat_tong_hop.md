# 12. Báo cáo giai đoạn: trợ lý chat tích lũy dữ liệu + đánh giá đa mô hình

> Tôi ghi lại những gì tôi vừa làm và đo được trong giai đoạn này: hoàn thiện
> giao diện trò chuyện với bệnh nhân, đưa báo cáo rủi ro lên kênh chat, gắn
> luận giải đa mô hình ML ngay trong hội thoại, và chạy lại benchmark trên bộ
> dữ liệu NHANES mở rộng. Tôi cố gắng nói đúng những gì đo được, nêu thẳng giới
> hạn, không thổi phồng — đúng tinh thần các báo cáo trước.

---

## 1. Tóm tắt những gì tôi đã làm

| Hạng mục | Nội dung |
|---|---|
| Trợ lý chat | UI `/chat` mới: header gradient, sidebar "Cấu hình suy luận" chọn model, bong bóng hội thoại, khối luận giải ML gắn trong chat |
| Đánh giá trong chat | Lệnh `trạng thái` / `báo cáo` / `xóa dữ liệu`, ghi nhận nhật ký theo ngày, đủ 7 ngày → BÁO CÁO ĐẦY ĐỦ (phân tích chuỗi thời gian cá nhân hóa) |
| Luận giải đa mô hình | Mỗi mô hình ML cho **điểm tổng hợp 3 tầng riêng** (thống kê + tri thức y khoa + model đó + xu hướng) để so sánh mức khác biệt; hiển thị tiếng Việt, gom chi tiết vào thẻ ấn/xem `<details>` |
| Data completeness | Đánh dấu mức độ đầy đủ dữ liệu của từng ca, theo chế độ chẩn đoán; nhấn mạnh đây là độ bao phủ, không phải mức rủi ro |
| Benchmark mới | Chạy lại 6 mô hình trên `nhanes_merged.csv` (3 chu kỳ NHANES), kèm `data_completeness` trong tổng hợp |
| Khác | Bỏ tích hợp LLM local (chậm ~90s/phản hồi, sai nội dung), gỡ sạch code; fix lỗi explain (NaN → impute median trước khi predict) |

## 2. Kết quả benchmark mở rộng (lần 2)

| Thành phần | Giá trị |
|---|---|
| Dataset | `data/datasets/nhanes_merged.csv` (NHANES 2015–2016, 2017–2018, 2019–2020) |
| Số mẫu | 16.314, positive = 7.991 (~49%) |
| Đặc trưng | 7 chỉ số lâm sàng: systolic_bp, diastolic_bp, heart_rate, glucose_fasting, hba1c, creatinine, bmi |
| Split | 70 / 15 / 15, test khóa lại, 5 seed (42/52/62/72/82) |
| Preprocessing | Imputer median fit trên train (tránh data leakage) |

| Mô hình | Họ | ROC-AUC | PR-AUC | Accuracy | F1 | Brier |
|---|---|---|---|---|---|---|
| **XGBoost** | Gradient boosting | **0.9356±0.0028** | 0.949 | 0.871±0.005 | 0.856±0.006 | 0.091±0.002 |
| **LightGBM** | Gradient boosting | 0.9349±0.0031 | 0.949 | 0.871±0.004 | 0.857±0.005 | 0.092±0.003 |
| **Random Forest** | Tree ensemble | 0.9338±0.0038 | 0.947 | 0.865±0.006 | 0.847±0.007 | 0.096±0.003 |
| **FT-Transformer** | Transformer tabular | 0.9257±0.0043 | 0.941 | 0.856±0.007 | 0.841±0.011 | 0.103±0.004 |
| **MLP** | Neural network | 0.8975±0.0123 | 0.912 | 0.827±0.015 | 0.816±0.016 | 0.129±0.010 |
| **Logistic Regression** | Baseline tuyến tính | 0.8844±0.0078 | 0.896 | 0.780±0.010 | 0.768±0.011 | 0.138±0.005 |

Số liệu chi tiết từ `experiments/summary.json`. XGB/LGBM/RF vẫn gần như tương
đương như lần 1; mở rộng dữ liệu giúp giảm độ lệch chuẩn đáng kể (AUC ±0.003 so
với ±0.005–0.008 trước). MLP vẫn tụt sau (cần tinh chỉnh hoặc không đủ dữ liệu
cho mạng nơ-ron).

### Mức độ đầy đủ dữ liệu (data_completeness)

- Confidence tổng thể: **TRUNG_BINH — một phần chỉ số bị khuyết, đã impute median**.
- `glucose_fasting`: **MISSING_CAO (52%)** — đây là điểm yếu của `nhanes_merged`
  (glucose đói chỉ đo trên nhóm con), cần lưu ý khi đọc kết quả của mô hình.
- Các chỉ số còn lại: đầy đủ hoặc thiếu 1 phần (5–8%).

Tôi đã đưa `data_completeness` vào `build_summary()` và giao diện `/benchmark`
hiển thị các flag này để người đọc không ngộ nhận chất lượng dữ liệu.

## 3. Trợ lý chat và đánh giá đa mô hình

### 3.1 Dòng chảy hội thoại

- Gửi nhật ký dạng "Huyết áp 135/85, nhịp tim 80" → parser trích xuất, tích lũy
  theo `patient_id`, mỗi ngày một lần đo (bỏ trùng lặp).
- Lệnh `trạng thái` → đếm ngày đo; `báo cáo` → chạy pipeline 3 tầng; `xóa dữ
  liệu` → reset bệnh nhân.
- Đủ 7 ngày → BÁO CÁO ĐẦY ĐỦ có phân tích cá nhân hóa (đường cơ sở, z-score,
  xu hướng); chưa đủ → BÁO CÁO SƠ BỘ theo luật lâm sàng, đánh dấu rõ THIẾU DỮ LIỆU.

### 3.2 Điểm tổng hợp theo từng mô hình (không fusion)

Tôi thử nghiệm phương án gộp (fusion) tất cả mô hình thành một điểm ML duy
nhất, nhưng anh đề xuất ngược lại: **mỗi mô hình giữ một điểm cuối riêng** để
thấy model nào đẩy mức rủi ro lên/cao hơn. Tôi đã làm theo — với mỗi mô hình:

```
điểm cuối(model) = stat×0.30 + knowledge×0.35 + ml(model)×0.25 + trend×0.10
```

trong đó `ml(model)` là xác suất nguy cơ riêng của mô hình đó, rồi áp ngưỡng
xếp loại (THẤP < 0.33 · TRUNG BÌNH 0.33–0.66 · CAO ≥ 0.66) và an toàn lâm sàng
(nâng sàn 0.50 khi có luật nghiêm trọng). Kết quả cho ca P9 (HA 150/95):

| Mô hình | Điểm ML | Điểm cuối (3 tầng) | Mức |
|---|---|---|---|
| Logistic Regression | 0.888 | 0.502 | TRUNG BÌNH |
| MLP | 0.917 | 0.509 | TRUNG BÌNH |
| Random Forest | 0.992 | 0.528 | TRUNG BÌNH |
| LightGBM | 1.000 | 0.530 | TRUNG BÌNH |
| XGBoost | 1.000 | 0.530 | TRUNG BÌNH |
| FT-Transformer | 1.000 | 0.530 | TRUNG BÌNH |

Điểm thú vị: mọi mô hình đều cho ML "CAO" nhưng khi tổng hợp 3 tầng, mức rủi
ro cuối đều là TRUNG BÌNH — vì tri thức y khoa (0.35 trọng số) không đạt ngưỡng
CAO. Báo cáo hiển thị cả hai số (ML riêng + điểm cuối) để người dùng thấy sự
khác biệt.

### 3.3 Giao diện

- Khối luận giải gắn trong chat: mỗi mô hình một thẻ, có vạch ngăn `<hr>`, mức
  rủi ro hiển thị tiếng Việt (TRUNG BÌNH / THẤP / CAO), không dùng mã.
- Thông tin chi tiết (công thức, chuẩn xếp loại, an toàn lâm sàng, điểm từng
  mô hình) gom vào thẻ `<details>` — ấn mới hiện, giữ báo cáo gọn.
- Render markdown bằng thư viện `marked` (CDN + fallback thủ công khi offline);
  sửa lỗi khoảng cách dòng không đều do `white-space: pre-wrap` cộng với
  `<br>/<p>` sinh ra bởi marked.

## 4. Fix quan trọng: giải thích XGBoost cho ra "không có gì tăng"

Lỗi: khối luận giải XGBoost cho điểm 1.0 CAO nhưng mọi đóng góp đều 0.000.

Nguyên nhân: `explain_patient` truyền **NaN** cho các đặc trưng thiếu khi
predict, trong khi lúc train đã **impute median** (`SimpleImputer`). Mô hình
chưa từng thấy NaN nên trả xác suất gần như không đổi cho mọi input → delta ≈ 0.

Fix: impute các đặc trưng thiếu bằng median **trước khi predict**, khớp với
luồng train. Kết quả ca P9 sau fix: XGBoost R_CV_01 = **+0.253** (trước 0.000),
Random Forest +0.421, có ý nghĩa lâm sàng.

## 5. Bỏ tích hợp LLM local

Tôi thử chạy Qwen2.5-0.5B qua llama.cpp trên máy (i3-4005U, 2 lõi, không GPU):

- Thời gian phản hồi **~90 giây** cho 60 token — không dùng được cho chat.
- Chất lượng kém: trả lời sai kiến thức y khoa ("huyết áp 150/95 không nguy
  hiểm"), thậm chí bịa nội dung ung thư.

Anh quyết định gỡ bỏ hoàn toàn — tôi đã xóa `models/llm/`, `scripts/llm_server.sh`,
`src/chat/llm_client.py` và toàn bộ code gọi LLM trong agent. Hướng đi tương
lai hợp lý hơn: gọi một API model bên ngoài (OpenAI-compatible) thay vì chạy
local. Hệ thống vẫn giữ sẵn `src/ingest/llm_extractor.py` (dùng API ngoài cho
trích xuất file khi regex không đủ mạnh, có fallback regex an toàn).

## 6. Dữ liệu thử nghiệm

Tôi đã tạo bệnh nhân demo `P1000` — 7 ngày liên tục (2026-08-10 → 2026-08-16),
35 bản ghi gồm systolic_bp, diastolic_bp, heart_rate, glucose_fasting, bmi.
Đủ để test BÁO CÁO ĐẦY ĐỦ có phân tích cá nhân hóa (đường cơ sở, xu hướng ổn
định, các hệ cơ quan: Chuyển hóa + Hệ tim mạch).

File mẫu để test upload trong `data/reports/`: `report_huyet_ap.pdf`,
`report_than.docx`, `report_tieu_duong.docx`, `report_khoe.docx`. Lưu ý mỗi file
là một lần đo → chỉ ra BÁO CÁO SƠ BỘ, chưa đủ 7 ngày.

## 7. Giới hạn còn lại

- `glucose_fasting` thiếu 52% trong benchmark — kết quả ML trên chỉ số này cần
  đọc thận trọng.
- Các model vẫn là điểm tham khảo bổ sung; kết luận cuối do bác sĩ xác nhận
  (disclaimer luôn hiển thị trong báo cáo).
- Máy hiện tại không chạy được LLM local; nếu muốn có câu trả lời tự nhiên cho
  chat, cần tích hợp API ngoài (chưa triển khai).
- Ca `P1000` mới chỉ 5/10 chỉ số (thiếu creatinine, egfr, glucose, hba1c, spo2)
  — đủ để minh họa, chưa phải ca đầy đủ lý tưởng.
