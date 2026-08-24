# 16. Trả lời các vấn đề mà tài liệu 13–15 chưa giải quyết được

> Tài liệu này chốt sổ từng mục "còn phải làm" trong `docs/13` (mục 4), `docs/14`
> (mục 4.1 bảng D1–D7 và mục 7 nhược điểm) và phần còn lại của lộ trình
> `docs/15`. Với mỗi vấn đề: trạng thái hiện tại, bằng chứng truy xuất được, và
> với những gì chưa làm được — trả lời thẳng câu hỏi *cần gì để làm* và *làm thế
> nào*. Sau hai đợt P0 (thí nghiệm nền tảng) và P1 (quản trị tri thức + giao
> diện biểu mẫu), danh sách việc chưa giải quyết chỉ còn thuộc nhóm **yêu cầu
> dữ liệu mới**, không còn mục nào nằm trong tầm tay dữ liệu hiện có mà bỏ trống.

---

## 1. Bảng tổng hợp trạng thái

| # | Vấn đề | Nguồn | Trạng thái | Nơi chứng minh |
|---|---|---|---|---|
| T1 | Hiệu chỉnh xác suất hậu kỳ (S4) | 13§4.1, 14§7.2 | **ĐÃ GIẢI QUYẾT** | `experiments/*/calibration.json`; production áp calibrator |
| T2 | Phân tích độ nhạy nhãn / rò rỉ nhãn (L6, D2) | 13§4.2 | **ĐÃ GIẢI QUYẾT** | `experiments/LABEL-SENSITIVITY/` |
| T3 | Quản trị tri thức: phiên bản, audit, phê duyệt (S5) | 13§4.5, 14§7.3 | **ĐÃ GIẢI QUYẾT** | `src/tier2_knowledge/governance.py`, tab Luật & Quản trị |
| T4 | Mất cân bằng lớp | 13§4.3 | ĐÃ XỬ LÝ | positive rate công bố trong summary; PR-AUC song song ROC-AUC |
| T5 | Ổn định baseline theo độ dài cửa sổ (L2) | 13 L2 | **ĐÃ GIẢI QUYẾT** | `experiments/BASELINE-STABILITY/`; UI cấm z-score < 7 ngày |
| T6 | Nhạy cảm trọng số fusion (L4) | 13 L4 | **ĐÃ GIẢI QUYẾT** | `experiments/WEIGHT-SENSITIVITY/`; đồng thuận ≥ 96% |
| T7 | Validation theo thời gian + lead time (S3, L1) | 13§4.4 | CHƯA — cần dữ liệu dọc | §3.1 dưới đây |
| T8 | Vòng lặp nhãn–đầu vào (D1) | 14§4.1 | ĐÃ ĐỊNH LƯỢNG; khắc phục căn bản cần nhãn biến cố | §3.2 |
| T9 | Mô hình chuỗi thời gian (L5) | 13§4.6, 14§7.4 | CHƯA — điểm cắm sẵn | §3.3 |
| T10 | Xác nhận ngoài (S3, D7) | 13§4.7 | CHƯA — cần dataset độc lập | §3.4 |
| T11 | glucose_fasting thiếu 52% (D3) | 14§4.1 | ĐÃ CÔNG BỐ; kiểm định thêm đề xuất ở §3.5 | summary MISSING_CAO |

---

## 2. Các vấn đề đã giải quyết — bằng chứng

### 2.1 Hiệu chỉnh xác suất (T1)

- Thí nghiệm: 6 model × 5 seed, calibrator (Platt/isotonic) fit trên validation,
  chọn theo Brier val; mọi model chọn **isotonic**. LGBM/XGB vốn đã cân bằng
  (ECE test ~1.5%); LR/MLP giảm rõ (ECE 5.1% → 1.8%). Evidence package:
  `experiments/EXP-ML-*-42/calibration.json` + `calibrator_isotonic.joblib`.
- **Production**: `_ml_score_for` (`src/core/pipeline.py`) nay nạp calibrator
  đi kèm model sản xuất và áp lên điểm trước khi đưa vào fusion — điểm ML trong
  tổng hợp có đúng nghĩa xác suất, không chỉ hiển thị cho đẹp. Calibrator lỗi
  thì fallback về điểm thô, không phá luồng đánh giá.
- Giao diện: panel Tầng 3 hiển thị cặp Raw vs Calibrated kèm cảnh báo "đầu ra
  chưa hiệu chỉnh không được diễn giải là xác suất bệnh".

### 2.2 Độ nhạy nhãn (T2)

Chạy lại benchmark trên cùng protocol với hai định nghĩa nhãn (có/không thành
phần thuốc HA): AUC nhãn B = 1.000 (LGBM/XGB) so với nhãn A = 0.935 → thay đổi
cách định nghĩa nhãn làm kết quả đổi hoàn toàn, xác nhận bằng thực nghiệm điều
docs 13 chỉ suy luận. Oracle (dùng nhãn B làm điểm xếp hạng nhãn A) đạt
0.815 ± 0.009 < model 0.935 → model học thêm tín hiệu thuốc từ chính mẫu đo,
không đơn thuần sao chép ngưỡng. Chi tiết: `docs/15` §5K-K2.

### 2.3 Quản trị tri thức (T3)

Luồng `Draft → Review → Approved → Active` (+ Rejected) với version tăng tự
động khi sửa nội dung (về lại Draft chờ duyệt lại), audit trail JSONL ghi actor/
timestamp cho mọi thao tác, API/UI đầy đủ. Kiểm chứng e2e 32/32 PASS gồm cả
"luật DRAFT không tham gia chấm điểm production". Chi tiết: `docs/15` §7.

### 2.4 Baseline cửa sổ (T5) và trọng số fusion (T6)

K3: cửa sổ < 7 ngày cho |Δμ| ≈ 0.62σ và tỷ lệ lật band mức 21.5% → UI cấm
hiển thị z-score cá nhân dưới 7 ngày lịch sử, coi là ổn định hơn từ 14 ngày.
K4: ba bộ trọng số hợp lý đồng thuận mức ≥ 96%, không có ca nào bị đẩy lên CAO
nguy hiểm. Cả hai đã thành quy tắc giao diện, không chỉ là báo cáo.

---

## 3. Các vấn đề chưa giải quyết — trả lời trực tiếp

### 3.1 Validation theo thời gian và lead time (T7)

*Câu hỏi gốc (13§4.4): khi nào chứng minh được "cảnh báo sớm"?*

**Cần gì:** dữ liệu dọc có biến cố ghi nhận theo thời gian — mỗi bệnh nhân nhiều
quan sát trải dài, biết thời điểm biến cố (vd nhập viện vì biến cố tim mạch,
khai triển đái tháo đường). NHANES không có và không thể có (mỗi người một dòng).

**Giao thức sẽ chạy khi có dữ liệu** (chốt ngay từ bây giờ để tránh chọn lọc
kết quả sau):

1. Split **theo mốc thời gian**: train = quan sát trước mốc T, test = sau T;
   không bao giờ để thông tin tương lai rò vào baseline/imputer.
2. Đặc trưng tại thời điểm t chỉ dùng dữ liệu ≤ t (baseline cửa sổ hướng quá khứ
   của tầng 1 đã đúng nguyên tắc này sẵn).
3. Metric chính: **lead time** = khoảng cách giữa lần đầu hệ thống cảnh báo mức
   CAO và thời điểm biến cố; phụ: AUC dự báo biến cố trong cửa sổ 30/90/180 ngày,
   sensitivity tại fixed specificity 0.9.
4. Tiêu chí thành công tối thiểu: lead time trung vị > 0 ngày với IQR dương,
   AUC cửa sổ 90 ngày > 0.70.

**Nguồn khả thi:** (a) kênh nhập liệu của chính hệ thống tích lũy dần — phù hợp
giai đoạn thử nghiệm có kiểm soát; (b) dataset công khai dọc có nhãn biến cố
(MIMIC-IV qua PhysioNet, cần chứng nhận CITI; hoặc hợp tác cơ sở y tế trong nước).
Đây là mục P2 duy nhất chặn việc dùng chữ "cảnh báo sớm" — cho đến khi có kết quả
thỏa tiêu chí trên, hệ thống chỉ gọi tên **phân tầng nguy cơ cắt ngang**.

**Tiến độ 24/08/2026 (docs/18):** đã tìm được nguồn công khai không cần credential
— NHANES Public-Use Linked Mortality File — và chạy được giao thức ở cấp cohort:
split theo thời gian 2015-16 → 2017-18, dự báo tử vong ≤12 tháng đạt AUC 0.821,
lead time trung vị **9 tháng > 0**, top-20% nguy cơ phủ 56% tử vong ≤24 tháng.
Tiêu chí "lead time trung vị > 0" đã đạt ở cấp tháng; tiêu chí AUC cửa sổ
30/90/180 ngày vẫn chờ dữ liệu dọc theo ngày (P2.4/P2.5).

### 3.2 Vòng lặp nhãn–đầu vào (T8)

Đã trả lời bằng số (§2.2): phần "học lại ngưỡng" chiếm bao nhiêu, phần tín hiệu
thêm là bao nhiêu, đo được trên đúng protocol. Khắc phục **căn bản** chỉ có hai
con đường, đều thuộc T7/T10: (a) nhãn từ biến cố tương lai độc lập với phép đo
tại thời điểm đặc trưng; (b) nhãn từ nguồn độc lập (chẩn đoán bác sĩ ghi nhận
trước/sau, không sinh từ ngưỡng đo). Trong phạm vi dữ liệu hiện có, giới hạn
này được **công bố thay vì giấu**: mọi báo cáo AUC đều kèm diễn giải "tái lập
phân tầng theo ngưỡng lâm sàng".

**Tiến độ 24/08/2026 (docs/18):** con đường (a) đã chứng minh khả thi trên dữ
liệu thật — mô hình huấn luyện từ biến cố tương lai (tử vong) đạt AUC 0.821,
trong khi nhãn cắt ngang cũ chỉ đạt 0.57–0.63 khi dự báo đúng outcome đó trên
cùng tập test. Việc chuyển hẳn huấn luyện sản xuất sang outcome biến cố thuộc
P2.7 (MIMIC-IV / EHRSHOT).

### 3.3 Mô hình chuỗi thời gian (T9)

*Câu hỏi gốc (13§4.6): khi nào thay/đổi model sản xuất?*

**Điều kiện kích hoạt** (không làm sớm vô nghĩa): ≥ 500 bệnh nhân × ≥ 60 ngày
quan sát liên tục, hoặc dataset dọc công khai tương đương. Dưới ngưỡng đó mọi
model sâu sẽ overfit và so sánh bất công với LightGBM.

**Kế hoạch khi đủ dữ liệu:** baseline đầu tiên không phải LSTM mà là LightGBM +
đặc trưng lag/rolling (slope 7/30 ngày, σ cửa sổ, EWMA) — rẻ, dễ giải thích, nếu
không thắng được thì chuỗi thời gian chưa có lý do tồn tại. Sau đó mới đến
GRU/LSTM 2 lớp và zero-shot foundation model (Chronos/TimesFM) qua điểm cắm
`src/tier1_anomaly/forecast.py` đã chừa. Đánh giá theo giao thức §3.1, so paired
cùng test window.

### 3.4 Xác nhận ngoài (T10)

*Câu hỏi gốc (13§4.7, D7): khái quát sang quần thể khác?*

Tiêu chí nghiệm thu đặt trước: trên dataset ngoài, ROC-AUC không tụt quá 0.05
so với test nội bộ và ECE sau hiệu chỉnh lại ≤ 0.05 (hiệu chỉnh lại trên val
của dataset đó). Dataset ứng viên theo thứ tự khả thi: NHANES chu kỳ chưa dùng
(2017–2018 làm hold-out thật nếu huấn luyện lại trên 2015–2016 + 2021–2023),
KNHANES (Hàn Quốc — cấu trúc khảo sát tương đồng, công khai), MIMIC-IV. Quần thể
Việt Nam chưa có bộ dữ liệu mở tương thích schema; nếu hợp tác cơ sở y tế trong
nước thì gộp chung mục tiêu với T7.

**Tiến độ 24/08/2026 (docs/18):** hold-out theo thời gian 2017-18 đã chạy với
mô hình outcome tử vong — AUC drop 0.02 (0.841 random → 0.821 temporal), đạt
tiêu chí ≤ 0.05. Xác nhận ngoài theo địa lý/quần thể vẫn mở (P2.3).

### 3.5 Khuyết dữ liệu glucose_fasting 52% (T11)

Impute median fit-train là lựa chọn mặc định an toàn nhưng chưa được đối chiếu.
Kiểm định bổ sung (không bắt buộc, chi phí thấp): chạy lại benchmark trên tập
complete-case (bỏ mẫu thiếu glucose_fasting, n giảm ~52%) rồi so AUC/ECE — nếu
kết quả ổn định (< 0.01) thì impute vô hại; nếu lệch lớn thì mọi kết quả liên
quan glucose phải báo cáo kèm phương sai do impute. Chưa chạy vì ưu tiên P1;
khi chạy sẽ lưu vào `experiments/COMPLETE-CASE-CHECK/`.

**Kết quả 24/08/2026 — đã chạy (`experiments/COMPLETE-CASE-CHECK/`):**

| Model | AUC imputed | AUC complete-case | Δ AUC |
|---|---|---|---|
| LightGBM | 0.9349±0.0030 | 0.9290±0.0057 | **−0.0059** |
| XGBoost | 0.9356±0.0027 | 0.9311±0.0061 | −0.0045 |
| LR | 0.8844±0.0078 | 0.9002±0.0074 | **+0.0158** |

- Mô hình **cây** (LightGBM/XGBoost — gồm model sản xuất): ổn định, |ΔAUC|
  trung bình < 0.01 → impute median chấp nhận được cho production.
- Mô hình **tuyến tính**: impute median làm GIẢM AUC ~0.016 một cách có hệ
  thống (glucose bị đặc thành một giá trị ở 52% mẫu làm mờ quan hệ tuyến tính);
  complete-case tốt hơn cho LR dù chỉ giữ ~45% số dòng test.
- Kết luận áp dụng: giữ nguyên impute median trong sản xuất (LightGBM); khi báo
  cáo kết quả của mô hình tuyến tính (LR dùng làm baseline giải thích), ghi chú
  phương sai do impute. Không cần đổi pipeline.

---

## 4. Lộ trình P2 với tiêu chí nghiệm thu

| Bước | Công việc | Điều kiện/tiêu chí xong |
|---|---|---|
| P2.1 | Hoàn thiện hồ sơ nghiên cứu: cập nhật docs 01–07 theo trạng thái mới (calibration, governance, đổi tên đề tài) | Mọi con số trong docs truy xuất được tới evidence package hiện tại |
| P2.2 | Complete-case check (§3.5) | Có `experiments/COMPLETE-CASE-CHECK/summary.json` + kết luận |
| P2.3 | Hold-out ngoài bằng chu kỳ NHANES chưa dùng (§3.4) | AUC drop ≤ 0.05, ECE ≤ 0.05 |
| P2.4 | Thu thập dữ liệu dọc qua kênh nhập hệ thống (§3.1) | ≥ 50 người dùng thật × ≥ 30 ngày là mốc đánh giá giữa |
| P2.5 | Temporal validation + lead time khi P2.4 đạt ngưỡng | Theo tiêu chí §3.1 |
| P2.6 | Chuỗi thời gian khi đủ dữ liệu (§3.3) | LightGBM+lag là baseline bắt buộc phải thắng |

*Cập nhật 24/08/2026:* ngoài các bước trên, đã hoàn thành **temporal validation
cấp cohort trên NHANES-LMF** (AUC tử vong 12 tháng 0.821, lead time trung vị
9 tháng — chi tiết docs/18 §4), mở thêm **P2.7: xin quyền MIMIC-IV (CITI +
DUA), đồng thời khảo sát EHRSHOT/CardioEHR** để chuyển huấn luyện sang outcome
biến cố tương lai, và hoàn thành **P2.2 complete-case check**
(`experiments/COMPLETE-CASE-CHECK/`): LightGBM/XGB ổn định (|ΔAUC| < 0.01 →
impute vô hại cho production), LR lệch +0.0158 nghiêng về complete-case — giữ
impute cho sản xuất, ghi chú phương sai khi báo cáo baseline tuyến tính.

## 5. Những gì hệ thống KHÔNG tuyên bố (cho tới khi P2 hoàn thành)

1. Không dùng thuật ngữ "cảnh báo sớm" trong sản phẩm/báo cáo — chỉ "phân tầng
   nguy cơ cắt ngang" và "hỗ trợ quyết định".
2. Không diễn giải AUC benchmark thành năng lực dự báo độc lập biến cố tương lai.
3. Không để luật chưa Active tham gia chấm điểm; không có thao tác quản trị nào
   đi ngoài audit trail.
4. Kết luận cuối luôn thuộc bác sĩ — disclaimer xuất hiện ở mọi đầu ra.

---

*Nhóm nghiên cứu biên soạn; mỗi mục truy xuất được tới evidence package hoặc mã
nguồn nêu tên. Tài liệu này thay thế danh sách "còn phải làm" rải trong docs 13–15
từ thời điểm 23/08/2026.*
