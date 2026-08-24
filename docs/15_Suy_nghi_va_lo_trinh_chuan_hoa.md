# 15. Định hướng chuẩn hóa, kiểm chứng và hoàn thiện hệ thống

> Tài liệu quyết định của nhóm nghiên cứu sau khi đọc `docs/13` (phản biện của
> giảng viên) và `docs/14` (mô tả trạng thái thật của hệ thống): **sửa gì,
> chuẩn hóa gì, chứng minh gì bằng thực nghiệm nào, và đưa những điều đó lên
> giao diện ra sao**. Đây không phải bản mô tả hệ thống thêm một lần nữa, cũng
> không phải danh sách tính năng — mỗi mục đều trả lời câu hỏi: sau phản biện,
> nhóm kiểm chứng và nâng cấp từng vấn đề bằng cách nào.
>
> Nguyên tắc xuyên suốt: **không cố sửa mọi thứ bằng code**. Mọi vấn đề thuộc
> một trong ba loại — (1) giải quyết trực tiếp bằng code/UI, (2) giải quyết bằng
> thực nghiệm/evidence, (3) không thể giải quyết nếu thiếu dữ liệu mới → giới
> hạn claim và ghi rõ lộ trình.

Chuỗi tài liệu:

```
#13 = GIẢNG VIÊN PHẢN BIỆN        (các vấn đề L1–L6, S1–S6)
#14 = NHÓM CHỨNG MINH HIỆN TRẠNG  (hệ thống thực sự đang có gì)
#15 = NHÓM QUYẾT ĐỊNH             (tài liệu này — kiểm chứng, chuẩn hóa, nâng cấp)
```

---

## 1. Mục tiêu của giai đoạn

Biến các nhận xét tại `docs/13` thành **một kế hoạch chuẩn hóa kiểm chứng được**,
đồng thời nâng giao diện từ "demo hệ thống" thành **bằng chứng trực quan cho kiến
trúc và độ tin cậy của hệ thống**.

UI hiện tại hiển thị kết quả cuối (`RỦI RO: … · Score: …`). Mục tiêu là người xem
nhìn thấy được **từng tầng đã làm gì để ra con số đó**:

```
┌───────────────────────────────────────────┐
│ RỦI RO: TRUNG BÌNH          Score: 0.53   │
├───────────────────────────────────────────┤
│ Tầng 1 — Cá nhân hóa                      │
│   7 ngày dữ liệu · baseline ổn định       │
│   SBP +2.1σ · DBP +1.8σ · xu hướng tăng   │
├───────────────────────────────────────────┤
│ Tầng 2 — Tri thức y khoa                  │
│   2 luật kích hoạt · ESH 2018 · sev 0.65  │
├───────────────────────────────────────────┤
│ Tầng 3 — ML                               │
│   LightGBM output 0.71 (chưa hiệu chỉnh)  │
├───────────────────────────────────────────┤
│ TỔNG HỢP                                  │
│   Stat .21 · Knowledge .23                │
│   ML .18 · Trend .05 → Score .67 → CAO    │
└───────────────────────────────────────────┘
```

## 2. Các vấn đề rút ra từ `docs/13`

| Mã | Vấn đề | Loại giải quyết |
|---|---|---|
| L1 | "Cảnh báo sớm" vượt bằng chứng dữ liệu cắt ngang | (3) đổi claim + roadmap dữ liệu dọc |
| L2 | Baseline cá nhân chưa ổn định khi ít quan sát | (2) thí nghiệm N quan sát + (1) UI độ đủ dữ liệu |
| L3 | Rule engine thiếu quy trình xác nhận chuyên gia | (1) versioning/workflow + audit trail |
| L4 | Trọng số/ngưỡng/hiệu chỉnh chưa được chứng minh | (2) sensitivity analysis + calibration |
| L5 | LightGBM không mô hình hóa thời gian dài | (3) interface mở rộng, production giữ nguyên |
| L6 | Rò rỉ dữ liệu/nhãn | (2) label sensitivity; phần xử lý trước đã xong |
| S1 | Thiếu dữ liệu dọc | (3) kênh chat thu thập là nguồn; roadmap |
| S2/S3/S4 | Pipeline, validation, calibration/Brier/FP/FN/lead time | (2) thực nghiệm bổ sung |
| S5 | Quản trị tri thức | (1) backend + UI |
| S6 | Giữ kiến trúc hybrid | đã đúng — củng cố, không đổi |

## 3. Đối chiếu từng vấn đề với hiện trạng (`docs/14`)

### 3.1 Early warning (L1)

- Hiện trạng: hệ thống chạy phân tầng nguy cơ trên dữ liệu cắt ngang; báo cáo
  luôn gắn disclaimer bác sĩ quyết cuối. Chưa có nơi nào phát biểu rõ phạm vi
  bằng chứng này trên UI.
- Việc cần làm: đổi wording toàn hệ thống (tài liệu + UI) từ "cảnh báo sớm" sang
  "**phát hiện bất thường / phân tầng nguy cơ**", kèm bảng phạm vi bằng chứng
  (mục 6.7). Không phải làm hệ thống yếu đi — làm nó khoa học hơn.

### 3.2 Personal baseline (L2)

- Hiện trạng: cửa sổ trượt 90 ngày hướng quá khứ, `min_periods=5`
  (`src/data/preprocess.py:61`); dưới ngưỡng → không tính z-score, đánh dấu thiếu;
  chat yêu cầu ≥ 7 ngày mới báo cáo đầy đủ. Guard đã đúng, nhưng **chưa chứng minh
  baseline ổn định thế nào theo số lượng quan sát**, và UI chưa hiển thị độ đủ dữ liệu.
- Việc cần làm: thí nghiệm N = 3/5/7/14/30/90 (mục 5.5); UI hiển thị trạng thái
  baseline theo mức quan sát (mục 6.2). Thuật ngữ: gọi là **data sufficiency /
  evidence coverage** cho đến khi có hiệu chỉnh thống kê — không gọi "xác suất đúng".

### 3.3 Knowledge provenance (L3)

- Hiện trạng: mỗi luật bắt buộc `source_url`, `severity`, `specialty`, `modes`,
  validate khi ghi (`rules.py:71-122`). Provenance cơ bản đã có; **governance chưa có**
  (phiên bản, trạng thái duyệt, lịch sử thay đổi).
- Việc cần làm: WS4 — mục 7.

### 3.4 Risk scoring (L4)

- Hiện trạng: trọng số công khai trong `src/config.py:25-31` (stat .30 /
  knowledge .35 / ml .25 / trend .10; ngưỡng 0.33/0.66; sàn an toàn 0.50 khi
  severity ≥ 0.7). Minh bạch nhưng chưa validated.
- Việc cần làm: đặt tên đúng — **"trọng số thiết kế ban đầu"** chứ không phải
  "trọng số tối ưu" (chưa có bằng chứng); gom tham số scoring thành một khối cấu
  hình có phiên bản `SCORING_VERSION`; chạy phân tích độ nhạy (mục 5.4).

### 3.5 Temporal modeling (L5)

- Hiện trạng: tầng 1 có EWMA/forecast + điểm cắm Chronos/TimesFM; production là
  LightGBM trên dữ liệu bảng.
- Việc cần làm: thiết kế interface `TemporalModel` (EWMA, ChronosAdapter,
  TimesFMAdapter, LSTMAdapter, TransformerAdapter) để mở rộng mà **không phá
  production**; benchmark temporal model chỉ diễn ra khi có dataset dọc. Không nhảy
  sang LSTM/Transformer chỉ vì được hỏi — dữ liệu bảng thì boosting vẫn là lựa chọn đúng.

### 3.6 Data/label leakage (L6)

- Hiện trạng: leakage tiền xử lý đã kiểm soát (impute fit-train, test khóa lại,
  mỗi dòng một bệnh nhân). Cấu trúc nhãn `(has_bp & has_lab) | has_meds` còn hai
  điểm nghi vấn: vòng lặp nhãn–đầu vào (D1: nhãn dm dùng chính hba1c/glucose làm
  input) và nhiễu thuốc (D2).
- Việc cần làm: **không thể fix D1/D2 bằng frontend** — giải quyết bằng thiết kế
  thực nghiệm: label sensitivity A/B (mục 5.2), và về lâu dài cần nhãn từ biến cố
  tương lai. Cho đến lúc đó, mọi kết quả AUC chỉ được phát biểu là tái lập phân
  tầng theo ngưỡng lâm sàng.

### 3.7 Dataset limitations

- Hiện trạng: cắt ngang; glucose_fasting thiếu 52%; không có kết cục tương lai;
  chưa có external/clinical validation — đã mô tả trung thực trong `docs/14`.
- Việc cần làm: expose các thông tin này lên UI (mục 6.6) thay vì chỉ nằm trong docs.

## 4. Nguyên tắc chuẩn hóa

- **Data**: mọi mức thiếu/outlier phải đo được và hiển thị được; không impute bằng
  thông tin từ test set; tách dữ liệu theo bệnh nhân mặc nhiên.
- **ML**: mọi metric xuất kèm mean ± std nhiều seed; xác suất đầu ra ghi rõ
  "trước/sau hiệu chỉnh"; fit hiệu chỉnh trên val, test chỉ đánh giá lần cuối.
- **Knowledge**: luật chưa duyệt không chạy production; mỗi thay đổi luật để lại
  dấu vết; nguồn gốc truy xuất được tới trang/chương.
- **Risk scoring**: tham số scoring gom một chỗ, có phiên bản; đổi trọng số phải
  đi kèm báo cáo so sánh; thuật ngữ "thiết kế ban đầu" đến khi có validation label.
- **UI**: phản ánh trung thực bằng chứng đã tính, không thêm tuyên bố mới; thuật
  ngữ thống nhất tiếng Việt; disclaimer nhất quán mọi trang.

## 5. Kế hoạch thực nghiệm

### 5.1 Calibration (P0)

Nâng Brier/calibration hiện có thành một **module hiệu chỉnh** trong luồng ML:

```
xác suất thô → [calibration_method: none | isotonic | platt] → xác suất hiệu chỉnh → tổng hợp rủi ro
```

Fit trên validation set, test chỉ đánh giá. Xuất bảng:

| Model | Raw Brier | Calibrated Brier | Δ |
|---|---|---|---|
| XGB | 0.0913 | … | … |
| LGBM | 0.0916 | … | … |
| RF | 0.0956 | … | … |

### 5.2 Label sensitivity (P0 — nghiên cứu quan trọng nhất)

```
Thí nghiệm A: label = htn | dm                       (nhãn gốc)
Thí nghiệm B: bỏ thành phần has_meds khỏi htn
Thí nghiệm C (nếu khả thi): nhãn độc lập với đặc trưng đo hiện tại
```

So ROC-AUC / PR-AUC / Brier / Precision / Recall giữa các thí nghiệm → Δ. Giảm
ít: nhãn robust; giảm nhiều: thành phần thuốc ảnh hưởng đáng kể, ghi rõ giới hạn.
Cả hai hướng kết luận đều là bằng chứng hợp lệ, trả lời trực tiếp L6.

### 5.3 Missingness

Bảng thiếu dữ liệu từng đặc trưng (đã đo: glucose 52%, creatinine 8%, còn lại
5%) đưa vào benchmark UI kèm chú thích phương pháp impute; chạy đối chứng bỏ hẳn
glucose_fasting để đo mức phụ thuộc kết quả vào cột nghèo nhất.

### 5.4 Weight sensitivity

Ba cấu hình trọng số trên cùng split/test:

```
v1 = .30/.35/.25/.10 (thiết kế ban đầu)
v2 = .25/.40/.25/.10
v3 = .30/.30/.30/.10
```

Đo risk-level agreement, score variance, số ca đổi mức Low/Med/High. Ổn định →
có cơ sở nói hệ thống ít nhạy với trọng số; dao động mạnh → ghi rõ giới hạn và
đặt thành câu hỏi nghiên cứu tiếp theo. Chỉ tối ưu trọng số theo dữ liệu khi có
validation label phù hợp.

### 5.5 Temporal validation & baseline stability

- Độ ổn định baseline theo N quan sát (3/5/7/14/30/90) — làm được ngay trên kênh
  dữ liệu chat đang tích lũy.
- Temporal split (train quá khứ / test tương lai) và lead time — **phải chờ bộ
  dữ liệu dọc có kết cục**; ghi là điều kiện tiên quyết, không hứa ngày.

## 5K. Kết quả thực nghiệm P0 (đợt chạy 23/08/2026)

### K1 — Calibration (chi tiết: `experiments/EXP-ML-*/calibration.json`)

Calibrator fit trên val, chọn theo val Brier; test chỉ báo cáo. Cả 6 model đều
chọn **isotonic**:

| Model | Brier raw | Brier Platt | Brier Isotonic | ECE raw | ECE isotonic |
|---|---|---|---|---|---|
| xgb | 0.0913±0.0023 | 0.0939±0.0021 | 0.0914±0.0022 | 1.49% | 1.50% |
| lgbm | 0.0916±0.0025 | 0.0940±0.0022 | 0.0920±0.0027 | 1.60% | 1.63% |
| rf | 0.0956±0.0029 | 0.0947±0.0036 | **0.0936±0.0032** | 4.46% | 1.69% |
| lr | 0.1375±0.0049 | 0.1405±0.0053 | **0.1329±0.0047** | 5.12% | **1.84%** |
| mlp | 0.1287±0.0098 | 0.1255±0.0083 | **0.1241±0.0075** | 5.37% | **2.12%** |
| fttransformer | 0.1028±0.0044 | 0.1045±0.0033 | **0.1020±0.0042** | 2.65% | **1.52%** |

Đọc kết quả: LGBM/XGB đã gần như hiệu chỉnh sẵn (ECE ~1.5%, hiệu chỉnh không
cải thiện thêm); với LR/MLP/RF/FT, isotonic giảm ECE mạnh (LR 5.1%→1.8%).
→ UI tầng ML hiển thị cặp Raw vs Calibrated theo bảng này.

### K2 — Label sensitivity (chi tiết: `experiments/LABEL-SENSITIVITY/`)

Nhãn A (có thành phần thuốc HA, positive 49.0%) vs nhãn B (bỏ thuốc, 30.8%);
2968/16314 ca (18.2%) dương **chỉ nhờ thuốc**:

| Model | ROC-AUC A | ROC-AUC B | Δ AUC |
|---|---|---|---|
| lgbm | 0.935±0.003 | **1.000±0.000** | +0.065 |
| xgb | 0.937±0.003 | **1.000±0.000** | +0.063 |
| lr | 0.884±0.008 | 0.970±0.002 | +0.086 |

Ba phát hiện định lượng:

1. Nhãn B → AUC = 1.000: bỏ nhiễu thuốc, bài toán **suy biến thành tái lập
   ngưỡng lâm sàng** (sbp≥140 ∨ dbp≥90 ∨ hba1c≥6.5 ∨ fpg≥7.0 nằm ngay trong
   feature set). D1 được chứng minh thực nghiệm, không còn là giả định.
2. Oracle dùng chính nhãn B làm score chỉ đạt AUC 0.827 trên nhãn A — model
   thật đạt 0.936, tức vượt oracle: một phần hiệu năng đến từ việc **suy ra
   trạng thái đang điều trị** từ pattern đo được (người uống thuốc HA vẫn để
   lại dấu vết trong số đo).
3. Hệ quả diễn giải bắt buộc: AUC 0.9356 của benchmark = khả năng tái lập phân
   tầng lâm sàng + suy đoán tình trạng điều trị; **không phải dự báo biến cố
   tương lai**. Ghi vào mọi báo cáo.

### K3 — Baseline stability (chi tiết: `experiments/BASELINE-STABILITY/`)

150 bệnh nhân mô phỏng × 150 ngày × 4 chỉ số, tham chiếu N=90, seed=42:

| N (ngày) | \|Δμ\| (σ₉₀) | \|Δσ\| tương đối | Đổi vùng z | Đổi cờ z≥2σ |
|---|---|---|---|---|
| 3 | 0.615 | 0.460 | 21.5% | 2.7% |
| 7 | 0.463 | 0.309 | 16.5% | 2.6% |
| 14 | 0.331 | 0.215 | 11.9% | 2.3% |
| 30 | 0.206 | 0.130 | 7.4% | 1.6% |

→ Cơ sở định lượng cho trạng thái baseline trong UI (U2): <7 ngày = CHƯA ỔN
ĐỊNH; 7–13 = ĐỦ ĐIỀU KIỆN (band flip ~17%); ≥14 = ỔN ĐỊNH HƠN; khuyến nghị
hiển thị cảnh báo khi N<14 vì tỉ lệ đảo kết luận z-band còn cao.

### K4 — Weight sensitivity (chi tiết: `experiments/WEIGHT-SENSITIVITY/`)

155 bệnh nhân (5 mẫu thật + 150 giả từ NHANES), pipeline 3 tầng đầy đủ:

| Cặp | Đồng thuận mức | Chuyển mức | std(Δscore) |
|---|---|---|---|
| v1 vs v2 | 96.8% | 5 ca TB→THẠP | 0.031 |
| v1 vs v3 | 99.4% | 1 ca THẠP→TB | 0.029 |
| v2 vs v3 | 96.1% | 6 ca THẠP→TB | 0.043 |

Không có chuyển đổi đi lên nguy hiểm (Low→High) và không mất case CAO nào.
→ Risk level khá bền vững với thiết kế trọng số; vẫn ghi rõ trọng số hiện tại
là thiết kế ban đầu, chưa tối ưu theo outcome (không tune "đẹp số").

## 6. Kế hoạch chuẩn hóa giao diện

UI trở thành **evidence surface** của nghiên cứu. Các hạng mục (U):

| Mã | Hạng mục | Nội dung |
|---|---|---|
| U1 | Patient timeline | `/chat`: dòng thời gian các lần đo của bệnh nhân (ngày → giá trị), cạnh đó baseline μ/σ, độ lệch mmHg và σ, xu hướng — đập thẳng vào nhận xét "so người này với lịch sử chính họ" |
| U2 | Tầng 1 evidence | Số ngày dữ liệu, trạng thái baseline (CHƯA ỔN ĐỊNH / ĐỦ ĐIỀU KIỆN / ỔN ĐỊNH HƠN theo N), z-score từng chỉ số, xu hướng |
| U3 | Tầng 2 evidence | Luật kích hoạt + nguồn (ESH 2018…), severity, chuyên khoa; chuẩn hóa layout provenance trên `/rules` |
| U4 | Tầng 3 evidence | Ghi "LightGBM output 0.71 — xác suất model (chưa hiệu chỉnh)", tách bạch tuyệt đối khỏi "risk score" tổng hợp; **không** dùng chữ "Confidence 71%" |
| U5 | Score breakdown | Khối tổng hợp stat/knowledge/ml/trend × trọng số = điểm cuối, ấn-mở dạng "WHY?" cho từng kết luận — explainability cấp hệ thống |
| U6 | Dataset/benchmark panel | `/benchmark` chia tab: Performance (AUC/F1…) · Calibration (Brier trước/sau + curve) · Robustness (5 seed, label sensitivity) · Data provenance (dataset, N, positive rate, missing %, split, preprocessing, định nghĩa nhãn); riêng panel dataset: loại cắt ngang, không có kết cục tương lai, external/clinical validation ✗ |
| U7 | Wording early warning | Toàn UI đổi "cảnh báo sớm" → "PHÁT HIỆN BẤT THƯỜNG / PHÂN TẦNG NGUY CƠ" + khối "Phạm vi bằng chứng hiện tại": ✓ phân tầng trên dữ liệu hiện có, ✓ phát hiện thay đổi so baseline cá nhân, ⚠ chưa chứng minh dự báo biến cố, ⚠ chưa có lead time |
| U8 | **Bỏ chatbot — chuyển UI sang dạng biểu mẫu** | Loại bỏ giao diện hội thoại; thay bằng form input/output chuẩn: ô điền tham số (patient_id, ngày đo…), chọn ngày, upload file (PDF/DOCX/TXT), nút chức năng tách bạch ("Đánh giá bệnh nhân", "Nhập file kết quả", "Xem timeline"…). Mục tiêu: UX đơn giản — dễ dùng — dễ hiểu — quy tắc rõ ràng, phù hợp bản chất hệ thống đánh giá đầu vào/đầu ra cố định thay vì hội thoại tự do |
| U9 | Render markdown chuẩn hóa | Toàn bộ nội dung giải thích/báo cáo động trong UI được render từ Markdown bằng thư viện Python (`python-markdown`, extension `tables` + `fenced_code`) qua endpoint `/api/render_markdown` — không nhồi text thô lộn xộn vào HTML; đảm bảo định dạng bảng/danh sách thống nhất, dễ đọc, đúng và đủ |
| U10 | **Quản lý bản ghi cá nhân theo ngày** | Vì hệ thống cá nhân hóa, cần trang quản trị dữ liệu cá nhân dạng bảng: mỗi hàng = một ngày đo (sắp xếp theo ngày), cho phép **edit trực tiếp các giá trị số** trên từng hàng (inline edit), **thêm/bớt bản ghi**; ô nào không có dữ liệu thì để trống hoàn toàn bình thường — hệ thống phải chấp nhận missing từng trường và hiển thị rõ trạng thái thiếu. Đây là nền để timeline/baseline (U1/U2) có dữ liệu thật để xem |

### 6.1 Bố cục evidence surface (mục tiêu P1)

Panel kết quả đánh giá một patient hiển thị theo khối, người chấm không cần
đọc source code vẫn hiểu hệ thống làm gì:

```text
PHÂN TẦNG NGUY CƠ        TRUNG BÌNH · 0.53
Phạm vi: dữ liệu hiện có · Chưa chứng minh dự báo biến cố tương lai
─────────────────────────────────────────────
TẦNG 1 · CÁ NHÂN      N ngày · baseline μ/σ · z-score từng chỉ số + xu hướng
TẦNG 2 · TRI THỨC     luật kích hoạt + id/version/source + severity
TẦNG 3 · ML           model · raw output · calibrated output · phương pháp hiệu chỉnh
TỔNG HỢP              stat/knowledge/ml/trend × trọng số = điểm cuối → mức nguy cơ
```

`/benchmark` trở thành trang nghiên cứu: Performance · Calibration (raw vs
calibrated Brier/ECE + curve) · Robustness (5 seed, label sensitivity,
missingness, weight sensitivity) · Data provenance · Evidence status
(✓/◐/○ theo TRIPOD+AI & DECIDE-AI — xem mục 13 R1/R2).

## 7. Quản trị tri thức (WS4)

Mở rộng schema luật và `/api/kb/*`:

```
rule_version, status (draft/review/approved/rejected),
created_by, approved_by, approved_at, updated_at, previous_version,
audit trail (ai sửa, khi nào, nội dung cũ/mới)
```

Workflow: `Draft → Review → Approved → Active`. Luật chưa Approved chỉ preview,
không tham gia chấm điểm production. Trang `/rules` hiển thị đầy đủ các trường
này ngay trên card luật — biến câu hỏi "nguồn tri thức lấy ở đâu?" thành tính
năng nhìn thấy được.

### 7.1 Trạng thái triển khai P1 (23/08/2026)

Đã hoàn tất và kiểm chứng qua API:

| Hạng mục | Triển khai | Kiểm chứng |
|---|---|---|
| Governance module | `src/tier2_knowledge/governance.py`: STATUSES, TRANSITIONS, `apply_transition`, audit JSONL (`data/kb/audit_log.jsonl`) | ✓ |
| KB tích hợp | Luật cũ tự migrate → active v1.0; luật mới luôn **draft v1.0**; sửa nội dung → bump version +0.1, reset draft, lưu `previous_version`; `evaluate()` mặc định chỉ chạy luật **active** | ✓ |
| Workflow | draft→review→approved→active (+rejected); chặn nhảy cóc (draft→active bị từ chối); sửa luật đang review → v1.1/draft | ✓ |
| Audit trail | create/edit/delete/transition đều ghi actor + timestamp + chi tiết; `GET /api/kb/audit` | ✓ |
| `/rules` UI | Card hiển thị version, badge trạng thái màu, nút chuyển trạng thái theo luồng, dòng "✓ production / ◐ preview" | ✓ |
| U10 Bản ghi | `ChatStore.upsert/delete_value/table_by_date` + `GET/PUT/DELETE /api/records/{pid}`: bảng theo ngày, sửa từng ô, ô trống hợp lệ | ✓ |
| U9 Markdown | `POST /api/render_markdown` (python-markdown, extensions tables+fenced_code); tầng ML và khuyến nghị render qua endpoint này | ✓ |
| U8 Bỏ chatbot | `/` giờ là SPA biểu mẫu `app.html`: Đánh giá · Bản ghi · Luật · Benchmark; không còn giao diện hội thoại | ✓ |
| Evidence surface | Panel phân tầng: badge mức + score + phạm vi dữ liệu; Tầng 1 (z-score/baseline), Tầng 2 (luật+nguồn), Tầng 3 (`/api/evidence/ml` — raw vs calibrated isotonic, cảnh báo không diễn giải), Tổng hợp components × trọng số | ✓ |

### 7.3 Hoàn tất P1 (23/08/2026)

- **Benchmark → trang nghiên cứu**: `/benchmark` có 6 tab — Kết quả tổng hợp,
  Hiệu chỉnh xác suất (Brier/ECE raw vs Platt vs isotonic), Độ bền vững K2–K4
  (đọc trực tiếp từ `/api/benchmark/research`), Dữ liệu & Evidence status
  (checklist ✓ done / ◐ partial / ○ todo), So sánh luận giải, Chi tiết thí nghiệm.
- **Calibration vào production**: `_ml_score_for` áp calibrator isotonic lên điểm
  ML trước khi fusion — đối sách S4 hoàn trọn, không chỉ dừng ở hiển thị.
- **Dữ liệu demo**: `scripts/seed_demo_data.py` seed P001–P005 (từ sample_long)
  và hai ca tổng hợp DEMO_HYPERTENSIVE / DEMO_DIABETIC (45 ngày, có cú sốc 7
  ngày cuối để thử Tầng 1).
- **Kiểm thử e2e**: 32/32 PASS (trang tĩnh, đánh giá 5 ca seeded, CRUD bản ghi,
  luồng governance đầy đủ, nl2br, evidence/ml, research endpoint). Lỗi phát hiện
  khi test: `add_rule` KeyError khi thiếu source_url — đã sửa (`rules.py`).
- Sửa lỗi render bảng bản ghi (placeholder chứa HTML làm vỡ ô nhập).

→ Toàn bộ mục "Còn lại cho P1" đã xong. Các vấn đề chưa giải quyết (temporal
validation, external validation, chuỗi thời gian…) được chốt sổ và trả lời trong
**docs/16**, kèm tiêu chí nghiệm thu cho từng mục P2.

### 7.2 Cải tiến giao diện theo phản hồi sử dụng (23/08/2026)

Sau khi dùng thử, trang chính được chỉnh lại bốn điểm:

1. **Quản trị luật đưa thẳng vào trang chính** (bỏ iframe): tab "Luật &
   Quản trị" giờ là giao diện native toàn màn hình — danh sách luật với badge
   trạng thái + version, nút chuyển trạng thái đúng luồng, form thêm/sửa luật
   (hệ và chỉ số lấy động từ `/api/kb`), hỗ trợ điều kiện `between` và JSON
   cho logic AND/OR lồng; audit trail hiển thị ngay dưới danh sách.
2. **Tầng 2 hiện tên tiếng Việt**: mỗi luật kích hoạt hiển thị
   `<mã> — <tên luật>` kèm hệ, độ nặng và link nguồn (trước đây chỉ có mã).
3. **Xuống dòng trong markdown**: thêm extension `nl2br` vào
   `/api/render_markdown` — mọi dòng đơn đều xuống dòng đúng khi render
   (trước đây các ý bị dính liền một khối).
4. **Bản ghi cá nhân đầy đủ hơn**: bộ chọn cột (thêm/bớt chỉ số trong 10 chỉ
   số hệ thống: HA tâm thu/trương, nhịp tim, glucose ngẫu nhiên/lúc đói,
   HbA1c, creatinine, eGFR, SpO₂, BMI) lưu vào localStorage; "Thêm ngày" tạo
   dòng trống ngay không cần ghi dữ liệu ảo; nút xóa cả ngày
   (`DELETE /api/records/{pid}?timestamp=...` không cần metric); ô nhập hiển
   thị đơn vị đo; bảng Tầng 1 bổ sung cột xu hướng và đơn vị.

### 7.4 Khởi động P2 — temporal validation trên NHANES-LMF (24/08/2026)

Giai đoạn P2 bắt đầu bằng việc giải quyết gốc chung của các vấn đề còn lại
(T7/T8/T10): thiếu dữ liệu có biến cố tương lai. Kết quả đầu tiên:

- **Tích hợp NHANES Public-Use Linked Mortality File 2019** (công khai, không
  cần credential): `scripts/fetch_nhanes_mortality.py` parse file fixed-width
  10 chu kỳ 1999–2018 và ghép với dataset hiện có qua SEQN — 10 065 người lớn
  có tình trạng sống/chết + số tháng follow-up (chi tiết docs/18 §3).
- **Thí nghiệm `EXP-TEMPORAL-LMF`** (`scripts/run_temporal_validation.py`):
  split theo thời gian 2015-16 → 2017-18, dự báo tử vong ≤12 tháng từ một kỳ
  khám. Kết quả: LR AUC **0.821** temporal / 0.841 random (gap chỉ 0.02),
  C-index 0.822, isotonic ECE 0.40%; lead time trung vị **9 tháng** với top-20%
  nguy cơ phủ 56% tử vong ≤24 tháng; đối chứng nhãn cắt ngang cũ chỉ đạt
  0.57–0.63 trên cùng outcome.
- Ý nghĩa: lần đầu có bằng chứng prospective không vòng lặp nhãn; tiêu chí
  "lead time trung vị > 0" của §3.1 đạt ở cấp cohort/horizon tháng; random
  split được chứng minh chỉ lạc quan hơn ~0.01–0.02 AUC. Chữ "cảnh báo sớm"
  vẫn chưa dùng trong sản phẩm (chờ dữ liệu dọc theo ngày — P2.4/P2.5).
- Bổ sung mục tiêu **P2.7**: xin quyền MIMIC-IV (CITI + DUA), khảo sát
  EHRSHOT/CardioEHR để chuyển huấn luyện sang outcome biến cố.

**Bổ sung 24/08/2026 — P2.2 complete-case check** (`experiments/COMPLETE-CASE-CHECK/`,
cùng split + seed với arm impute): LightGBM/XGB ổn định khi bỏ mẫu thiếu glucose
(|ΔAUC| trung bình −0.0059/−0.0045 < 0.01) → impute median vô hại cho model sản
xuất; LR lệch có hệ thống +0.0158 nghiêng về complete-case → khi báo cáo baseline
tuyến tính phải ghi chú phương sai do impute. Pipeline không đổi. Chi tiết:
docs/16 §3.5.

Chi tiết đầy đủ: **docs/18**.

**Bổ sung 24/08/2026 — UI v2 + bộ e2e chính thức:**

- Bộ kiểm thử end-to-end chuyển vào repo tại `scripts/e2e_test.py`
  (trước nằm tạm ở /tmp): 35 kiểm tra — trang tĩnh, đánh giá 5 bệnh nhân
  seeded, CRUD bản ghi, luồng governance đầy đủ, evidence/research payload.
  Kết quả 24/08/2026: **35 PASS / 0 FAIL**.
- Nâng cấp giao diện: dark mode (CSS variables + nút chuyển ở header, lưu
  localStorage), biểu đồ xu hướng SVG trong tab Bản ghi (mỗi chỉ số một đường,
  chuẩn hóa min–max, legend giá trị cuối), xuất CSV cho bảng bản ghi và audit
  trail (khắc phục hạn chế mục 3, docs/17 §6), chọn chế độ chuyên khoa đầy đủ
  6 hệ (htn/dm/cv/ckd/met/resp).
- `/api/benchmark/research` bổ sung trường `complete_case`; trang Benchmark
  có panel riêng bên cạnh "Validation theo thời gian".
- Tổng kết đã giải quyết / giới hạn còn lại của cả dự án: **docs/19**.

## 8. Quản lý trạng thái bằng chứng

Thêm panel **SYSTEM EVIDENCE STATUS** (trên `/benchmark` hoặc trang tổng quan):

```
✓ Internal validation (CV 5 seed, test khóa)
✓ Kiểm soát leakage tiền xử lý
✓ Personal baseline + guard min_periods
✓ Rule provenance (source_url, severity)
✓ Multi-model benchmark + Brier
◐ Probability calibration      (đang làm)
◐ Label sensitivity            (đang làm)
○ Temporal validation          (cần dữ liệu dọc có kết cục)
○ External validation          (cần cohort ngoài)
○ Clinical validation          (cần đối tác y tế)
```

Người xem biết ngay hệ thống đang chứng minh được gì và chưa chứng minh được gì —
đúng tinh thần không thổi phồng của `docs/13`.

## 9. Phân loại công việc theo khả năng giải quyết

| Loại | Công việc |
|---|---|
| (1) Code/UI giải quyết trực tiếp | U1–U7; SCORING_VERSION; wording; panel trạng thái bằng chứng; rule versioning + audit trail |
| (2) Thực nghiệm/evidence | Calibration 5.1; label sensitivity 5.2; missingness 5.3; weight sensitivity 5.4; baseline stability 5.5a |
| (3) Cần dữ liệu mới — giới hạn claim + roadmap | Lead time; temporal split validation; LSTM/foundation model; external validation; clinical validation |

## 10. Thứ tự triển khai

**P0 — phải làm:** label sensitivity · calibration module · data provenance/missingness UI · personal baseline UI (U1/U2) · score breakdown (U5) · wording early warning (U7).

**P1 — nên làm:** rule versioning + approval workflow + audit trail · benchmark robustness panel · baseline sufficiency status · evidence panel hoàn chỉnh (U3/U4/U6).

**P2 — khi có dữ liệu:** temporal split · lead-time evaluation · longitudinal outcome · external validation.

**P3 — research extension:** LSTM/GRU · Temporal Transformer · EHR foundation models (qua interface `TemporalModel`, không phá production).

Không lao vào 20 việc cùng lúc: chốt P0 thành bằng chứng rồi mới chuyển P1.

## 11. Tiêu chí hoàn thành

- P0 xong khi: bảng Δ-AUC nhãn gốc vs bỏ thuốc xuất hiện trong evidence package;
  bảng Brier trước/sau hiệu chỉnh có số thật; UI hiển thị timeline + trạng thái
  baseline + breakdown điểm; không còn cụm "cảnh báo sớm" sai phạm vi nào trong
  tài liệu/UI.
- P1 xong khi: một luật sửa qua `/rules` để lại vết version + audit trail; luật
  draft không chạy production.
- P2/P3 xong khi có điều kiện dữ liệu — tiêu chí chi tiết sẽ viết khi dataset sẵn sàng.

## 12. Các giới hạn vẫn phải giữ nguyên (kể cả khi đã chuẩn hóa)

1. Dữ liệu cắt ngang: kết quả là **phân tầng nguy cơ**, không phải dự báo biến cố.
2. Vòng lặp nhãn–đầu vào (D1): AUC đọc là tái lập phân tầng theo ngưỡng lâm sàng
   cho tới khi có nhãn độc lập/từ tương lai.
3. Positive rate ~49% do cách định nghĩa nhãn rộng — không diễn giải thành "dễ".
4. glucose_fasting thiếu 52% — mọi kết quả liên quan phải kèm cảnh báo này.
5. ML output chưa hiệu chỉnh không được gọi là confidence/xác suất bệnh.
6. Kết luận cuối thuộc về bác sĩ — disclaimer mọi đầu ra.

## 13. Tài liệu tham khảo đã xác thực

Toàn bộ mục dưới đây đã được đối chiếu trực tiếp qua PubMed/EuropePMC (PMCID,
DOI, volume, trang đều khớp) ngày 23/08/2026. Các bài đều là truy cập mở (OA);
PDF không tải tự động được do nhà xuất bản chặn bot — truy cập đầy đủ qua link
DOI/PMCID kèm theo.

### 13.1 Chuẩn báo cáo và đánh giá lâm sàng giai đoạn đầu

**[R1] TRIPOD+AI** — Collins GS, Moons KGM, Dhiman P, Riley RD, Beam AL,
Van Calster B, et al. *TRIPOD+AI statement: updated guidance for reporting
clinical prediction models that use regression or machine learning methods.*
BMJ 2024;385:e078378.
DOI: [10.1136/bmj-2023-078378](https://doi.org/10.1136/bmj-2023-078378) ·
PMCID: PMC11019967.
→ Khung chuẩn hóa báo cáo cho toàn bộ benchmark trong `docs/11` và các thí
nghiệm P0; checklist reporting gắn với mục 4 (nguyên tắc chuẩn hóa).

**[R2] DECIDE-AI** — Vasey B, Nagendran M, Campbell B, Clifton DA, Collins GS,
Denaxas S, et al. (DECIDE-AI Steering Group). *Reporting guideline for the
early-stage clinical evaluation of decision support systems driven by
artificial intelligence: DECIDE-AI.* Nat Med 2022;28:924–933.
DOI: [10.1038/s41591-022-01772-9](https://doi.org/10.1038/s41591-022-01772-9).
Bản nghiên cứu đi kèm (open access): BMJ 2022;377:e070904,
DOI: [10.1136/bmj-2022-070904](https://doi.org/10.1136/bmj-2022-070904),
PMCID: PMC9116198.
→ Thiết kế đánh giá giai đoạn đầu của hệ hỗ trợ quyết định; đối chiếu với
kế hoạch giao diện U1–U7 ở mục 6.

### 13.2 Đánh giá, hiệu chỉnh và cập nhật mô hình

**[R3] Hướng dẫn đánh giá/cập nhật (systematic review)** — Binuya MAE,
Engelhardt EG, Schats W, Schmidt MK, Steyerberg EW. *Methodological guidance
for the evaluation and updating of clinical prediction models: a systematic
review.* BMC Med Res Methodol 2022;22:316.
DOI: [10.1186/s12874-022-01801-8](https://doi.org/10.1186/s12874-022-01801-8) ·
PMCID: PMC9742671.
→ Cơ sở cho lộ trình recalibration → revision → extension; bám vào P0
(calibration) trước khi nghĩ đến retrain (mục 5.1).

**[R4] Phát hiện calibration drift** — Davis SE, Greevy RA Jr, Lasko TA,
Walsh CG, Matheny ME. *Detection of calibration drift in clinical prediction
models to inform model updating.* J Biomed Inform 2020;112:103611.
DOI: [10.1016/j.jbi.2020.103611](https://doi.org/10.1016/j.jbi.2020.103611) ·
PMCID: PMC8627243.
→ Tham chiếu kỹ thuật cho monitoring hiệu chỉnh theo thời gian; nối với giới
hạn L4/L5 và tiêu chí hoàn thành P2.

**[R5] Dynamic prediction systems** — Jenkins DA, Martin GP, Sperrin M,
Riley RD, Debray TPA, Collins GS, et al. *Continual updating and monitoring of
clinical prediction models: time for dynamic prediction systems?* Diagn Progn
Res 2021;5:1.
DOI: [10.1186/s41512-020-00090-3](https://doi.org/10.1186/s41512-020-00090-3) ·
PMCID: PMC7797885.
→ Tầm nhìn dài hạn (P3): hệ "sống", giám sát liên tục; hiện chưa khả thi do
thiếu luồng dữ liệu dọc — giữ làm hướng phát triển, không cam kết sớm.

### 13.3 Dataset shift và tính bền vững của ML y tế

**[R6] Tổng quan giảm suy giảm hiệu năng do dataset shift** — Guo LL,
Pfohl SR, Fries J, Posada J, Fleming SL, Aftandilian C, Shah N. *Systematic
Review of Approaches to Preserve Machine Learning Performance in the Presence
of Temporal Dataset Shift in Clinical Medicine.* Appl Clin Inform
2021;12:808–815.
DOI: [10.1055/s-0041-1735184](https://doi.org/10.1055/s-0041-1735184) ·
PMCID: PMC8410238.
→ Kết luận chính trùng với định vị của nhóm nghiên cứu: mọi chiến lược cứu được
calibration nhưng không chắc cứu được discrimination — lý do AUC không được
thổi phồng khi nhãn vòng tròn (D1, mục 3.6).

**[R7] Validation dưới data shift trên EHR dọc** — Li Y, Salimi-Khorshidi G,
Rao S, Canoy D, Hassaine A, Lukasiewicz T, Rahimi K. *Validation of risk
prediction models applied to longitudinal electronic health record data for
the prediction of major cardiovascular events in the presence of data shifts.*
Eur Heart J Digit Health 2022;3(4):535–547.
DOI: [10.1093/ehjdh/ztac061](https://doi.org/10.1093/ehjdh/ztac061) ·
PMID: 36710898 · PMCID: PMC9779795.
→ Ví dụ thực tế về validation mô hình khi phân bố dữ liệu trôi; tham vọng
longitudinal outcome ở P2 cần đọc theo chuẩn này.

### 13.4 Ghi chú sử dụng

- Trích dẫn trong báo cáo/bài viết luôn dùng DOI chính thức; PMCID chỉ để tra
  cứu nhanh bản full text.
- Khi bổ sung tài liệu mới: xác minh qua PubMed/EuropePMC trước khi ghi vào đây;
  không chép tay số trang từ nguồn thứ cấp.
- PDF đầy đủ đã lưu cục bộ tại `docs/references/` (7 bài, đặt tên theo mẫu đã
  quy ước); danh sách xem `docs/references/README.md`.

## 14. Quyết định đổi tên đề tài (23/08/2026)

### 14.1 Lý do

Tên cũ — *"Hệ thống cảnh báo sớm nguy cơ sức khỏe cá nhân hóa dựa trên khung
hỗ trợ quyết định lâm sàng tích hợp học máy"* — có hai nhược điểm:

1. **Overclaim học thuật:** "cảnh báo sớm" hàm ý early prediction / dự báo biến
   cố tương lai, trong khi bằng chứng hiện có chỉ đủ cho **đánh giá nguy cơ tại
   thời điểm hiện tại** (cross-sectional ML + baseline cá nhân + luật lâm
   sàng) — đúng như phân tích ở mục 12 và giới hạn D1.
2. **Dài, khó đọc** và dễ mời câu hỏi phản biện kiểu *"outcome tương lai là gì?
   Lead time bao nhiêu ngày?"* mà hệ thống chưa trả lời được.

### 14.2 Tên chính thức mới

> **Hệ thống đánh giá nguy cơ sức khỏe cá nhân hóa tích hợp học máy và hỗ trợ
> quyết định lâm sàng**

Tên tiếng Anh:

> *A Personalized Health Risk Assessment System Integrating Machine Learning
> and Clinical Decision Support*

Ánh xạ tên ↔ kiến trúc (`docs/14`): "đánh giá nguy cơ" = năng lực thực tế;
"cá nhân hóa" = Tier 1 (baseline, z-score, timeline); "tích hợp học máy" = ML là
một thành phần của tầng 3; "hỗ trợ quyết định lâm sàng" = Tier 2 + Tier 3 +
evidence package.

### 14.3 Điều kiện để dùng lại "cảnh báo sớm"

Chỉ xem xét bổ sung "cảnh báo sớm" vào tên/mô tả sau khi đạt đồng thời:

```text
Dữ liệu T0  →  outcome tại T+Δ (từ tương lai, độc lập đầu vào)
           →  mô hình temporal được validate (temporal split)
           →  đánh giá lead-time (mục 5.5, P2)
```

Trước đó, "cảnh báo sớm" chỉ được xuất hiện ở vị trí **mục tiêu phát triển**
(P2/P3), không nằm trong tên đề tài hay cam kết tính năng.

### 14.4 Phạm vi đã cập nhật

| Vị trí | Xử lý |
|---|---|
| `README.md` (H1 + tên tiếng Anh) | Đổi sang tên mới |
| `docs/01`, `docs/02` (dòng "Đề tài") | Đổi sang tên mới, kèm chú dẫn về mục này |
| `src/api.py` (FastAPI title, trang demo), `src/main.py`, `src/__init__.py` | Đổi mô tả tương ứng |
| `docs/AI y tế.md` | **Giữ nguyên** — bản gốc đề bài của giảng viên, mang tính lịch sử |
| UI chat (`index.html`) | Không đổi ("HealthRisk · Trợ lý sức khỏe", không chứa cụm cần sửa) |

---

*Tài liệu định hướng nội bộ của nhóm nghiên cứu; truy xuất mã nguồn và số liệu
theo `docs/14`, phản biện gốc theo `docs/13`. Không thay thế chẩn đoán của bác sĩ.*
