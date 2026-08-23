# 14. Mô tả chi tiết kiến trúc hệ thống — đầu vào/đầu ra, dữ liệu huấn luyện, nhãn, hạn chế, kết quả

> Tài liệu mô tả **kiến trúc thực sự đang chạy** của toàn hệ thống (không phải thiết
> kế trên giấy): các cổng vào/ra, luồng xử lý qua ba tầng, cách dựng tập dữ liệu
> huấn luyện, cách tạo nhãn kèm các điểm lỗi của dữ liệu, ưu/nhược điểm, và kết quả
> đo được. Mọi khẳng định đều truy xuất được tới file mã nguồn cụ thể.

---

## 1. Kiến trúc tổng thể

```
                        ┌─────────────────────────────────────────────┐
  CỔNG VÀO              │           LÕI XỬ LÝ (src/core/pipeline.py)   │        CỔNG RA
                        │                                             │
CSV chuẩn schema ──────►│ resample_to_daily → impute_missing          │──► Báo cáo Markdown
(patient_id,timestamp,  │ snapshot giá trị hiện tại                   │    (report/<pid>.md)
 metric,value)          │                                             │
                        │ TẦNG 1  zscore + IsolationForest + EWMA     │      JSON đầy đủ
PDF/DOCX/TXT báo cáo ──►│         + sai số dự báo  (≥7 điểm mới chạy) │      (--json-out)
(src/ingest/pipeline.py)│                    ↓ AnomalyRecord[]        │
                        │ TẦNG 2  rule engine trên JSON               │──► REST API trả JSON
Câu chữ tự nhiên ──────►│         9 luật / 10 chỉ số                  │      cho UI chat
(ChatParser regex)      │                    ↓ Hit[] severity         │
                        │ TẦNG 3  total = stat×.30 + knowledge×.35    │──► Trang web:
File upload qua chat ──►│         + ml×.25 + trend×.10                │      /chat  /rules
(kéo-thả/dán/📎)        │         ngưỡng .33/.66, sàn an toàn .50     │      /benchmark
                        └─────────────────────────────────────────────┘
```

### 1.1 Các cổng vào

| Cổng | Điểm vào | Định dạng đầu vào |
|---|---|---|
| CLI | `python -m src.main --input <csv>` (`src/main.py`) | CSV schema chuẩn `patient_id,timestamp,metric,value` |
| REST API | `src/api.py` — 21 endpoint | JSON / form / multipart |
| Chat UI | `/chat` (`src/chat/static/index.html`) | Câu tiếng Việt có chỉ số, file PDF/DOCX/TXT (📎, kéo-thả, Ctrl+V), chọn ngày đo |
| Quản trị tri thức | `/rules` | Form thêm/sửa/xóa luật + hệ cơ quan + chỉ số (validate trước khi ghi) |
| Benchmark | `/benchmark`, `scripts/run_benchmark.py` | Form nhập ca mẫu; dataset CSV |

Endpoint chính (`src/api.py`): `/api/chat`, `/api/chat_file` (kèm `date` tùy chọn),
`/api/chat/patients`, `/api/chat/status`, `/api/chat/reset`, `/api/assess`,
`/api/assess_docs`, `/api/kb/*` (CRUD luật/hệ cơ quan/chỉ số), `/api/benchmark`,
`/api/benchmark/explain`.

### 1.2 Các cổng ra

- **Báo cáo Markdown theo bệnh nhân**: mức rủi ro tiếng Việt (THẤP/TRUNG BÌNH/CAO),
  điểm số 0–1, hệ cơ quan ảnh hưởng, cảnh báo theo luật kèm link nguồn, bảng chỉ
  số (giá trị, đường cơ sở cá nhân, xu hướng, z-score), mục "Cách tính điểm",
  điểm tổng hợp theo từng mô hình ML (trong `<details>`).
- **JSON cấu trúc**: `risk_level`, `risk_score`, `affected_systems`, `evidence`,
  `recommendations`, `components`, `metrics_detail`.
- **Evidence package benchmark**: `experiments/EXP-ML-<MODEL>-<SEED>/` gồm
  metrics.json, predictions.csv, curves.png, feature_importance.csv, model.joblib;
  tổng hợp `summary.json/md/csv`.

### 1.3 Lõi xử lý — `assess_patient` (`src/core/pipeline.py:61`)

1. **Làm sạch**: nội suy về chuỗi ngày (`resample_to_daily`), impute theo giới
   hạn thiếu (`impute_missing`: thiếu > 30% → không nội suy, chỉ đánh dấu).
2. **Snapshot** giá trị hiện tại = dòng cuối cùng — tầng 2 luôn kích hoạt luật
   theo giá trị hiện tại, kể cả mới chỉ 1 lần đo.
3. **Tầng 1** chỉ chạy khi ≥ `min_points` (7) điểm: nếu thiếu, ghi chú rõ "chưa đủ
   lịch sử" và vẫn đánh giá được bằng tri thức y khoa — không bao giờ bịt đầu ra.
4. **Tầng 2**: rule engine quét snapshot qua điều kiện AND/OR trên 10 chỉ số,
   trả danh sách hit kèm severity 0–1 và chuyên khoa.
5. **Tầng 3**: tổng hợp điểm; ML nạp lazy singleton (`risk_lgbm_real.joblib`),
   không có model thì ml_score = 0 và hệ thống vẫn chạy.
6. **Tầng 4 (tùy chọn)**: interface giải thích (`src/tier4_explain/base.py`) —
   hiện triển khai bằng perturbation/SHAP trong `src/experiments/view.py`.

---

## 2. Ba tầng chi tiết

### 2.1 Tầng 1 — bất thường cá nhân hóa (`src/tier1_anomaly/detector.py`)

| Thành phần | File | Ý nghĩa |
|---|---|---|
| Z-Score cá nhân | `zscore.py` | So giá trị hiện tại với μ/σ **cá nhân** (cửa sổ trượt 90 ngày hướng quá khứ, `min_periods=5`, `preprocess.py:61`); đầu ra: z, độ lệch tuyệt đối, so nửa đầu/nửa sau chuỗi |
| Isolation Forest | `isolation_forest.py` | Gắn cờ thay đổi đồng thời nhiều chỉ số |
| EWMA crossing | `trend.py` | Xu hướng tăng/giảm từng chỉ số |
| Sai số dự báo | `forecast.py` | Đột biến so với dự báo EWMA (điểm cắm sẵn Chronos/TimesFM) |

Đúng nguyên tắc giảng viên yêu cầu: **so người dùng với lịch sử của chính họ**,
không so quần thể; dưới 5 mẫu baseline = NaN → không tính z-score, đánh dấu thiếu.

### 2.2 Tầng 2 — tri thức y khoa (`src/tier2_knowledge/rules.py`)

- KB: 9 luật, 10 chỉ số, lưu `knowledge_base.json`; mỗi luật bắt buộc `rule_id`,
  `condition` (and/or lồng nhau, toán tử > < >= <= between…), `severity` 0–1,
  `specialty`, `system`, `modes`, `source_url` (link hướng dẫn gốc ESH 2018 /
  ADA / KDIGO…). Validate chặt khi ghi qua API (dòng 71–122).
- Sửa KB tại `/rules` không cần sửa code — điểm cắm cho quy trình chuyên gia duyệt.

### 2.3 Tầng 3 — tổng hợp rủi ro (`src/tier3_risk/scoring.py`)

```
total = stat×0.30 + knowledge×0.35 + ml(model)×0.25 + trend×0.10
ngưỡng: THẤP < 0.33 · TRUNG BÌNH 0.33–0.66 · CAO ≥ 0.66
an toàn lâm sàng: có luật severity ≥ 0.7 → total = max(total, 0.50)
```

Trọng số nằm công khai trong `src/config.py:25-29`. Chế độ chẩn đoán chuyên biệt
(htn/dm/ckd…) lọc luật tương ứng. Với benchmark, mỗi model nhận một điểm cuối riêng
(`score_context`) để so sánh model nào đẩy mức rủi ro — đã thử nghiệm fusion gộp
một điểm duy nhất và loại bỏ vì mất khả năng đối sánh (`docs/12` mục 3.2).

---

## 3. Tập dữ liệu huấn luyện — dựng bằng cách nào

Script `scripts/build_nhanes_dataset.py`:

1. **Tải XPT gốc từ CDC** (không qua bên thứ ba): NHANES 2015–2016, 2017–2018,
   2021–2023 (chu kỳ "2019–2020" thực chất là pre-pandemic trùng bệnh nhân với
   2017–2018 nên không gộp để tránh nhân bản mẫu).
2. **Gộp file theo SEQN** (demographics + BPX/BPXO + lab BMX/GLU/GH/SCR + questionnaire
   thuốc huyết áp); mỗi chu kỳ tự khai báo alias cột nguồn (2021–2023 đổi codebook:
   huyết áp oscillometric BPXOSY1/BPXODI1).
3. **Chuyển đơn vị**: glucose mg/dL → mmol/L (chia 18.016); huyết áp lấy trung bình
   các lần đo hợp lệ trong khoảng sinh lý (SBP 40–300, DBP 20–160).
4. **Lọc đối tượng**: tuổi 20–80, không mang thai, và phải đủ thông tin xác định
   nhãn: `(has_bp & has_lab) | has_meds`.

Kết quả: `data/datasets/nhanes_merged.csv` — **n = 16.314 dòng**, positive 7.991
(49,0%). Dataset cũ 1 chu kỳ (n = 4.949) giữ lại để đối chứng.

## 4. Nhãn được tạo như thế nào

Nhãn **quy tắc lâm sàng**, không phải tự khai (`build_nhanes_dataset.py:186-197`):

```python
htn = (systolic_bp >= 140) | (diastolic_bp >= 90) | (đang dùng thuốc huyết áp)
dm  = (hba1c >= 6.5%) | (glucose_fasting >= 7.0 mmol/L)
label = htn | dm
```

Phân bố: label_htn = 7.197, label_dm = 2.432, cả hai = 1.638.

### 4.1 Lỗi và điểm yếu của dữ liệu/nhãn (nói thẳng)

| # | Vấn đề | Hệ quả |
|---|---|---|
| D1 | **Vòng lặp nhãn–đầu vào**: nhãn dm định nghĩa bằng chính hba1c/glucose_fasting, nhãn htn định nghĩa bằng chính huyết áp đo — mà các biến này đồng thời là đặc trưng đầu vào | Model một phần học lại ngưỡng y khoa chứ không "phát hiện" bệnh tiềm ẩn → AUC cao (0.93+) phải đọc là *tái lập phân tầng theo ngưỡng*, không phải năng lực dự báo độc lập |
| D2 | **Nhiễu thuốc**: người dùng thuốc hạ áp có huyết áp đo thấp hơn thật nhưng vẫn label = 1 | Đầu vào và nhãn xung đột trên nhóm này; cần phân tích độ nhạy (bỏ `has_meds` khỏi nhãn rồi chạy lại) |
| D3 | **glucose_fasting thiếu 52%** (chỉ đo trên nhóm con nhịn ăn) | Impute median; mọi kết quả liên quan glucose phải đọc thận trọng; flag MISSING_CAO công bố trong summary |
| D4 | **Creatinine thiếu 8%**, còn lại thiếu ~5% | Impute median fit-train; egfr/spo2 không có trong dataset benchmark |
| D5 | **Dữ liệu cắt ngang**: mỗi người đúng một dòng | Không huấn luyện được chuỗi thời gian; tầng 1 chỉ chạy trên dữ liệu dọc thu thập qua chat, không liên quan benchmark |
| D6 | **Codebook khác biệt giữa chu kỳ** (BP oscillometric 2021–2023; heart_rate 60-giây thay vì trung bình 3 lần) | Nguồn phương sai giữa chu kỳ; chấp nhận vì gộp 3 chu kỳ giảm overfit một chu kỳ (phương sai AUC giảm ~3 lần) |
| D7 | **Khái quát hóa**: quần thể khảo sát Mỹ, không phải bệnh nhân Việt | Chỉ dùng để chứng minh cơ chế; chưa thay thế xác nhận lâm sàng |

Mất cân bằng lớp: positive 49% → **không** mất cân bằng nghiêm trọng; các model
vẫn đặt `class_weight="balanced"` (`src/experiments/models.py`) phòng khi tách
nhãn riêng từng bệnh.

## 5. Quy trình huấn luyện và chống rò rỉ

- **Model sản xuất**: `scripts/train_nhanes.py` — LightGBM trên `nhanes_merged`,
  CV 5-fold: **AUC 0.9356 ± 0.0016** → `data/models/risk_lgbm_real.joblib` (nạp lazy
  bởi pipeline, `src/config.py:41`).
- **Benchmark đa mô hình** (`scripts/run_benchmark.py`, protocol v1.0
  `src/experiments/protocol.py`): 6 model × 5 seed (42/52/62/72/82), split
  70/15/15 stratified, test khóa lại, **imputer median fit trên train only**
  (chống data leakage tiền xử lý), mỗi dòng là một người nên tách theo bệnh nhân
  mặc nhiên. Model chưa cài thư viện tự bỏ qua, không làm hỏng chạy.

## 6. Kết quả đo được

### 6.1 Benchmark 6 mô hình (nhanes_merged, n = 16.314, 5 seed)

| Model | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|
| **xgb** | **0.9356±0.0028** | 0.9491 | **0.0913** |
| lgbm | 0.9349±0.0031 | 0.9488 | 0.0916 |
| rf | 0.9338±0.0038 | 0.9473 | 0.0956 |
| fttransformer | 0.9257±0.0043 | 0.9405 | 0.1028 |
| mlp | 0.8975±0.0123 | 0.9117 | 0.1287 |
| lr | 0.8844±0.0078 | 0.8960 | 0.1375 |

Đọc đúng ý nghĩa: khoảng cách XGB/LGBM/RF ≤ 0.001 với std ~0.003 → **chưa đủ căn
kết luận model nào hơn model nào**; LR thấp hơn ~0.05 xác nhận phi tuyến có lợi;
Brier tốt nhất 0.091. Calibration curve 10 bins xuất kèm từng evidence package.

### 6.2 Trên ca thực tế

- Ca mẫu HA 150/95 (`docs/11`): RF luận giải đúng chiều cả SBP/DBP; MLP đảo chiều
  DBP/nhịp tim (weights cancellation) → chọn LightGBM làm model sản xuất và luôn
  hiển thị luận giải kèm mức tin cậy.
- Ca P9 sau fix impute-trước-predict: XGBoost đóng góp luật R_CV_01 +0.253
  (trước fix 0.000 do NaN lệch giữa explain và train).

### 6.3 Hệ thống đầu-đầu

Chat nhận câu tự nhiên/file → tích lũy theo ngày (dedup metric+date) → đủ 7 ngày
ra báo cáo cá nhân hóa; 1 ngày vẫn ra báo cáo sơ bộ theo luật kèm THIẾU DỮ LIỆU
được đánh dấu. Endpoint `/chat` `/rules` `/benchmark` đều hoạt động (test curl).

## 7. Ưu điểm và nhược điểm (tự đánh giá)

**Ưu điểm**
1. Luật + thống kê là lõi: hệ thống chạy được ngay cả khi không có model, không đủ
   lịch sử; ML chỉ là nguồn điểm bổ sung — đúng kiến trúc hybrid.
2. Baseline cá nhân cửa sổ trượt hướng quá khứ, không nhìn tương lai; từ chối kết
   luận khi < 5 mẫu thay vì đưa cảnh báo nhiễu.
3. Minh bạch: công thức điểm, ngưỡng, sàn an toàn, nguồn từng luật, mức thiếu dữ
   liệu đều hiển thị trong báo cáo; mọi suy luận ML gắn disclaimer bác sĩ quyết cuối.
4. Quy trình thực nghiệm chống leakage (impute fit-train, test khóa, cùng split
   giữa các model, 5 seed) và evidence package truy ngược được.
5. KB quản trị được qua UI với validate; ingest PDF/DOCX/TXT hoạt động offline.

**Nhược điểm** (thẳng thắn)
1. Vòng lặp nhãn–đầu vào (D1) khiến AUC không thể diễn giải thành năng lực dự báo
   độc lập — đây là giới hạn lớn nhất của bằng chứng hiện tại, chỉ khắc phục bằng
   nhãn từ biến cố tương lai (dữ liệu dọc) hoặc nhãn độc lập với đầu vào.
2. Chưa có hiệu chỉnh xác suất hậu kỳ (isotonic/Platt) dù đã đo calibration/Brier.
3. Tầng 2 phụ thuộc chất lượng nguồn và chưa có phiên bản luật + audit trail +
   trạng thái chuyên gia duyệt.
4. Không mô hình hóa quan hệ thời gian dài (LightGBM dữ liệu bảng); LSTM/foundation
   model EHR mới là điểm cắm, chưa có dữ liệu dọc đủ để huấn luyện.
5. Chưa có validation theo thời gian và xác nhận ngoài; lead time cảnh báo chưa đo
   được trên dữ liệu cắt ngang.

## 8. Kết luận

Hệ thống là một khung hỗ trợ quyết định lâm sàng hybrid hoàn chỉnh đầu-cuối (ingest
→ 3 tầng → báo cáo/API/UI) với quy trình thực nghiệm nghiêm túc. Kết quả benchmark
cho thấy các model boosting tái lập tốt phân tầng nguy cơ theo ngưỡng lâm sàng
(AUC ~0.94, Brier 0.09) nhưng **bằng chứng hiện tại dừng ở phân tầng cắt ngang**.
Bước đi kế tiếp có thứ tự ưu tiên: hiệu chỉnh xác suất, phân tích độ nhạy nhãn,
validation theo thời gian khi có dữ liệu dọc (kênh chat thu thập đang là nguồn),
quản trị phiên bản tri thức, rồi mới đến mô hình chuỗi thời gian.

---

*Tài liệu mô tả trạng thái mã nguồn tại thời điểm viết; mọi đường dẫn file và số
liệu truy xuất được trong repo. Không thay thế chẩn đoán của bác sĩ.*
