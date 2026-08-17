# Ghi chép: Quy hoạch lại cấu trúc source code theo 3 tầng + điểm cắm Tầng 4 (LLM)

> Ghi chép nội bộ — đề xuất tách lại cấu trúc thư mục để các module chức năng
> gắn đúng 3 tầng kiến trúc (stat → rule → ML+fusion), import được và tái sử
> dụng được, đồng thời dự phòng sẵn một **điểm cắm tầng 4 (LLM giải thích)**.
> Ghi lại để cả nhóm làm cùng một mục tiêu, không đụng nhau khi sửa code.

---

## 1. Hiện trạng (tôi đã rà lại toàn bộ source)

Cấu trúc hiện tại đã có 3 nhánh tầng nhưng **chưa tách bạch hoàn toàn**:

```
src/
├── tier1_anomaly/     # Tầng 1 — phát hiện bất thường cá nhân hóa (OK)
│   ├── detector.py    # run_tier1() — điểm vào tổng hợp
│   ├── zscore.py, trend.py, forecast.py, isolation_forest.py
│   └── __init__.py    # định nghĩa AnomalyRecord (dataclass dùng chung)
├── tier2_knowledge/   # Tầng 2 — tri thức y khoa (OK)
│   ├── rules.py       # KnowledgeBase, RuleHit, CRUD
│   └── knowledge_base.json
├── tier3_risk/        # Tầng 3 — tổng hợp rủi ro (gần OK, còn lẫn ML)
│   ├── scoring.py     # RiskScorer — fusion điểm + báo cáo chi tiết
│   └── report.py      # render markdown/json
├── models/            # ❌ LẪN LỘN — không nằm trong tầng nào
│   ├── risk_model.py  # RiskModel (load/predict LightGBM sản xuất)
│   ├── train.py, synthetic_data.py, explain.py
├── experiments/       # tách được rồi (benchmark, protocol, view)
├── data/  ingest/  chat/  config.py  main.py  api.py
```

### 1.1. Vấn đề hiện tại

1. **`src/models/` đứng ngoài 3 tầng** — nhưng về bản chất nó là **Tầng 3.1
   (ML risk model)**: `RiskScorer` (tier3) hiện **không nhận model từ ngoài**,
   mà `main.py` load `RiskModel` rồi truyền `ml_score` vào qua
   `assess_patient(..., scorer=...)` — nói cách khác ML bị **bơm ngang qua
   tham số**, không phải là một thành phần import-able gọn gàng của Tầng 3.
2. **Phụ thuộc ngược chiều**: `tier2/rules.py` import `AnomalyRecord` từ
   `tier1` — chấp nhận được (cùng chiều), nhưng cần chuẩn hoá quy tắc để ai
   mới vào cũng không import bậy.
3. **Tầng 4 chưa có chỗ**: LLM giải thích (80–100M model) về nguyên tắc là
   **extension sau FUSION**, nhưng hiện chưa có interface để cắm vào mà không
   đụng `agent.py`/`report.py`.
4. `chat/agent.py` và `chat/parser.py` đang vừa gọi tầng 1–3 vừa lo UI — nên
   tách rõ: agent chỉ **điều phối**, không cài logic tầng nào bên trong.

## 2. Quy hoạch mục tiêu — nguyên tắc

1. **Một module = một trách nhiệm**, nằm đúng tầng, import **cùng chiều**:
   `tier1 → tier2 → tier3 → tier4`. Tầng dưới không bao giờ import tầng trên.
2. **Mỗi tầng có điểm vào công khai** (public API) để `api.py`, `main.py`,
   `chat/agent.py`, notebook, script **import và tái sử dụng** — không phải sao
   chép code.
3. **Tầng 4 là tuỳ chọn (pluggable)**: hệ thống chạy đầy đủ khi tầng 4 vắng
   mặt. Có nó thì thêm lời giải thích tự nhiên, không có nó thì vẫn ra báo cáo.
4. **Dữ liệu truyền giữa tầng là dataclass có sẵn** (`AnomalyRecord`, `RuleHit`,
   `RiskResult`...) — không truyền dict lỏng lẻo giữa tầng, tránh typo.

## 3. Cấu trúc đích

```
src/
├── core/                       # cấu hình + kiểu dữ liệu dùng chung
│   ├── config.py               # Config, CONFIG (chuyển từ src/config.py)
│   ├── types.py                # AnomalyRecord, RuleHit, RiskResult (dataclass)
│   └── pipeline.py             # assess_patient() — điều phối 3 tầng, 1 chỗ gọi
│
├── tier1_anomaly/              # giữ nguyên bộ phát hiện
│   ├── detector.py             # run_tier1(df, value_cols, config) -> [AnomalyRecord]
│   └── zscore.py / trend.py / forecast.py / isolation_forest.py
│
├── tier2_knowledge/
│   ├── rules.py                # KnowledgeBase (load/CRUD/evaluate)
│   └── knowledge_base.json
│
├── tier3_risk/
│   ├── ml/                     # ML risk model — NHẬP từ src/models vào đây
│   │   ├── model.py            # RiskModel (load/predict)
│   │   └── training.py         # train_nhanes, train_real_datasets, explain
│   ├── scoring.py              # RiskScorer — fusion
│   └── report.py               # render markdown/json
│
├── tier4_explain/              # TẦNG 4 — MỚI, tuỳ chọn, pluggable
│   ├── base.py                 # Explainer (interface/protocol)
│   └── (chưa có triển khai — để chỗ cắm model 80–100M sau này)
│
├── experiments/                # giữ nguyên (benchmark tách khỏi tầng)
├── data/  ingest/  chat/       # giữ nguyên, agent chỉ điều phối
└── api.py  main.py
```

## 4. Điểm cắm Tầng 4 (interface thiết kế sẵn)

```python
# src/tier4_explain/base.py
class Explainer(Protocol):
    def explain(self, context: dict) -> str:
        """Sinh lời giải thích tự nhiên từ bộ bằng chứng fusion.

        context = {
          "ml_risk": float, "stat_anomalies": [...],
          "rules": [...], "trend": [...], "final_risk": float,
          "level": "CAO|TRUNG_BINH|THAP", "features": {...},
        }
        """
```

- **Fusion (Tầng 3) KHÔNG phụ thuộc Tầng 4**: `RiskScorer` sinh `context`
  đầy đủ; nếu `tier4` vắng mặt thì bỏ qua bước giải thích tự nhiên.
- `report.py` / `agent.py` gọi **qua interface** (`Explainer`), không gọi thẳng
  một model cụ thể → sau này cắm model 80–100M nào cũng được, đổi không chạm
  tầng dưới.
- Đánh giá Tầng 4 bằng Faithfulness / Completeness / Consistency /
  Hallucination (đã chốt ở `notes/02`), **không dùng ROC-AUC**.

## 5. Thứ tự làm (đề xuất, tách theo từng bước nhỏ)

> **Cập nhật 17/08**: đã triển khai xong Bước 1–5 (xem mục 7). Bước 6 còn lại là
> xoá file `src/config.py` cũ nếu muốn, sau khi chắc chắn không ai import.

| Bước | Việc | Rủi ro nếu đổi tên |
|---|---|---|
| 1 | Tạo `src/core/types.py`, chuyển `AnomalyRecord` từ `tier1/__init__.py` | import cũ `from src.tier1_anomaly import AnomalyRecord` vẫn giữ alias để không vỡ |
| 2 | Tạo `src/core/config.py` (re-export từ `src/config.py`) | giữ file cũ re-export để tương thích |
| 3 | Tách `src/models/*` → `src/tier3_risk/ml/` | cập nhật import trong main/api/experiments |
| 4 | Tạo `src/tier4_explain/base.py` (interface) | chỉ thêm mới, không phá gì |
| 5 | Chuyển logic fusion về `src/core/pipeline.py` (1 điểm gọi duy nhất) | main.py/api.py trỏ về đây |
| 6 | Xoá alias tạm sau khi tất cả script chạy lại OK | — |

## 6. Giới hạn / rủi ro cần nói thẳng

1. **Bước 3 và 5 là rủi ro nhất** (đổi import, đụng nhiều file). Tôi khuyến nghị
   làm từng bước, mỗi bước chạy lại `run_api.sh restart` + test curl các
   endpoints trước khi qua bước kế.
2. **Đổi tên thư mục `src/models/`** có thể gây lỗi ở script cũ không nằm trong
   danh sách đã rà (`scripts/*`, notebook nếu có) — phải grep toàn repo trước.
3. **Tầng 4 hiện chỉ có interface, chưa có model thật** — không hứa hẹn kết quả
   LLM trong giai đoạn này; đây là dự phòng kiến trúc, không phải tính năng.
4. Việc này **không đổi hành vi nghiệp vụ** — chỉ tái cấu trúc (refactor).
   Nếu kết quả hệ thống thay đổi sau khi refactor là do lỗi, không phải do thiết
   kế.

---

*Ghi chép nội bộ. Không thay thế chẩn đoán của bác sĩ.*

## 7. Kết quả triển khai (17/08)

Đã triển khai xong toàn bộ refactor Bước 1–5:

1. **`src/core/types.py`** — chuyển `AnomalyRecord` về đây; `tier1/__init__.py`
   giữ alias `from src.core.types import AnomalyRecord` để không vỡ import cũ.
   **Lưu ý**: `src/core/__init__.py` cố ý KHÔNG import pipeline (tránh vòng lặp
   core → tier1 → core).
2. **`src/core/config.py`** — re-export `Config, CONFIG` từ `src/config.py`.
3. **`src/tier3_risk/ml/`** — đã chuyển `risk_model.py, train.py, synthetic_data.py,
   explain.py` từ `src/models/`, xoá hẳn `src/models/`; cập nhật import trong
   `main.py`, `api.py`, `scripts/train_nhanes.py`, `scripts/train_real_datasets.py`.
4. **`src/tier4_explain/base.py`** — interface `Explainer(Protocol)` với
   `explain(context) -> str`. Đã test với một Explainer giả qua `assess_patient`,
   hệ thống vẫn chạy bình thường khi tầng 4 vắng mặt.
5. **`src/core/pipeline.py`** — chuyển `assess_patient`, `load_ml_model`,
   `VALUE_COLUMNS` từ `main.py`; `main.py` và `api.py` giờ gọi qua đây.
   `assess_patient` nhận thêm tham số tuỳ chọn `explainer` → trả
   `natural_explanation` (lỗi tầng 4 không phá luồng chính).

**Kiểm chứng sau refactor**: import toàn repo OK, `python -m src.main` chạy được
trên `data/sample_long.csv`, server restart + 4 trang `/chat /rules /benchmark /
api/health` trả 200, endpoint `/api/benchmark/explain` trả đầy đủ results +
clinical context + rule attribution.

Ngoài ra, bổ sung thêm cho giao diện `/benchmark` (theo yêu cầu mở rộng):

- **Form nhập động 10 chỉ số** (không còn cố định 7): systolic_bp, diastolic_bp,
  heart_rate, glucose, glucose_fasting, hba1c, creatinine, egfr, spo2, bmi —
  lấy từ KB (`/api/benchmark` trả `input_metrics`).
- **Điểm nguy cơ theo từng luật** cho mỗi model: với mỗi luật kích hoạt, đặt
  đúng các chỉ số luật dùng về baseline quần thể → đo độ giảm điểm = mức đóng
  góp của luật đó (trường `rule_attribution`). Cùng ca SBP 165/DBP 95:
  RF/LGBM/XGB/FTT gán R_CV_01 (THA) ~0.6–0.7, còn MLP lại đảo lạ (R_CV_03 +0.67,
  R_MET_01 âm) — minh chứng thêm lỗi luận giải của MLP.