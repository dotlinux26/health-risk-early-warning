# Ghi chép: Phân loại mô hình & chiến lược thực nghiệm (nhóm 3 tầng)

> Ghi chép nội bộ — chốt **taxonomy mô hình** và **cách tổ chức thực nghiệm**
> để vừa có chiều sâu nghiên cứu, vừa không biến đề tài thành cuộc chạy
> benchmark mô hình. Phần 80–100M model được đặt đúng vai trò **extension
> giải thích**, không tham gia cạnh tranh dự đoán nguy cơ.

---

## 1. Taxonomy mô hình (chốt)

| Nhóm | Model | Vai trò trong luận văn | Mức ưu tiên |
|---|---|---|---|
| Baseline | Logistic Regression | baseline tuyến tính | Bắt buộc |
| Ensemble | Random Forest | baseline tree ensemble | Bắt buộc |
| Boosting | XGBoost | đối chứng boosting | Bắt buộc |
| Boosting | **LightGBM** | model hiện tại / ứng viên chính | Bắt buộc |
| Neural | MLP | neural baseline | Bắt buộc |
| Transformer | **FT-Transformer** | so sánh deep-learning trên tabular | Nên có |
| Transformer | TabNet | thêm nếu còn thời gian | Tùy chọn |
| Foundation tabular | TabPFN | thêm nếu môi trường chạy ổn | Tùy chọn |
| Reasoning | **80–100M model** | giải thích / diễn giải bằng chứng | **Extension** |

Chuỗi đủ để trình bày: **linear → tree → boosting → neural → transformer → reasoning**,
trong khi câu chuyện chính vẫn là **framework**, không phải "benchmark model nào
mạnh nhất".

---

## 2. Vai trò của 80–100M model — đặt đúng vị trí

Không cho nó cạnh tranh với LightGBM / FT-Transformer về ROC-AUC. Nó nằm ở
**cuối luồng**, sau FUSION, nhận toàn bộ bằng chứng và sinh lời giải tự nhiên:

```text
 Patient Features ──► ML Risk Score
                         │
   Statistical  ◄──┐     │
   Medical Rules ◄─┼──► FUSION ──► Final Risk ──► Evidence
   Trend         ◄──┘     │
                         ▼
                  80–100M model ──► Natural-language rationale
```

- Model nhỏ này **không quyết định risk** — nó chỉ trả lời *"Tại sao hệ thống
  đưa ra cảnh báo này?"* dựa trên evidence đã có.
- Nếu nó giải thích kém → **risk model vẫn hoạt động bình thường**.
- Kiến trúc này là **prediction ≠ explanation** — điểm dễ bảo vệ trước hội đồng.

### Đánh giá model giải thích — không dùng ROC-AUC

| Tiêu chí | Nội dung |
|---|---|
| Faithfulness | Giải thích đúng những evidence framework thực sự đưa vào không? |
| Completeness | Có bỏ sót evidence quan trọng không? |
| Consistency | Cùng một evidence → các lần chạy giải thích tương tự nhau không? |
| Hallucination | Có tự bịa thông tin (ví dụ "bệnh nhân có tiền sử X") ngoài input không? |

LLM chỉ được phép giải thích dựa trên evidence đầu vào (ml_risk, stat_anomaly,
trend, rules). Kiểm tra bằng cách phân loại câu: supported / unsupported / missing
/ contradiction — đây có thể trở thành một **experimental safety check**.

---

## 3. Tổ chức thực nghiệm theo 3 tầng

Thay vì "9 model × full protocol" (dễ chết deadline), chia thành:

### Tầng A — Core benchmark (6 model)

Logistic Regression, Random Forest, XGBoost, LightGBM, MLP, **FT-Transformer**.

- Đây là bảng chính của Chương 3.
- FT-Transformer đại diện hướng Transformer cho tabular data — đủ để nói:
  *"các nhóm tuyến tính, ensemble tree, gradient boosting, neural network và
  Transformer được khảo sát"* mà không cần TabNet.

### Tầng B — Framework experiment (thí nghiệm chính của đề tài)

Chọn **model tốt nhất từ Tầng A** làm ML component, rồi chạy:

```text
EXP-F01   ML only
EXP-F02   Statistical only
EXP-F03   Rule only
EXP-F04   Trend only
EXP-F05   Statistical + Rule
EXP-F06   Statistical + Rule + Trend
EXP-F07   Full Fusion
```

Bản rút gọn (khi cần giảm workload):

```text
ML-only
Stat+Rule+Trend
Full-Fusion
```

Đây mới là thí nghiệm chứng minh **contribution của fusion**, không phải Tầng A.

### Tầng C — Extension

Nhét TabNet, TabPFN, 80–100M reasoning model. **Không bắt buộc hoàn thiện:**

- Kịp FT-Transformer → luận văn đã hoàn chỉnh.
- Kịp thêm 80M → có subsection "Explainability Extension".
- Không kịp → bỏ extension khỏi luận văn chính, framework vẫn đứng vững.

Cách này là **bảo hiểm deadline**.

---

## 4. Nguyên tắc đánh giá công bằng (sửa lại so với bản cũ)

Không bắt mọi model dùng y hệt một preprocessing nếu preprocessing đó không phù
hợp (tree vs neural). Điều cần khóa là:

> **same dataset + same target + same information boundary + same
> train/validation/test partitions**

không nhất thiết **same preprocessing implementation**.

```text
Raw NHANES
   ├── common inclusion/exclusion
   ├── common target
   └── common split
          │
    ┌─────┴─────┐
    ▼           ▼
 Tree pipeline Neural pipeline
    │           │
 LightGBM   FT-Transformer
    │           │
    └─────┬─────┘
          ▼
      Test set
```

---

## 5. Đa seed giữ nguyên

5 seed: **42, 52, 62, 72, 82** → báo `μ ± σ`. Bảng kết quả chỉ là template,
chưa điền số:

| Model | ROC-AUC | PR-AUC | F1 | Brier |
|---|---|---|---|---|
| LR | μ ± σ | μ ± σ | μ ± σ | μ ± σ |
| RF | μ ± σ | μ ± σ | μ ± σ | μ ± σ |
| XGB | μ ± σ | μ ± σ | μ ± σ | μ ± σ |
| LGBM | μ ± σ | μ ± σ | μ ± σ | μ ± σ |
| MLP | μ ± σ | μ ± σ | μ ± σ | μ ± σ |
| FT-Transformer | μ ± σ | μ ± σ | μ ± σ | μ ± σ |

**Test set cuối cùng vẫn khóa** — không được lấy 5 seed rồi vô tình biến test
set thành validation.

---

## 6. Chốt phạm vi theo thời gian

| Tình huống | Phạm vi |
|---|---|
| Deadline căng | Core: LR + RF + XGB + LGBM + MLP + FT-Transformer + fusion experiment. Không cần TabNet/TabPFN/LLM |
| Còn thời gian | Thêm TabPFN hoặc TabNet |
| Còn nhiều thời gian | Thêm 80–100M reasoning model, gọi là **Evidence-grounded explanation module** (không gọi là model dự đoán nguy cơ) |

---

## 7. Lý do không chạy "20 model"

Câu trả lời khoa học khi hội đồng hỏi *"Sao không thử 20 model?"*:

> Nghiên cứu không nhằm benchmark toàn bộ ML ecosystem; việc chọn đại diện theo
> các họ mô hình (tuyến tính, tree, boosting, neural, transformer) nhằm đánh giá
> **khả năng thay thế model của framework** trong điều kiện dữ liệu và tài nguyên
> xác định. Mô hình 80–100M là **extension cho explainability**, không phải ứng
> viên cạnh tranh trong bài toán dự đoán nguy cơ.

---

*Ghi chép nội bộ. Không thay thế chẩn đoán của bác sĩ.*
