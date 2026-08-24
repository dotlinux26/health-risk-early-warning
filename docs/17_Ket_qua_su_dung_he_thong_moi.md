# 17. Kết quả sử dụng hệ thống sau P0–P1 — báo cáo chạy thực tế

> Tài liệu ghi lại kết quả khi sử dụng trực tiếp hệ thống (giao diện biểu mẫu mới
> tại `/`, ngày 23–24/08/2026) trên bốn tab: Đánh giá, Bản ghi cá nhân, Luật &
> Quản trị, Benchmark. Mọi con số dưới đây là **đầu ra thật của server đang
> chạy**, không phải mô phỏng; đối chiếu chéo với các thí nghiệm nền tảng trong
> `docs/15` §5K và danh mục vấn đề `docs/16`.

---

## 1. Kịch bản 1 — Đánh giá ca P001 (người khỏe, lịch sử dài)

**Đầu vào:** P001, 120 ngày dữ liệu, đủ 4 chỉ số đo thường xuyên (HA tâm
thu/trương, nhịp tim, BMI); chế độ luật: tất cả chuyên khoa.

**Đầu ra của hệ thống:**

| Thành phần | Giá trị |
|---|---|
| Mức nguy cơ | **THẤP — score 0.067** |
| Tầng 1 | 4/4 chỉ số trong ngưỡng; z-score lớn nhất +1.18σ (nhịp tim), còn lại ≤ +0.75σ; xu hướng "ổn định" toàn bộ |
| Tầng 2 | Không có luật nào kích hoạt |
| Tầng 3 | ML = 0.27 (đã hiệu chỉnh isotonic) |
| Tổng hợp | stat 0×0.30 + knowledge 0×0.35 + ml 0.27×0.25 + trend 0×0.10 = **0.067 → THẤP** |

**Luận điểm cần thấy:** ca này minh họa đúng thiết kế hybrid — ML đưa tín hiệu
dương tính nhẹ (0.27) nhưng không có luật lâm sàng nào và không có bất thường cá
nhân nên tổng hợp vẫn THẤP thoải mái dưới ngưỡng 0.33. Panel hiển thị đầy đủ
phạm vi dữ liệu ("120 ngày") và disclaimer phân tầng cắt ngang. Panel Tầng 3
giải thích bằng chứng: phương pháp hiệu chỉnh isotonic, Brier/ECE raw vs
calibrated, ROC-AUC test kèm cảnh báo "không phải dự báo biến cố".

## 2. Kịch bản 2 — Quản lý bản ghi cá nhân (U10)

Bảng theo ngày hoạt động như thiết kế:

- Cột hiển thị điều khiển được: người dùng bật/tắt từng chỉ số trong 10 chỉ số
  hệ thống; các cột không có dữ liệu hiển thị placeholder đơn vị đo (mmHg,
  mmol/L…) — ô trống là trạng thái hợp lệ.
- Sửa một ô tự lưu khi rời ô; nút × xóa một ô; 🗓 xóa cả ngày
  (`DELETE /api/records/{pid}?timestamp=...`).
- Dữ liệu demo đã seed qua `scripts/seed_demo_data.py`: P001–P005 (từ
  `data/sample_long.csv`, 120→5 ngày) và hai ca tổng hợp DEMO_HYPERTENSIVE /
  DEMO_DIABETIC (45 ngày, có cú sốc 7 ngày cuối).

Kết quả kiểm tra trên ca seeded (chạy e2e 32/32 PASS):

| Ca | Ngày | Mức | Điểm | Luật kích hoạt | Ghi chú |
|---|---|---|---|---|---|
| P001 | 120 | THẤP | 0.067 | 0 | người khỏe |
| P003 | 90 | TRUNG BÌNH | 0.53 | 3 | |
| P005 | 5 | THẤP | 0.239 | 0 | < 7 ngày → **không có z-score**, đúng quy tắc K3 |
| DEMO_HYPERTENSIVE | 45 | **CAO** | 0.699 | R_CV_01, R_CV_03, R_MET_01 | Tầng 1 bắt z = **+2.37σ** HA tâm thu, flagged |
| DEMO_DIABETIC | 45 | **CAO** | 0.721 | 3 luật | trục nội tiết |

Ca DEMO_HYPERTENSIVE chứng minh trọn vẹn chuỗi giá trị: dữ liệu dọc tự sinh →
baseline cá nhân phát hiện lệch +2.4σ so với trung bình của chính anh ta → luật
ESC/ESH kích hoạt → fusion nâng lên CAO.

## 3. Kịch bản 3 — Luật & Quản trị (S5 hoàn chỉnh)

Tab quản trị native toàn màn hình hiển thị 9 luật active v1 kèm nguồn
(ESC/ESH 2018, ADA 2023, KDIGO 2022, WHO). Audit trail thực tế ghi nhận đủ chu kỳ
sống của một luật kiểm thử R_E2E_TEST:

```
create (v1, draft) → review → approved → active (approved_by=tester)
→ edit severity 0.5→0.6 (v1.0→v1.1, status_reset_to=draft)
→ delete
```

Các ràng buộc an toàn được xác nhận trong lần chạy này:

- Chỉ luật `active` tham gia chấm điểm production; luật draft/approved chỉ preview
  (kiểm chứng: luật test draft KHÔNG xuất hiện trong evidence của assess).
- Chặn nhảy cóc draft→active (API từ chối, yêu cầu đi đủ luồng).
- Mọi thao tác đều có actor + timestamp trong `data/kb/audit_log.jsonl`.

## 4. Kịch bản 4 — Trang nghiên cứu Benchmark

### 4.1 Hiệu chỉnh xác suất (T1 docs/16)

| Model | Brier raw | Brier isotonic | ECE raw | ECE isotonic |
|---|---|---|---|---|
| XGBoost | 0.0913 | 0.0914 | 1.49% | 1.50% |
| LightGBM | 0.0916 | 0.0920 | 1.60% | 1.63% |
| Random Forest | 0.0956 | **0.0936** | 4.46% | **1.69%** |
| FT-Transformer | 0.1028 | 0.1020 | 2.65% | 1.52% |
| MLP | 0.1287 | **0.1241** | 5.37% | **2.12%** |
| Logistic Regression | 0.1375 | **0.1329** | 5.12% | **1.84%** |

Đọc đúng: boosting vốn cân bằng sẵn nên isotonic trung tính; RF/MLP/LR giảm ECE
xuống ~1.5–2% — đúng như thiết kế chọn theo Brier validation. Production áp
calibrator cho mọi dự đoán (§2.1 docs/16).

### 4.2 Độ bền vững K2–K4 (T2/T5/T6 docs/16)

- **K2**: nhãn B (bỏ thuốc) AUC = 1.000 vs nhãn A = 0.935 (LGBM/XGB); 2968/16314
  ca đổi nhãn; oracle nhãn-B-xếp-nhãn-A = 0.815 ± 0.009 < model 0.935 → model
  học thêm tín hiệu thuốc từ mẫu đo; kết quả dừng ở **phân tầng cắt ngang**.
- **K3**: bảng cửa sổ 3→90 ngày với kết luận UI từng dòng (< 7 ngày CHƯA ỔN ĐỊNH;
  ≥ 14 ổn định hơn) — quy tắc đã chạy thật ở kịch bản 2 (ca P005).
- **K4**: ba bộ trọng số đồng thuận mức 96.1–99.4%, không có chuyển mức lên CAO.

### 4.3 Evidence status checklist

Trạng thái công khai ngay trên trang: ✓ nội bộ / ✓ baseline cá nhân / ✓ K2 /
◐ calibration (đã fit val, chưa re-fit external) / ✓ K3 / ✓ K4 /
○ temporal / ○ external / ○ thử nghiệm lâm sàng — khớp 1:1 với bảng T1–T11
trong docs/16.

### 4.4 So sánh luận giải cùng một ca (165/95, HR 88, glucose 100…)

Cả 6 model đều xếp **CAO** nhưng cơ cấu đóng góp khác nhau — đây chính là lý do
hệ thống giữ đa model trong benchmark:

| Model | Score | Đóng góp luật chủ lực | Đặc trưng dẫn |
|---|---|---|---|
| LR | 0.9894 | R_CV_03 +0.273 (tâm thu đơn độc) | systolic_bp +0.273 |
| RF | 0.9931 | R_CV_01 +0.538 | systolic_bp +0.117 |
| LGBM | 0.9999 | R_CV_01 +0.523 | systolic_bp +0.010 |
| XGB | 0.9998 | R_CV_01 +0.574 | systolic_bp +0.018 |
| MLP | 0.9927 | R_CV_01 +0.552, R_CV_03 +0.523 | systolic_bp +0.523 |
| FT-Transformer | 1.0000 | R_CV_01 +0.685 | attribution ≈ 0 |

Ghi nhận hai quan sát kỹ thuật: (a) LR phân bổ qua R_CV_03 trong khi tree ensembles
qua R_CV_01 — khác biệt cấu trúc tuyến tính/phi tuyến đáng giảng dạy; (b)
FT-Transformer cho attribution đặc trưng gần 0 dù score max — perturbation khó
làm thay đổi attention, khẳng định lựa chọn **LightGBM làm model production vì
tính giải thích được** (docs/11).

### 4.5 Chi tiết thí nghiệm

30 evidence package (6 model × 5 seed) hiển thị kèm metrics riêng và đường
ROC/PR/calibration — truy vết từng con số trong bảng tổng hợp về đúng lần chạy.

## 5. Đối chiếu với danh mục docs/16

Lần sử dụng thực tế này củng cố trạng thái đã chốt:

| Vấn đề | Trạng thái qua phiên dùng thực tế |
|---|---|
| T1 Calibration | ✓ hiển thị + áp vào fusion (ML 0.27 của P001 là điểm đã hiệu chỉnh) |
| T2 Label sensitivity | ✓ bảng K2 render trực tiếp từ evidence package |
| T3 Governance | ✓ audit trail nhìn thấy được, luồng chuẩn hoạt động |
| T5 Baseline cửa sổ | ✓ ca P005 bị chặn z-score theo đúng quy tắc |
| T6 Weight sensitivity | ✓ bảng K4 + kết luận |
| T7/T10 Temporal & external | ○ checklist minh bạch — chưa làm, đúng cam kết docs/16 §5 |

## 6. Hạn chế quan sát được trong phiên chạy

1. Panel Tầng 3 trước bản vá hiển thị "Đầu ra thô: xem ô Đánh giá" vì chưa truyền
   `ml_score` vào endpoint evidence — đã sửa (`app.html` truyền `?score=`).
2. R_CV_01 dùng AND (sbp > 140 **AND** dbp > 90) trong khi ESC/ESH định nghĩa
   tăng huyết áp là OR — ca chỉ cao một loại vẫn được R_CV_03/R_KID… bắt riêng,
   nhưng về lâu dài nên tách R_CV_01 thành OR hoặc bổ sung rule OR rõ ràng
   (việc này giờ làm được ngay trên UI governance mà không sửa code).
3. Audit trail chưa có chức năng xuất CSV (hiện xem trên trang, dữ liệu JSONL
   đầy đủ trong repo).
4. Bảng bản ghi hiển thị tối ưu cho desktop; mobile cần cuộn ngang (đã chấp nhận
   trong phạm vi đồ án).

---

*Kết quả ghi lại từ phiên sử dụng ngày 23–24/08/2026 trên server cục bộ; e2e
32/32 PASS. Mọi con số truy xuất được tới API/evidence package tương ứng. Không
thay thế chẩn đoán của bác sĩ.*
