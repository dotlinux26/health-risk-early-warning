# Ghi chép: Mở rộng chẩn đoán (1 chỉ số + chế độ chuyên biệt) và dataset NHANES nhiều chu kỳ

> Ghi chép nội bộ — đợt nâng cấp 17/08, gồm 3 trụ:
> (1) **đánh giá tức thì** — chẩn đoán được ngay khi có ≥ 1 chỉ số (máy đo HA
> ở nhà, cân tự đo), không bắt buộc đủ chuỗi 7 ngày;
> (2) **chế độ chẩn đoán chuyên biệt theo bệnh** — htn / dm / ckd / resp / met...;
> (3) **dataset thật lớn hơn, nhiều năm** — NHANES ghép 3 chu kỳ, n = 16.314.

---

## 1. Lý do (nói thẳng)

Người dùng thực tế ở nhà thường chỉ có **một thiết bị** (máy đo huyết áp, cân
điện tử), đo **một chỉ số** bất kỳ lúc nào — không phải đi khám sức khỏe đầy đủ
7 ngày liên tục như pipeline chuỗi thời gian mặc định. Hệ thống cũ chặn đánh giá
khi chưa đủ 7 điểm, khiến thiết bị ở nhà gần như vô dụng. Đợt này nới lại:

1. **Chẩn đoán tức thì từ snapshot** — có ≥ 1 chỉ số là đánh giá ngay theo luật
   y khoa (Tầng 2), không cần chờ chuỗi thời gian.
2. **Chế độ chuyên biệt theo bệnh** — khi người dùng chỉ quan tâm một nhóm bệnh
   (vd chỉ theo dõi tăng huyết áp), hệ thống lọc đúng luật của hệ cơ quan đó
   thay vì trộn cả 5 hệ vào một báo cáo.
3. **Dữ liệu lớn hơn** — gộp nhiều chu kỳ NHANES để huấn luyện trên dân số rộng
   và nhiều năm hơn (giảm phương sai, tăng tính đại diện).

## 2. Triển khai đánh giá tức thì (snapshot)

### 2.1. Về bản chất đã hỗ trợ sẵn một phần

- `RiskScorer.score(records=[], snapshot={...})` **đã** chạy `kb.evaluate(snapshot)`
  — Tầng 2 kích hoạt luật từ snapshot mà không cần lịch sử.
- `assess_patient` trả `INSUFFICIENT_DATA` chỉ khi snapshot **rỗng**; còn ≥ 1
  chỉ số là chạy bình thường.
- Rào cản thật nằm ở **chat agent**: `status.ready = unique_dates >= min_points(7)`
  — chưa đủ 7 ngày thì chỉ trả `_quick_snapshot` ngắn gọn, không có báo cáo.

### 2.2. Sửa đổi

`src/chat/agent.py` — trong `_reply_after_record`, khi **chưa đủ** 7 ngày:

- Gọi thẳng `_assess_df(df)` (pipeline 3 tầng) thay vì chỉ `_quick_snapshot`;
- Kết quả được bọc thành **BÁO CÁO SƠ BỘ** (khác tiêu đề với báo cáo đầy đủ
  khi đủ chuỗi thời gian) + dòng nhắc gửi thêm ngày đo;
- Nếu pipeline trả `INSUFFICIENT_DATA` (không có chỉ số nào) mới fall về
  `_quick_snapshot` như cũ.

Kết quả: gửi **một lần** "Huyết áp 158/96" → ngay lập tức có BÁO CÁO SƠ BỘ:
mức rủi ro, hệ cơ quan, luật kích hoạt (R_CV_01, R_CV_03), chỉ số theo dõi.

### 2.3. Giới hạn phải nói rõ

- **Không cá nhân hóa** theo lịch sử — không có Tầng 1 (z-score, xu hướng, dự
  báo) vì chưa đủ 7 điểm; báo cáo sơ bộ dựa **chỉ trên tri thức y khoa + ML
  snapshot**, chưa phản ánh biến thiên của từng người.
- **Mức rủi ro không so sánh được** với báo cáo đầy đủ — đây là đánh giá khởi
  điểm để gợi ý theo dõi, không thay thế chẩn đoán.

## 3. Chế độ chẩn đoán chuyên biệt (mode)

### 3.1. Thiết kế

Bảng ánh xạ mode → hệ cơ quan (`src/tier2_knowledge/rules.py`):

| Mode | Ý nghĩa | Hệ cơ quan |
|---|---|---|
| `htn` | tăng huyết áp | tim_manh |
| `cv` | tim mạch tổng hợp | tim_manh |
| `dm` | đái tháo đường | noi_tiet |
| `endo` | nội tiết + chuyển hóa | noi_tiet, chuyen_hoa |
| `ckd` | suy thận mạn | than |
| `resp` | hô hấp / SpO2 | ho_hap |
| `met` | chuyển hóa / cân nặng | chuyen_hoa |
| `all` | mọi luật (mặc định) | — |

Mỗi luật trong KB thêm field `modes` (danh sách). `evaluate(snapshot, modes)`
lọc theo `rule["system"]` trước khi đối chiếu. `normalize_modes()` chuẩn hoá
None / "all" → xét tất cả.

### 3.2. Nhận diện từ câu nói tự nhiên

`agent._detect_modes(message)` quét từ khoá tiếng Việt:
"tăng huyết áp / huyết áp" → `htn`; "tiểu đường / đái tháo đường / đường
huyết" → `dm`; "thận" → `ckd`; "hô hấp / spo2" → `resp`; "cân nặng / bmi" → `met`.

Ví dụ đã test:
- "đánh giá nguy cơ tăng huyết áp: huyết áp 152/94" → mode htn, chỉ 2 luật tim
  mạch được xét.
- "đánh giá nguy cơ tiểu đường: đường huyết đói 7.8" → mode dm, chỉ R_END_01.
- "/api/assess" nhận thêm `"mode": "ckd"` (hoặc danh sách cách nhau dấu phẩy).

### 3.3. UI

- `/rules`: rule card hiển thị dòng "Chế độ: htn cv..." + form thêm/sửa luật có
  ô nhập `modes` (validate theo `DIAGNOSTIC_MODES` khi lưu qua API).
- Chat: báo cáo có dòng "Chế độ đánh giá: tăng huyết áp" + gợi ý dùng mode
  trong phần nhắc gửi thêm dữ liệu.
- KB version bump **0.2.0 → 0.3.0** (thêm modes cho 9 luật).

## 4. Dataset NHANES nhiều chu kỳ

### 4.1. Hiện trạng phát hành (tính tới 17/08/2026)

| Chu kỳ | Ký hiệu | Trạng thái |
|---|---|---|
| 2015–2016 | I | tải được ✅ |
| 2017–2018 | J | tải được ✅ |
| 2019–2020 | P | **KHÔNG phải chu kỳ riêng** — CDC gộp vào "2017–Mar 2020 pre-pandemic", trùng bệnh nhân với J → **không thêm** để tránh trùng mẫu khi gộp |
| 2021–2023 (Aug 21–Aug 23) | L | tải được ✅ (phát hành 09/2024) |
| 2023–2024, 2025 | — | **chưa phát hành công khai** tại thời điểm viết |

### 4.2. Script `scripts/build_nhanes_dataset.py` — mở rộng

- Cấu trúc lại thành `CYCLES: dict[cycle -> {path, files, keep, aliases,
  htn_med_col}]` — mỗi chu kỳ tự khai cột nguồn.
- **Chu kỳ 2021–2023 đổi codebook**:
  - Huyết áp: `BPXSY1/BPXDI1` (chu kỳ cũ) → `BPXOSY1/BPXODI1` (mới, đo kiểu
    oscillometric);
  - Nhịp tim: `BPXPLS` → trung bình `BPXOPLS1..3`;
  - Thuốc huyết áp: `BPQ050A` → `BPQ030`;
- Lệnh: `python scripts/build_nhanes_dataset.py` (mặc định 3 chu kỳ khả dụng)
  hoặc `--cycles 2015_2016 2017_2018 2021_2023`.
- Output: `data/datasets/nhanes_merged.csv` kèm cột `cycle`.

### 4.3. Kết quả

| Đợt | Chu kỳ | n | positive |
|---|---|---|---|
| cũ | 2017–2018 | 4.949 | 2.503 (50,6%) |
| mới | 2015–2016 + 2017–2018 | 10.103 | 4.793 (47,4%) |
| mới | + 2021–2023 | **16.314** | 7.991 (49,0%) |

### 4.4. Huấn luyện lại model sản xuất

`python scripts/train_nhanes.py --data data/datasets/nhanes_merged.csv`

- CV 5-fold: **AUC 0.9356 ± 0.0016** (trước: 0.9425 ± 0.0046) — AUC thấp hơn
  chút nhưng **phương sai giảm ~3×**, phản ánh đúng việc gộp chu kỳ khác nhau
  (chu kỳ L đo BP kiểu oscillometric + thuốc khác) thay vì học quá khớp một
  chu kỳ.
- Meta ghi đúng nguồn: "NHANES (CDC/NCHS): 2015_2016, 2017_2018, 2021_2023".

## 5. API

- `POST /api/assess` — thêm field `mode`: chuỗi "htn,ckd" hoặc danh sách.
- `POST /api/assess_docs` — giữ nguyên (chưa thêm mode form).
- `/api/kb` trả `modes` theo luật (version 0.3.0).
- Chat `/api/chat` — nhận diện mode từ câu tự nhiên, báo cáo sơ bộ tức thì.

## 6. Test đã chạy

1. Pipeline trực tiếp: 1 lần đo HA 155 → TRUNG_BINH, Hệ tim mạch; 1 chỉ số BMI 31
   (mode met) → TRUNG_BINH, Chuyển hóa.
2. Chat: "Huyết áp 158/96, nhịp tim 82" → BÁO CÁO SƠ BỘ với R_CV_01, R_CV_03.
3. Mode dm: "đường huyết đói 7.8" → R_END_01 (Tăng đường huyết lúc đói).
4. API: `/api/assess` mode=htn → chỉ R_CV_03; mode=ckd + creatinine 2.1 → R_KID_01.
5. Server restart: `/`, `/chat`, `/rules`, `/benchmark`, `/api/benchmark` đều 200.
6. `/api/kb` version 0.3.0; `/rules` hiển thị "Chế độ:" trên rule card.

## 7. Giới hạn / hướng tiếp theo (nói thẳng)

1. **2019–2020 không thêm được riêng** vì CDC gộp chung với 2017–2018 — nếu thêm
   sẽ nhân đôi mẫu 2017–2018. Khi CDC phát hành chu kỳ độc lập hơn (2023+), bổ
   sung thêm.
2. **2023–2024 và 2025 chưa có dữ liệu công khai** — cần theo dõi trang NHANES
   What's New để cập nhật khi phát hành.
3. **Báo cáo sơ bộ 1 chỉ số chưa cá nhân hóa** — đúng bản chất: 1 lần đo không
   thể biết biến thiên cá nhân. Hệ thống ghi rõ điều này trong báo cáo và luôn
   nhắc tiếp tục đo để chuyển lên báo cáo đầy đủ.
4. **Chu kỳ L đổi phương pháp đo** (oscillometric) và codebook thuốc — gộp cùng
   chu kỳ I/J là đúng về dân số nhưng cần hiểu sự khác biệt về đo lường khi đọc
   kết quả (một phần lý do AUC thấp hơn chút nhưng ổn định hơn).
5. Mode mới cần ghi thêm vào tài liệu hướng dẫn sử dụng (chưa làm trong đợt này).

---

*Ghi chép nội bộ. Không thay thế chẩn đoán của bác sĩ.*
