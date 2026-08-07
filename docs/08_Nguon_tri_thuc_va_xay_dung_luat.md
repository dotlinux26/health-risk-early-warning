# 08. Nguồn tri thức y khoa và quy trình xây dựng luật (Knowledge Base)

> Mục đích: trả lời câu hỏi hội đồng thường hỏi *"Luật của hệ thống lấy từ đâu?
> Có chính xác không? Ai kiểm chứng?"* — làm rõ rằng các luật trong Tầng 2 là
> **tri thức y khoa đã công bố**, được trích dẫn và quản lý phiên bản, không phải
> do hệ thống hay mô hình AI tự sinh ra.

---

## 1. Luật không tự sinh ra — luật được trích từ hướng dẫn lâm sàng

Luật trong `src/tier2_knowledge/knowledge_base.json` **không phải là đầu ra của
mô hình AI**. Chúng là các **ngưỡng và quy tắc lâm sàng** được đối chiếu trực tiếp
từ các **hướng dẫn chính thống** (clinical guidelines) do các tổ chức y khoa quốc
tế công bố. Vai trò của nhóm là: đọc, chọn lọc, chuẩn hóa thành cấu trúc máy đọc
được, và ghi rõ nguồn trích dẫn cho từng luật.

| Hướng dẫn | Tổ chức | Dùng cho |
|---|---|---|
| ESC/ESH 2018 [1] | ESC & ESH (châu Âu) | Tăng huyết áp — ngưỡng 140/90, tăng huyết áp tâm thu đơn độc |
| ADA 2023 [2] | American Diabetes Association | Đái tháo đường — đường huyết đói 7.0, HbA1c 6.5% |
| KDIGO 2022 [3] | KDIGO (quốc tế) | Bệnh thận — creatinine, eGFR < 60 |
| WHO [4][6] | Tổ chức Y tế Thế giới | BMI thừa cân ≥ 25, bão hòa oxy |
| Hướng dẫn nhịp tim 2023 [5] | Hội Tim mạch (ACC/AHA) | Nhịp tim nhanh/chậm kéo dài |

### 1.1. Trích dẫn từng luật hiện tại

| Luật | Ý nghĩa | Ngưỡng | Nguồn |
|---|---|---|---|
| R_CV_01 | Tăng huyết áp | HA ≥ 140/90 mmHg | ESC/ESH 2018 [1] |
| R_CV_02 | Nhịp tim nhanh kéo dài | nhịp tim > 100 / 7 ngày | Hướng dẫn nhịp tim ACC/AHA [5] |
| R_CV_03 | Tăng huyết áp tâm thu đơn độc | HATT > 140 mmHg | ESC/ESH 2018 [1] |
| R_END_01 | Tăng đường huyết lúc đói | glucose đói > 7.0 mmol/L | ADA 2023 [2] |
| R_END_02 | HbA1c vượt ngưỡng tiểu đường | HbA1c > 6.5% | ADA 2023 [2] |
| R_KID_01 | Creatinine tăng — nghi suy thận | creatinine > 1.3 mg/dL | KDIGO 2022 [3] |
| R_KID_02 | eGFR giảm | eGFR < 60 | KDIGO 2022 [3] |
| R_RES_01 | Giảm bão hòa oxy | SpO2 < 94% | WHO [6] |
| R_MET_01 | BMI vượt ngưỡng thừa cân | BMI > 25 | WHO [4] |

Mỗi luật trong file JSON đều có trường `evidence` ghi nguồn trích dẫn và
`specialty` ghi chuyên khoa phụ trách — để bất kỳ ai cũng tra ngược được gốc
tri thức.

---

## 2. Quy trình xây dựng luật (5 bước)

```
1. Thu thập    →  đọc các hướng dẫn lâm sàng chính thống (ESC/ESH, ADA, KDIGO, WHO)
2. Chọn lọc    →  chỉ giữ luật liên quan các chỉ số hệ thống hỗ trợ (có thể đo tại nhà)
3. Chuẩn hóa   →  chuyển ngưỡng + điều kiện thành cấu trúc JSON máy đọc được
4. Kiểm chứng  →  rà soát chéo với y văn; đối chiếu nhiều nguồn nếu có thể
5. Phiên bản   →  ghi version, nguồn, ngày; luật thay đổi theo phiên bản hướng dẫn mới
```

### 2.1. Nguyên tắc chọn luật

- Chỉ nhận **ngưỡng được công bố** trong hướng dẫn chính thống, không tự đặt ngưỡng.
- Ưu tiên chỉ số **đo được tại nhà** (huyết áp, đường huyết, SpO2, cân nặng, nhịp tim).
- Mỗi luật gắn với một **hệ cơ quan** và một **chuyên khoa khuyến nghị** để cảnh báo
  có hướng xử lý rõ ràng.
- Nếu nhiều hướng dẫn cho ngưỡng khác nhau → chọn ngưỡng phổ biến nhất và ghi chú
  thêm nguồn thay thế trong `evidence`.

---

## 2.2. Các nguồn dữ liệu đã tải xuống

Ngoài tri thức y khoa dạng luật, đề tài còn sử dụng các **bộ dữ liệu thật tải
xuống** để huấn luyện và đối chứng mô hình (thành phần ML Tầng 3). Tất cả được
lưu trong `data/datasets/`:

| Bộ dữ liệu | Nguồn | Nội dung | Vai trò trong đề tài |
|---|---|---|---|
| NHANES 2017-2018 | CDC/NCHS (Mỹ) — tải file XPT gốc | 4949 người trưởng thành; HA, đường huyết, HbA1c, creatinine, BMI | **Train model sản xuất** (`risk_lgbm_real.joblib`) |
| Pima Indians Diabetes | UCI Machine Learning Repository | 768 mẫu, 268 tiểu đường | **Đối chứng** pipeline ML trên dữ liệu thật |
| Cleveland Heart Disease | UCI Machine Learning Repository | 303 mẫu, 139 bệnh tim | **Đối chứng** pipeline ML trên dữ liệu thật |

Chi tiết quy trình tải và kết quả huấn luyện: [docs/06](06_Bao_cao_huong_train_va_gioi_han.md).

> **Phân biệt 2 loại "nguồn":** luật Tầng 2 lấy từ **hướng dẫn lâm sàng**
> (ESC/ESH, ADA, KDIGO, WHO); còn các bộ dữ liệu trên dùng để **huấn luyện/đối
> chứng mô hình ML** — hai nguồn phục vụ hai mục đích khác nhau, cần nói rõ khi
> trình bày để tránh nhầm lẫn.

---

## 3. Độ tin cậy của luật được bảo đảm thế nào

1. **Trích dẫn nguồn gốc:** mọi luật đều có trường `evidence` trỏ tới hướng dẫn
   lâm sàng cụ thể — đối chiếu được, kiểm toán được.
2. **Nguồn chính thống:** chỉ dùng hướng dẫn của các tổ chức y khoa uy tín quốc tế,
   cập nhật theo phiên bản mới nhất (ADA 2023, KDIGO 2022, ESC/ESH 2018).
3. **Chuyên khoa kiểm chứng:** mỗi luật gắn chuyên khoa phụ trách; trong triển
   khai thực tế cần bác sĩ chuyên khoa rà soát lại trước khi đưa vào sản xuất.
4. **Kiểm thử tự động:** bộ luật được kiểm thử bằng các ca mẫu (ví dụ bệnh nhân
   có HA 202/62 kích hoạt đúng R_CV_03, không kích hoạt nhầm luật khác) — xem
   mục 4.

### 3.1. Giới hạn cần nêu trước hội đồng

- **Ngưỡng phổ quát ≠ ngưỡng cá nhân:** ngưỡng lâm sàng áp cho quần thể; cá thể có
  bệnh nền, tuổi tác, thuốc đang dùng có thể cần ngưỡng riêng. Khung xử lý điều này
  bằng Tầng 1 (so với chính cá nhân) kết hợp Tầng 2 (so với ngưỡng y khoa).
- **Chưa có bác sĩ lâm sàng trong quy trình phê duyệt:** quy trình hiện tại dựa
  trên tài liệu chuẩn; bước tiếp theo là đưa luật qua hội đồng chuyên môn hoặc bác
  sĩ nội/ngoại thẩm định trước khi coi là "đã kiểm chứng lâm sàng".
- **Hướng dẫn thay đổi theo thời gian:** ngưỡng có thể được cập nhật khi có hướng
  dẫn mới (ví dụ ACC/AHA 2017 hạ ngưỡng xuống 130/80) — cần cơ chế theo dõi phiên
  bản, không coi tri thức là tĩnh.

---

## 4. Kiểm thử luật với dữ liệu thật

Quy trình kiểm chứng hiện tại dùng **dữ liệu thật NHANES (CDC)**:

- Các ca mẫu trong `data/sample_nhanes/` được chạy qua pipeline để kiểm tra luật
  kích hoạt **đúng** và **không kích hoạt nhầm**:

| Ca mẫu | Kỳ vọng | Luật kích hoạt | Kết quả |
|---|---|---|---|
| NOK0001 (khỏe mạnh) | KHÔNG có luật | — | Đạt (THAP) |
| NHTN0001 (HA 202/62) | R_CV_03 | Tăng HA tâm thu đơn độc | Đạt (TRUNG_BINH) |
| NDM0001 (HbA1c ≥ 7%) | R_END_02 | HbA1c vượt ngưỡng tiểu đường | Đạt (TRUNG_BINH) |
| NCKD0001 (creatinine ≥ 1.5) | R_KID_01 | Creatinine tăng — nghi suy thận | Đạt (TRUNG_BINH) |

- Chính quá trình kiểm thử này đã phát hiện lỗi thật: ca NHTN0001 (202/62) trước
  đây chỉ kích hoạt tăng huyết áp cả hai chiều, bỏ sót **tăng huyết áp tâm thu
  đơn độc** → nhóm bổ sung luật R_CV_03 theo ESC/ESH 2018. Đây là minh chứng quy
  trình luật được kiểm thử bằng dữ liệu thực, không chỉ viết trên giấy.

---

## 5. Vị trí của tri thức trong kiến trúc tổng thể

```
Tầng 1  Thống kê cá nhân hóa  →  "chỉ số lệch so với CHÍNH bệnh nhân"
                                   │
Tầng 2  Luật tri thức y khoa   →  "chỉ số vượt ngưỡng y học PHỔ QUÁT"  ◄── nguồn: guidelines
                                   │
Tầng 3  LightGBM + tổng hợp    →  "học từ dữ liệu có nhãn"
                                   │
                                   ▼
                  Cảnh báo kèm GIẢI THÍCH (từng tầng, tra ngược được)
```

Tri thức y khoa là **lớp tham chiếu chuẩn**: nó bắt buộc mọi kết luận của hệ thống
(do thống kê hay mô hình ML đưa ra) phải được soi vào ngưỡng lâm sàng đã công bố.
Đây là lý do hệ thống luôn ra quyết định **theo luật**, kể cả khi không có mô hình
AI — khớp với định vị trong [docs/07](07_Dinh_vi_de_tai.md).

---

## 6. Lộ trình nâng cao chất lượng luật

| Giai đoạn | Việc cần làm |
|---|---|
| G1 ✅ | Tập luật khởi đầu (9 luật / 5 hệ cơ quan) từ guidelines, kèm trích dẫn — đã xong |
| G2 | Phê duyệt luật bởi bác sĩ chuyên khoa (đối chiếu lâm sàng Việt Nam) |
| G3 | Theo dõi phiên bản hướng dẫn; cập nhật ngưỡng khi có guideline mới (ACC/AHA 2017...) |
| G4 | Bổ sung luật theo cơ chế "bệnh nhân cá thể hóa" (điều chỉnh ngưỡng theo tuổi/bệnh nền/thuốc) |

---

## 7. Trách nhiệm pháp lý và đạo đức

- Luật chỉ dùng cho **hỗ trợ quyết định**, không phải chẩn đoán; mọi đầu ra đều
  kèm disclaimer.
- Ghi rõ nguồn trích dẫn từng luật để **kiểm toán được** — nếu ngưỡng sai, tìm được
  ngay gốc rễ và phiên bản.
- Không đưa luật tự suy diễn ngoài hướng dẫn vào hệ thống sản xuất.

---

## 8. Tài liệu tham khảo (căn cứ xây dựng luật)

### 8.1. Hướng dẫn lâm sàng (nguồn tri thức luật Tầng 2)

1. Williams, B., Mancia, G., Spiering, W., et al. (2018). 2018 ESC/ESH Guidelines
   for the management of arterial hypertension. *European Heart Journal*, 39(33),
   3021–3104. https://doi.org/10.1093/eurheartj/ehy339
2. American Diabetes Association Professional Practice Committee. (2023).
   Standards of Care in Diabetes — 2023. *Diabetes Care*, 46(Suppl 1), S1–S291.
   https://doi.org/10.2337/dc23-Sint
3. KDIGO 2022 Clinical Practice Guideline for Diabetes Management in Chronic
   Kidney Disease. *Kidney International*, 102(5S), S1–S127 (2022).
   https://doi.org/10.1016/j.kint.2022.06.008
4. World Health Organization. (2000). *Obesity: preventing and managing the global
   epidemic* (WHO Technical Report Series 894) — ngưỡng BMI ≥ 25 (thừa cân),
   ≥ 30 (béo phì). https://www.who.int/publications/i/item/WHO-TRS-894
5. Writing Committee Members et al. (2023). 2023 ACC/AHA/ACCP/HRS Guideline for
   the Diagnosis and Management of Atrial Fibrillation — và các tài liệu ACC/AHA
   về nhịp tim trong lâm sàng (ngưỡng nhịp nhanh > 100 lần/phút kéo dài).
6. World Health Organization. (2019). *Guideline on use of pulse oximetry for
   monitoring of patients* — ngưỡng bão hòa oxy SpO2 < 94% cần đánh giá lâm sàng.
   https://www.who.int/publications/i/item/9789241550482

### 8.2. Bộ dữ liệu (nguồn huấn luyện/đối chứng mô hình ML)

7. Centers for Disease Control and Prevention, National Center for Health
   Statistics. (2020). *National Health and Nutrition Examination Survey
   (NHANES) 2017–2018*. Các bảng dữ liệu: DEMO_J, BPX_J, BPQ_J, BMX_J, GHB_J,
   GLU_J, BIOPRO_J. https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2017
8. UCI Machine Learning Repository. *Pima Indians Diabetes Database*.
   https://archive.ics.uci.edu/dataset/34/diabetes
9. UCI Machine Learning Repository. *Heart Disease (Cleveland) Data Set*.
   https://archive.ics.uci.edu/dataset/45/heart+disease

### 8.3. Tiêu chuẩn tham chiếu bổ sung

- JNC-8 (James et al., 2014, *JAMA* 311(5):507–520) — ngưỡng huyết áp theo nhóm tuổi.
- ACC/AHA 2017 (Whelton et al., 2018, *Hypertension* 71(6):e13–e115) — ngưỡng 130/80
  (phiên bản chặt hơn, đang cân nhắc cập nhật cho luật R_CV_01/R_CV_03).

> Cách dùng trong báo cáo: trích theo số [n] gắn vào từng luật ở cột `evidence`
> (ví dụ "R_CV_01 — theo [1]"), đồng thời liệt kê đầy đủ ở mục Tài liệu tham khảo
> để hội đồng tra cứu được từng ngưỡng.

---

*Tài liệu nội bộ. Không thay thế chẩn đoán của bác sĩ.*
