# 01. Tổng hợp nghiên cứu khoa học

**Đề tài:** *Nghiên cứu và xây dựng hệ thống cảnh báo sớm nguy cơ sức khỏe dựa trên phân tích chuỗi thời gian cá nhân hóa và ánh xạ tri thức y khoa.*

---

## 1. Bức tranh nghiên cứu chung

Báo cáo tổng hợp **5 công bố khoa học** bao phủ toàn bộ chuỗi giá trị của AI trong dự đoán bệnh dựa trên dữ liệu sức khỏe cá nhân:

| # | Công bố | Trọng tâm | Dữ liệu | Mô hình | Kết quả chính |
|---|---------|-----------|---------|---------|---------------|
| 1 | Guo et al., 2024 (npj Digital Medicine) | Foundation model cho EHR | 2,57M bệnh nhân Stanford; ngoại kiểm tại SickKids, MIMIC-IV | CLMBR-T-base (141M), GBM baseline | Foundation model ngoại vi ≈ GBM; cải thiện 13% few-shot; chỉ cần <1% dữ liệu khi tiếp tục pretrain |
| 2 | Swinckels et al., 2024 (JMIR) | Scoping review ML/DL trên EHR dọc | 20 nghiên cứu (2018–2022) | RNN, LSTM (phổ biến nhất) | Dự đoán tốt nhất: tiểu đường, thận, tim mạch, tâm thần; 90% mô hình thiếu ngoại kiểm |
| 3 | Kraljevic et al., 2024 (Lancet Digital Health) | Generative transformer dòng thời gian bệnh nhân | 811.336 bệnh nhân, 3 bệnh viện (cấu trúc + văn bản) | Foresight (GPT-2), MedCAT | Precision@10 = 0,68–0,91; 97% dự báo đỉnh được 5 bác sĩ đánh giá phù hợp lâm sàng |
| 4 | Nghiên cứu Delphi (2025) | Diễn tiến tự nhiên đa bệnh lý | UK Biobank; ngoại kiểm 1,93M người Đan Mạch | Delphi-2M (GPT-2, ~2,2M tham số) | AUROC 0,76 trung bình; tử vong 0,97; giảm nhẹ còn 0,70 khi dự báo 10 năm; xuyên quốc gia 0,67 (r=0,76) |
| 5 | Nghiên cứu EHR 25 năm (2025) | Tiến hóa, khả năng tương tác EHR | 2.212 nghiên cứu, NVivo | Phân tích định tính | 3 thập kỷ: POMR → Big Data/RWE → AI/NLP/Genomics; rào cản: liên chuẩn, privacy |

---

## 2. Khoảng trống nghiên cứu (Research Gaps) nhận diện được

Từ tổng hợp 5 bài báo, nổi lên **4 khoảng trống** mà đề tài đề xuất có thể chiếm lĩnh:

1. **Tính cá nhân hóa còn yếu:** Hầu hết mô hình so sánh bệnh nhân với *quần thể chung* (ngưỡng thống kê cộng đồng), bỏ qua "đường cơ sở sinh lý" riêng của từng cá thể → gây sai số với người có chỉ số nằm ngoài phân phối quần thể nhưng ổn định với chính họ.
2. **Hộp đen (black-box):** LSTM/RNN và các generative transformer (Delphi, Foresight) cho hiệu năng cao nhưng khó truy xuất *lý do* ra cảnh báo → khó được bác sĩ tin dùng và khó đáp ứng quy định (GDPR, "right to explanation").
3. **Thiếu ngoại kiểm & minh bạch lâm sàng:** Chỉ ~10% nghiên cứu có external validation; nhiều mô hình dừng ở mức prototype, không gắn với quy trình "hỗ trợ quyết định" an toàn.
4. **Phụ thuộc dữ liệu khổng lồ:** Foundation model (CLMBR-T) và transformer yêu cầu triệu bệnh nhân + hạ tầng lớn — không phù hợp môi trường nghiên cứu/đơn vị nhỏ có dữ liệu khiêm tốn.

---

## 3. Vị thế của đề tài trong bản đồ nghiên cứu

> **Lưu ý định vị:** đề tài không tạo ra một mô hình AI mới, mà đề xuất một
> **khung hỗ trợ quyết định lâm sàng (CDSF)** — trong đó mô hình học máy
> (LightGBM) chỉ là một thành phần có thể thay thế. Các so sánh trong mục này
> là so sánh **ở cấp kiến trúc khung**, không phải cấp mô hình. Xem thêm
> [docs/07](07_Dinh_vi_de_tai.md).

Đề tài đề xuất là **khung thiết kế lai (hybrid framework)** nằm giữa 2 cực:

```
[Polar 1: Đơn giản, minh bạch]          [Polar 2: Hiệu năng, hộp đen]
   Z-Score, luật lâm sàng        <------>   LSTM, Transformer (Delphi/Foresight)
          ▲                                         ▲
          │              ĐỀ TÀI NÀY                │
          └───────────────────┬─────────────────────┘
          Tầng 1 (thống kê + ML bất thường)
          Tầng 2 (tri thức y khoa dạng luật)
          Tầng 3 (điểm rủi ro + giải thích)
```

- **Kế thừa** bài báo 2 (RNN/LSTM chứng minh giá trị dữ liệu dọc) và bài báo 1 (GBM đạt hiệu năng tương đương foundation model với chi phí thấp).
- **Khác biệt hóa** so với bài báo 3, 4 (transformer tạo sinh): thay vì "đoán sự kiện tiếp theo", hệ thống **cảnh báo sớm + giải thích được** dựa trên sai lệch cá nhân hóa và luật y khoa.
- **Trả lời** cho bài báo 5 (rào cản EHR) bằng thiết kế chạy được trên dữ liệu tối giản (chỉ số cơ thể dạng bảng theo thời gian), không đòi hỏi toàn bộ hồ sơ bệnh án phi cấu trúc.

---

## 4. Bảng đối chiếu: Tiêu chí đề tài vs. 5 bài báo

| Tiêu chí | Bài 1 (Foundation) | Bài 2 (ML/DL review) | Bài 3 (Foresight) | Bài 4 (Delphi) | Bài 5 (EHR 25 năm) | **Đề tài này** |
|----------|--------------------|-----------------------|-------------------|----------------|---------------------|-----------------|
| Cá nhân hóa | Vừa (theo quần thể) | Vừa | Vừa | Vừa | Không áp dụng | **Cao (đường cơ sở từng cá nhân)** |
| Khả năng giải thích | Thấp | Trung bình | Thấp | Vừa (SHAP hậu kiểm) | N/A | **Cao (XAI theo thiết kế, multi-tier)** |
| Chi phí dữ liệu | Rất cao | Vừa | Cao | Cao | N/A | **Thấp (chỉ số cơ thể, mẫu nhỏ)** |
| Ngoại kiểm lâm sàng | Có (3 trung tâm) | Chỉ 10% | Có (3 bệnh viện) | Có (UK→Đan Mạch) | N/A | **Thiết kế dự kiến** |
| Độ an toàn lâm sàng | N/A | N/A | Không (hỗ trợ nghiên cứu) | Không | N/A | **Cao (chỉ hỗ trợ quyết định)** |

---

## 5. Các phát hiện then chốt (Key Takeaways) để vận dụng vào đề tài

1. **GBM là baseline mạnh:** Bài báo 1 chỉ ra Gradient Boosting đạt ngang foundation model → nên dùng **LightGBM/XGBoost** làm mô hình điểm rủi ro tại Tầng 3 (nhanh, mạnh, đi kèm SHAP).
2. **LSTM hợp lý cho chuỗi dọc:** Bài báo 2 xác nhận LSTM là lựa chọn hiệu quả cho dữ liệu theo thời gian → dùng như **mô hình so sánh/đối chứng** tại Tầng 1–3, không dùng làm mô hình chính (vì kém minh bạch).
3. **Dữ liệu tổng hợp là hướng đi:** Delphi chứng minh huấn luyện trên dữ liệu AI sinh ra đạt AUROC 0,74 → nên thiết kế **simulator sinh chuỗi chỉ số cơ thể tổng hợp** để phát triển & thử nghiệm khi thiếu dữ liệu thực.
4. **Bảo toàn tính lâm sàng:** Foresight bị chê vì "ưu tiên xác suất xuất hiện thay vì tính cấp bách" → hệ thống của ta phải *nhúng trực tiếp tri thức y khoa (ngưỡng lâm sàng, hệ cơ quan)* chứ không chỉ học từ dữ liệu.
5. **Rào cản dữ liệu:** Bài báo 5 cảnh báo về chuẩn hóa & quyền riêng tư → dữ liệu đầu vào nên thiết kế theo chuẩn tối giản (CSV/OMOP-lite), đảm bảo GDPR/HIPAA ngay từ thiết kế.

---

## 6. Trích dẫn chuẩn (tài liệu tham khảo cho báo cáo/đề cương)

1. Guo, L. L., et al. (2024). A multi-center study on the adaptability of a shared foundation model for electronic health records. *npj Digital Medicine*, 7(171).
2. Swinckels, L., et al. (2024). The Use of Deep Learning and Machine Learning on Longitudinal Electronic Health Records... *JMIR*, 26, e48320.
3. Kraljevic, Z., et al. (2024). Foresight — a generative pretrained transformer for modelling of patient timelines using EHRs. *The Lancet Digital Health*, 6(4), e281–e290.
4. Delphi-2M (2025). Learning the natural history of human disease with generative transformers. (UK Biobank).
5. Twenty-five years of evolution and hurdles in electronic health records (2025). (Scoping review, NVivo).
