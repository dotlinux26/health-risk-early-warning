# 04. Lựa chọn & So sánh mô hình

**Câu hỏi then chốt:** *Với bài toán "cảnh báo sớm + giải thích được + dữ liệu nhỏ", thành phần mô hình học máy nào là tối ưu cho khung hỗ trợ quyết định của đề tài?*

> **Lưu ý định vị (quan trọng):** tài liệu này so sánh các **mô hình học máy**
> ở Tầng 3 — tức là so sánh **thành phần**, không so sánh toàn hệ thống. Đề tài
> xây dựng một **khung CDSF**, trong đó LightGBM là mô hình hiện tại nhưng **có
> thể thay thế**; việc so sánh dưới đây giúp chọn và báo cáo về thành phần đó,
> không tuyên bố "mô hình của đề tài chính xác hơn Delphi". Xem thêm
> [docs/07](07_Dinh_vi_de_tai.md).

---

## 1. Bảng so sánh: Mô hình trong các bài báo vs. Đề xuất cho đề tài

| Mô hình | Nguồn (bài báo) | Điểm mạnh | Điểm yếu cho đề tài này | Vai trò trong đề tài |
|---------|-----------------|-----------|--------------------------|----------------------|
| **GBM (Gradient Boosting)** | Bài 1 (baseline), mọi bài | Hiệu năng cao ngang foundation model; nhanh; tốt với dữ liệu nhỏ | Không mô hình hóa thời gian trực tiếp | **Chính** (LightGBM/XGBoost) cho điểm rủi ro Tầng 3 |
| **RNN / LSTM / GRU** | Bài 2 (review) | Chuyên cho chuỗi thời gian; phát hiện sớm | Hộp đen; cần dữ liệu lớn; dễ overfit trên mẫu nhỏ | **Đối chứng** benchmark, không dùng làm chính |
| **Transformer tạo sinh (Delphi-2M, Foresight)** | Bài 3, 4 | Mô hình hóa đa bệnh lý, tạo quỹ đạo tương lai | Rất nặng (triệu bệnh nhân, GPU); hộp đen; ưu tiên tần suất không phải mức nguy | Không dùng trong v1; ghi nhận là hướng mở rộng |
| **Foundation model EHR (CLMBR-T)** | Bài 1 | Chia sẻ mô hình, few-shot | Cần chuẩn OMOP, hạ tầng lớn, hộp đen | Không dùng |
| **LLM tổng quát (LLaMA 3.1 8B)** | Bài 4 (so sánh) | Hiểu ngữ cảnh | Kém hơn hẳn mô hình chuyên dụng y tế (AUROC thấp hơn) | Không dùng cho dự đoán; có thể dùng để sinh văn bản giải thích (tùy chọn) |
| **Thống kê truyền thống (Z-Score, EWMA, STL)** | Đề xuất mới | Minh bạch tuyệt đối, không cần dữ liệu lớn | Đơn giản, không tự học | **Chính** tại Tầng 1 |

---

## 2. Khuyến nghị kiến trúc mô hình (Recommended Stack)

```
TẦNG 1 (bất thường):        Z-Score cá nhân hóa + Isolation Forest + EWMA/STL   [thống kê + ML vô giám sát]
TẦNG 2 (tri thức):          Rule Engine trên JSON (luật lâm sàng)              [symbolic / interpretable]
TẦNG 3 (điểm rủi ro):       LightGBM (đặc trưng time-series) + SHAP            [ML giám sát + XAI]
Đối chứng (benchmark):      LSTM/GRU (PyTorch)  — so AUROC/AUPRC               [deep sequence]
```

### Vì sao LightGBM thắng cho vai trò chính?
- **Bằng chứng từ bài báo 1:** GBM đạt hiệu năng *tương đương* foundation model 141M tham số — với chi phí cực thấp.
- **Xử lý dữ liệu bảng (chỉ số cơ thể) tốt nhất:** đặc trưng hóa chuỗi (rolling mean, std, delta, z) vẫn giữ được "ngữ nghĩa thời gian".
- **Có SHAP** → giải thích từng đặc trưng đóng góp vào điểm rủi ro (đáp ứng yêu cầu XAI, "right to explanation").
- **Đã được kiểm chứng rộng rãi** trong vô số hệ thống cảnh báo lâm sàng.

### Khi nào nên cân nhắc LSTM/Transformer cho v1?
- Khi dữ liệu ≥ 10.000–100.000 bệnh nhân và cần mô hình hóa phụ thuộc dài hạn phức tạp.
- Khi bỏ được yêu cầu giải thích từng bước (hoặc chấp nhận SHAP trên đầu ra).
- **Kết luận:** để v1 ở mức *đối chứng*, đủ để báo cáo so sánh; không đưa vào hệ thống chính.

---

## 3. Bảng so sánh chi tiết các lựa chọn mô hình Tầng 3

| Tiêu chí | **LightGBM/XGBoost (khuyến nghị)** | LSTM/GRU | TCN (Temporal Conv) | Transformer thời gian |
|----------|-------------------------------------|----------|---------------------|-----------------------|
| Dữ liệu cần | Nhỏ–vừa | Lớn | Vừa | Rất lớn |
| Hiệu năng trên bảng + feature | Rất tốt | Tốt nếu đủ dữ liệu | Tốt | Tốt nếu rất nhiều dữ liệu |
| Khả năng giải thích | SHAP/importance | Kém | Kém | Kém |
| Chi phí huấn luyện | Thấp | Trung bình | Trung bình | Cao |
| Độ phức tạp code/triển khai | Thấp | Trung bình | Trung bình | Cao |
| **Kết luận** | **Chọn làm chính** | Đối chứng | Không cần | Không cần |

---

## 4. Ba cải tiến của khung so với các công bố (ở cấp kiến trúc)

> Đây là các cải tiến của **khung CDSF** (toàn hệ thống), không phải cải tiến
> của riêng mô hình ML — cần phân biệt rõ khi trình bày với hội đồng.

### 4.1. Cá nhân hóa bằng dữ liệu của chính người đó (thay vì ngưỡng cộng đồng)
- **Trong báo cáo:** Z-Score/ngưỡng thường tính theo *quần thể* (chuẩn hóa theo dân số) → sai với cá thể bình thường nhưng khác phân phối chung.
- **Cải tiến:** Mọi Z-Score, ngưỡng, độ dốc đều so với **chuỗi lịch sử của chính cá nhân** (cửa sổ trượt 30/90 ngày). Đây là khác biệt cốt lõi với Delphi/Foresight (đều học theo quần thể).

### 4.2. Rule + ML cộng hưởng (hybrid), không chọn 1 trong 2
- **Trong báo cáo:** Foresight/Delphi *chỉ* học từ dữ liệu → ưu tiên tần suất, bỏ sót tình huống hiếm nhưng nguy hiểm; bài báo 1 đã tự thừa nhận GBM ≈ foundation nhưng GBM không có tri thức y khoa.
- **Cải tiến:** Chạy **song song** luật lâm sàng (Tầng 2) và mô hình ML (Tầng 3); khi **đồng thuận** thì tăng độ tin cậy; khi **xung đột** thì báo "cần xem xét" — tận dụng cả symbolic + statistical learning.

### 4.3. Học trên dữ liệu tổng hợp để khắc phục thiếu dữ liệu
- **Trong báo cáo:** Delphi chứng minh dữ liệu AI sinh ra có thể đạt AUROC 0,74.
- **Cải tiến:** Xây simulator sinh chuỗi chỉ số cơ thể (theo mô hình sinh lý + nhiễu) để (a) phát triển pipeline ngay khi chưa có dữ liệu thực, (b) làm augmentation, (c) kiểm tra độ bền hệ thống.

---

## 5. Đề xuất metric đánh giá

| Mục tiêu | Metric |
|----------|--------|
| Bất thường Tầng 1 | Precision, Recall, F1 (nếu có nhãn); contamination tự ước lượng |
| Rủi ro Tầng 3 | **AUROC** (ưu tiên), **AUPRC** (quan trọng khi class hiếm), F1, Calibration (Brier score) |
| Cảnh báo sớm | Lead time (bao lâu trước khi chẩn đoán thực tế), bài học từ bài báo 2 (LSTM phát hiện sớm hơn lâm sàng) |
| Giải thích | Đánh giá bác sĩ (như Foresight: 97% dự báo phù hợp lâm sàng), kiểm tra reason_chain |
| Thống kê | CV (Stratified K-Fold), DeLong test khi so AUROC 2 mô hình |

---

## 6. Pipeline thí nghiệm đề xuất

```
1. Split: train / val / test (theo bệnh nhân, KHÔNG shuffle ngẫu nhiên — tránh data leakage)
2. Feature engineering: rolling mean/std, delta, z-score lịch sử, slope, EWMA
3. Huấn luyện LightGBM (tuning: Optuna/RayTune)
4. Huấn luyện LSTM/GRU (đối chứng)
5. So sánh AUROC/AUPRC + DeLong; kiểm tra calibration
6. SHAP global (đặc trưng quan trọng) + SHAP local (lý do từng bệnh nhân)
7. Kết hợp luật Tầng 2: mô phỏng "đồng thuận / xung đột"
8. Viết báo cáo benchmark -> docs/05_benchmark.md
```

> **Lưu ý về rò rỉ dữ liệu:** không fit scaler/hàm nhìn tương lai vào quá khứ; cửa sổ rolling chỉ dùng dữ liệu ở thời điểm trước hoặc bằng thời điểm dự báo.
