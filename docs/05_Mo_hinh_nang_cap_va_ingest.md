# 05. Nâng cấp mô hình & Mô hình con trích xuất dữ liệu tài liệu

## 1. Trạng thái triển khai hiện tại

| Thành phần | Trạng thái | Ghi chú |
|------------|-----------|---------|
| Tầng 1 (Z-Score cá nhân, Isolation Forest, EWMA) | Đã triển khai | Đã kiểm thử trên dữ liệu mẫu |
| Tầng 2 (Rule engine tri thức y khoa) | Đã triển khai | 8 luật / 5 hệ cơ quan, cấu hình JSON |
| Tầng 3 (LightGBM) | Khung mô hình | Cần dữ liệu có nhãn để huấn luyện (giai đoạn 3) |
| LSTM (đối chứng) | Khung mô hình | Cần torch + dữ liệu |
| API FastAPI + Trích xuất PDF/DOCX | Đã triển khai | Đã kiểm thử end-to-end |

Hệ thống đang chạy trên hai tầng thống kê và tri thức. Mô hình học máy (LightGBM/LSTM) mới ở dạng khung; khi có dữ liệu thực sẽ huấn luyện, benchmark và lập báo cáo theo quy trình tiêu chuẩn.

---

## 2. Lộ trình nâng cấp mô hình

Các công bố đã khảo sát sử dụng GBM, LSTM, generative transformer (Delphi/Foresight), foundation model EHR (CLMBR-T) và LLM tổng quát. Lộ trình nâng cấp gồm 3 cấp độ, mỗi cấp tăng hiệu năng nhưng duy trì khả năng giải thích.

### Cấp 1 (hiện tại) — Hybrid giải thích được: thống kê + luật + GBM
- Z-Score cá nhân hóa, Isolation Forest, EWMA/STL, rule engine, LightGBM + SHAP.
- Phù hợp v1, nghiên cứu và báo cáo. Hạn chế: không mô hình hóa phụ thuộc dài hạn phức tạp.

### Cấp 2 — Time-series Foundation Model (khuyến nghị v2)
Ghép thêm foundation model chuỗi thời gian đã huấn luyện sẵn vào Tầng 1:
- **Chronos (Amazon)**, **TimesFM (Google)**, **Lag-Llama**: dự báo chuỗi chỉ số cơ thể; so sánh giá trị dự báo với thực tế để phát hiện bất thường.
- Điểm mới so với các công bố: phát hiện bất thường dựa trên **lỗi dự báo vượt ngưỡng cá nhân** thay vì giá trị thô — nhạy hơn với biến động bất thường, bỏ qua biến động sinh lý bình thường.
- Ưu điểm: đã pretrain (không cần dữ liệu lớn), đầu ra có phân phối nên tính được khoảng tin cậy.

### Cấp 3 — Token-level Healthcare Transformer (khi có dữ liệu lớn)
Kế thừa hướng Delphi-2M/Foresight nhưng khắc phục hai hạn chế được chỉ ra:
1. **Ưu tiên mức nguy thay vì tần suất:** bổ sung đầu ra "mức độ nghiêm trọng dự kiến" song song với xác suất sự kiện.
2. **Nhúng tri thức y khoa:** đưa luật lâm sàng (Tầng 2) vào dạng knowledge embedding trong cơ chế attention.
- Yêu cầu: từ 100.000 bệnh nhân, GPU. Định hướng nghiên cứu mở rộng, không thuộc phạm vi v1.

### Bảng quyết định

| Cấp | Mô hình | Dữ liệu cần | Minh bạch | Chi phí | Thời điểm |
|-----|---------|-------------|-----------|---------|-----------|
| 1 | Thống kê + luật + LightGBM | Nhỏ | Cao | Thấp | Hiện tại (v1) |
| 2 | + Chronos/TimesFM (bất thường theo lỗi dự báo) | Vừa (pretrain sẵn) | Trung bình–cao | Trung bình | Khi có chuỗi lịch sử thực |
| 3 | GPT-style + knowledge injection | Rất lớn | Trung bình | Cao | Nghiên cứu mở rộng |

---

## 3. Mô hình con trích xuất dữ liệu từ PDF/DOCX

Frontend nhận báo cáo sức khỏe định kỳ ở dạng file (PDF/DOCX); mô hình con tự động trích xuất thành dataset chuẩn trước khi đưa vào pipeline 3 tầng.

### Kiến trúc

```
PDF / DOCX / TXT
      |
      v
[1. extractor]   PyMuPDF (pdf) + python-docx (docx, đọc cả bảng)
      |
      v
[2. parser]      Regex: alias tiếng Việt/Anh -> chỉ số chuẩn + giá trị + đơn vị + ngày
      |          (patterns.json - cấu hình thuật ngữ)
      v
[3. LLM extractor] (tùy chọn) gọi Ollama/API để lấy JSON cấu trúc khi regex không đủ
      |              tự động fallback về regex nếu không có server
      v
[4. pipeline]    loại trùng -> DataFrame long-format (patient_id|timestamp|metric|value|unit)
      |
      v
   Lưu CSV  -->  pipeline 3 tầng  -->  API /api/assess_docs
```

### Chỉ số hỗ trợ (patterns.json)
`systolic_bp, diastolic_bp, heart_rate, glucose, glucose_fasting, hba1c, creatinine, egfr, spo2, bmi, weight, height` — alias song ngữ (ví dụ "Huyết áp tâm thu", "HATT", "Systolic BP").

### Hạn chế & hướng mở rộng
- Regex hạn chế với văn bản tự do phức tạp: bật `use_llm=True` khi có Ollama.
- Chưa xử lý PDF dạng ảnh quét: bổ sung OCR (Tesseract/PaddleOCR).
- Chưa chuẩn hóa đơn vị (mg/dL - mmol/L): bổ sung bảng quy đổi.

---

## 4. API

| Endpoint | Chức năng | Body |
|----------|-----------|------|
| `GET /api/health` | Kiểm tra dịch vụ | — |
| `GET /api/kb` | Cơ sở tri thức (cho frontend) | — |
| `POST /api/assess` | Đánh giá từ JSON | `{patient_id, records:[{timestamp, metric, value}]}` |
| `POST /api/assess_docs` | Upload PDF/DOCX, trích xuất và đánh giá | multipart: `file`, `patient_id`, `use_llm` |
| `POST /api/chat` | Nhập nhật ký hàng ngày (tích lũy theo bệnh nhân) | `{patient_id, message}` |
| `POST /api/chat_file` | Upload file qua hộp thoại chat | multipart: `file`, `patient_id` |
| `GET /chat` | Giao diện hộp thoại tích lũy dữ liệu | — |
| `GET /docs` | Swagger UI | — |

### Vận hành

```bash
./run_api.sh start      # khởi động
./run_api.sh stop       # dừng
./run_api.sh restart    # khởi động lại
./run_api.sh log        # xem log
```
