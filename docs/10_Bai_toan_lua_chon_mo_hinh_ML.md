# 10. Bài toán lựa chọn mô hình học máy trong khung hỗ trợ quyết định

> Tài liệu mô tả bài toán ở mức khái niệm, không gắn với cá nhân hay lựa chọn
> cụ thể. Mục đích: làm rõ vì sao "chọn mô hình nào" là một bài toán thiết kế
> đúng đắn, và cách tiếp cận hợp lý khi tích hợp nhiều mô hình.

---

## 1. Bối cảnh: mô hình chỉ là một thành phần trong khung

Trong một khung hỗ trợ quyết định lâm sàng nhiều tầng, mô hình học máy đóng vai
trò **hỗ trợ** — góp một phần bằng chứng vào quyết định cuối, không tự đưa ra
chẩn đoán. Quyết định cuối là kết hợp nhiều nguồn:

| Nguồn | Vai trò |
|---|---|
| Thống kê cá nhân hóa | Phát hiện bất thường so với chính bệnh nhân |
| Tri thức y khoa (luật) | Gán ý nghĩa lâm sàng cho bất thường |
| Mô hình học máy | Ước lượng nguy cơ học được từ dữ liệu nhiều bệnh nhân |
| Xu hướng | Đánh giá diễn biến theo thời gian |

Vì mô hình chỉ đóng góp một phần, việc chọn mô hình không làm thay đổi toàn bộ
hành vi hệ thống — đây là **điểm mạnh về kiến trúc**, không phải hạn chế.

---

## 2. Bài toán: chọn mô hình nào để hỗ trợ?

Bài toán đặt ra là: **với dữ liệu dạng bảng / chuỗi thời gian của chỉ số sức
khỏe, dùng mô hình học máy nào để ước lượng nguy cơ một cách tin cậy và giải
thích được?**

Các yêu cầu của bài toán:

1. **Đầu vào phù hợp** — dữ liệu là đặc trưng số (giá trị trung bình, độ dốc,
   độ lệch chuẩn, xu hướng theo cửa sổ thời gian), không phải văn bản.
2. **Dữ liệu huấn luyện hạn chế** — tập dữ liệu thật có quy mô hàng nghìn mẫu,
   không đủ lớn cho các mô hình tham số lớn.
3. **Đầu ra phải giải thích được** — cần biết đặc trưng nào đóng góp vào điểm
   nguy cơ, để báo cáo minh bạch cho người dùng.
4. **Thay thế được** — kiến trúc cho phép hoán đổi mô hình mà không đụng tới
   các thành phần khác (luật, báo cáo, giao diện).

---

## 3. Hai nhóm mô hình khả dĩ

### Nhóm A — Mô hình học có giám sát cổ điển / dựa trên cây

- **Đặc điểm:** mạnh trên dữ liệu dạng bảng quy mô vừa và nhỏ; huấn luyện nhanh;
  giải thích được bằng mức độ quan trọng của đặc trưng.
- **Đại diện:** Gradient Boosting (LightGBM), Random Forest, XGBoost.
- **Phù hợp:** làm mô hình chính khi cần độ tin cậy cao, chi phí tính toán thấp.

### Nhóm B — Mô hình dựa trên mạng nơ-ron cho dữ liệu bảng

- **Đặc điểm:** dùng cơ chế attention để học quan hệ giữa các đặc trưng; linh
  hoạt hơn khi có dữ liệu lớn; nhưng **nhạy cảm với quy mô dữ liệu nhỏ**.
- **Đại diện:** TabNet, FT-Transformer, MLP nhiều lớp.
- **Phù hợp:** thử nghiệm, so sánh; hoặc khi muốn mở rộng khả năng học đặc trưng
  phức tạp hơn.

---

## 4. Cách tiếp cận hợp lý: dùng song song và so sánh

Không nhất thiết phải "chọn một bỏ một". Cách tiếp cận được khuyến nghị:

1. **Xây dựng chung một giao diện mô hình** — mọi mô hình đều nhận cùng ma trận
   đặc trưng và trả về một điểm nguy cơ trong khoảng [0, 1].
2. **Đánh giá công bằng trên cùng bộ dữ liệu** — chạy từng mô hình trên tập
   kiểm định riêng, so sánh chỉ số AUC (diện tích dưới đường cong ROC).
3. **Chọn mô hình có chỉ số tốt hơn làm mặc định**, giữ mô hình còn lại để đối
   chứng và đối chiếu chéo kết quả.
4. **Sử dụng cả hai để tăng độ tin cậy** — khi hai mô hình cùng nghiêng về một
   kết luận, mức độ tin cậy của cảnh báo tăng lên; nếu chúng khác nhau, hệ thống
   hiển thị đây là vùng chưa chắc chắn, cần bác sĩ xem xét.

> Nguyên tắc cốt lõi: **đa mô hình, đối chiếu chéo** — không phụ thuộc vào bất kỳ
> mô hình nào, đồng thời cung cấp cái nhìn đầy đủ hơn cho người ra quyết định.

---

## 5. Giới hạn cần nêu rõ

- **Không mô hình nào chẩn đoán được bệnh** — đầu ra chỉ là ước lượng nguy cơ.
- **Dữ liệu huấn luyện nhỏ** có thể làm mô hình nơ-ron kém hơn mô hình cây —
  cần kiểm chứng thực nghiệm thay vì giả định.
- **Chỉ số AUC tốt chưa đủ** — cần kiểm tra trên từng nhóm bệnh nhân và từng
  loại chỉ số để tránh nhiễu giả.
- Quyết định cuối luôn thuộc về **bác sĩ**, không thuộc về mô hình.

---

## 6. Tóm tắt

> Việc lựa chọn mô hình học máy trong khung hỗ trợ quyết định là một bài toán
> **thiết kế có điều kiện ràng buộc rõ**: dữ liệu dạng bảng, quy mô hạn chế,
> yêu cầu giải thích được và khả năng thay thế. Cách tiếp cận hợp lý là **tích
> hợp nhiều mô hình cùng một giao diện, đánh giá công bằng trên cùng dữ liệu,
> dùng đối chiếu chéo để tăng độ tin cậy** — thay vì tuyệt đối tin vào một mô
> hình duy nhất.

---

*Tài liệu nội bộ. Không thay thế chẩn đoán của bác sĩ.*
