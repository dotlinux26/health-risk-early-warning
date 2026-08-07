**1\. Tên đề tài Tiếng Việt: Nghiên cứu diễn tiến tự nhiên của bệnh lý ở người bằng mô hình Transformer tạo sinh**

**2\. Phân tích Đầu vào / Đầu ra và Mục tiêu**

**\- Bài toán cần giải quyết:** Hầu hết các mô hình y tế hiện nay chỉ dự báo cho một bệnh lý đơn lẻ. Bài báo này xây dựng một mô hình duy nhất có khả năng theo dõi tiến trình đa bệnh lý (multi-morbidity) và dự báo nguy cơ cạnh tranh của **hơn 1.000 bệnh lý** thuộc hệ thống mã hóa ICD-10. 

**\- Dữ liệu đầu vào (Input):** Chuỗi thời gian gồm các mã chẩn đoán ICD-10 gắn liền với độ tuổi chẩn đoán, giới tính, các yếu tố lối sống (chỉ số BMI, mức độ hút thuốc, uống rượu) và các thẻ đệm "không có sự kiện" (padding). 

**\- Dữ liệu đầu ra (Output):** Tỷ lệ mắc (incidence rate) theo ngày cho từng loại bệnh trong tương lai và thời gian dự kiến diễn ra sự kiện tiếp theo. 

**3\. Bóc tách Kiến trúc Hệ thống & Thuật toán Cốt lõi**

    Tác giả phát triển mô hình **Delphi-2M** (khoảng 2.2 triệu tham số) dựa trên kiến trúc GPT-2 nhưng thực hiện 4 cải tiến quan trọng để phù hợp với dữ liệu chuỗi thời gian y tế: 

**3.1.Mã hóa độ tuổi liên tục:**Thay thế Positional Encoding.

Khác với văn bản thông thường có vị trí từ rời rạc, dữ liệu y tế diễn ra trên trục thời gian liên tục. Delphi thay thế bộ mã hóa vị trí của GPT bằng các hàm cơ sở sin/cosine dựa trên độ tuổi thực tế của bệnh nhân.

**3.2.Mô hình thời gian chờ lũy thừa:**Thêm đầu ra dự báo thời gian.

Bổ sung một đầu ra mới dựa trên lý thuyết "Exponential Waiting Time" để dự báo khoảng thời gian cho đến khi sự kiện tiếp theo xảy ra, bên cạnh việc dự báo loại bệnh.

**3.3.Mặt nạ chú ý theo thời gian liên tục:**Xử lý sự kiện đồng thời.

Điều chỉnh cơ chế Causal Attention Mask để che (mask) các sự kiện xảy ra cùng một thời điểm, đảm bảo mô hình chỉ nhìn về quá khứ và không bị "rò rỉ" dữ liệu tương lai.

**3.4.Chèn thẻ đệm 'Không có sự kiện':**Tần suất trung bình 1 lần / 5 năm.

Tự động chèn các thẻ đệm ngẫu nhiên vào các giai đoạn bệnh nhân không có bệnh (đặc biệt là khi còn trẻ) để tránh khoảng trống dữ liệu quá dài và giúp mô hình cập nhật nguy cơ theo độ tuổi.

**4\. Đánh giá Hiệu năng & Kết quả So sánh (Metrics)**

**\- Độ chính xác tổng thể:** Trong bộ dữ liệu kiểm chứng, chỉ số AUROC trung bình đạt **0.76** trên toàn bộ phổ bệnh ICD-10. Dự báo tử vong đạt AUROC cực kỳ cao là **0.97**. 

**\- Dự báo dài hạn:** Khi kéo dài khoảng thời gian dự báo đến 10 năm sau, AUROC chỉ giảm nhẹ từ 0.76 xuống 0.70, chứng minh giá trị trong việc tiên lượng dài hạn. 

**\- So sánh với các mô hình khác:**

* Vượt trội hoặc tương đương với các thang điểm nguy cơ lâm sàng truyền thống như QRISK3, Score2, Framingham (đối với bệnh tim mạch) hay Charlson/Elixhauser (đối với tử vong). 

  * Vượt trội hơn mô hình ngôn ngữ lớn tổng quát LLaMA-3.1 (8B) khi áp dụng vào bài toán đánh giá rủi ro y tế. 

**\- Khả năng tổng quát hóa xuyên quốc gia:** Khi đem mô hình đã huấn luyện từ UK Biobank (Anh) sang kiểm chứng trên **1.93 triệu người Đan Mạch** mà *không hề huấn luyện lại hay chỉnh sửa tham số*, mô hình vẫn đạt AUROC 0.67 và giữ độ tương quan rất cao (r \= 0.76). 

**5\. Khai phá Dữ liệu Tạo sinh, Tính Giải thích (XAI) & Rào cản**

**Các điểm sáng đột phá:**

1. **Mô phỏng quỹ đạo tương lai:** Cho phép "chạy thử" diễn biến sức khỏe của một cá nhân trong 10–20 năm tới. 

2. **Huấn luyện trên Dữ liệu Tổng hợp:** Lần đầu tiên chứng minh mô hình Delphi huấn luyện hoàn toàn trên dữ liệu giả lập (do AI sinh ra) vẫn đạt AUROC 0.74. Điều này mở ra giải pháp cho bài toán bảo mật dữ liệu y tế. 

3. **Khả năng giải thích qua SHAP & UMAP:**

   * Bảng ánh xạ không gian tự động gom cụm các bệnh có cùng bản chất lại gần nhau (ví dụ: tiểu đường đi kèm bệnh lý võng mạc và thần kinh). 

   * Phân tích SHAP cho thấy **mức độ ảnh hưởng theo thời gian**: Ung thư gây tăng nguy cơ tử vong kéo dài nhiều năm, trong khi nhiễm trùng huyết (septicaemia) làm tăng nguy cơ tử vong đột ngột nhưng giảm nhanh sau vài tháng. 

**Rào cản & Mất cân bằng dữ liệu:**

* **Immortal Time Bias:** UK Biobank chỉ tuyển chọn người từ 40–70 tuổi, dẫn đến thiếu hụt dữ liệu tử vong ở độ tuổi trẻ. 

* **Sai lệch từ nguồn dữ liệu:** Mức độ ghi nhận bệnh phụ thuộc vào nơi thu thập (bệnh viện vs. phòng khám ban đầu). Các bệnh chỉ chẩn đoán ở bệnh viện (như nhiễm trùng huyết) tạo ra các cụm rủi ro nhân tạo trong mô hình. 

