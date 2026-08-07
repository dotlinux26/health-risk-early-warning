1. Tên đề tài để xuất: 25 năm tiến hóa và rào cản của Hồ sơ sức khỏe điện tử và Khả năng tương tác trong nghiên cứu y khoa và tổng quan.  
2. Phân tích mục tiêu:  
- Mục tiêu: Tổng hợp sự tiến hóa và khả năng tương tác của EHR trong nghiên cứu y khoa suốt 25 năm qua.   
- Đầu vào: Bài báo sàng lọc 2212 nghiên cứu từ các cơ sở dữ liệu lớn, và phân tích định tính 2102 bài báo hợp lệ bằng phần mềm NVivo.   
- Đầu ra: Sự phân kỳ rõ rệt của EHR qua 3 thập kỷ, cùng với việc xác định các rào cản về kiến trúc dữ liệu và bảo mật hiện tại.   
3. Kiến trúc hệ thống   
   Tầng 1: Khởi nguyên & Hồ sơ định hướng vấn đề:  
- Các nền tảng đầu tiên như Hệ thống Bệnh án Định hướng Vấn đề (POMR) và Regenstrief (RMRS) bắt đầu số hóa dữ liệu. Các nghiên cứu giai đoạn này chủ yếu lấy dữ liệu từ một bệnh viện đơn lẻ với thiết kế quan sát cơ bản. Việc Mỹ thông qua Đạo luật HITECH năm 2009 đã tạo cú hích lớn cho việc số hóa y tế diện rộng.

  Tầng 2: Kỷ nguyên Big Data & Thử nghiệm thực tiễn

- EHR trở thành kho dữ liệu khổng lồ phục vụ cho các nghiên cứu thuần tập lớn và Thử nghiệm lâm sàng thực tiễn (PCTs). Đây là bước ngoặt khi các cơ quan như FDA và EMA bắt đầu chấp nhận Bằng chứng Thực tế (RWE) trích xuất từ EHR để phê duyệt các loại thuốc mới thay vì chỉ dựa vào thử nghiệm truyền thống.

  Tầng 3: AI, NLP & Y học chính xác

- EHR đóng vai trò cốt lõi trong việc giám sát đại dịch COVID-19 theo thời gian thực, tiêu biểu là nền tảng OpenSAFELY tại Anh. Công nghệ Xử lý Ngôn ngữ Tự nhiên (NLP) (như mô hình GatorTron) được áp dụng mạnh mẽ để đọc hiểu các ghi chú lâm sàng phi cấu trúc. Các hệ thống cũng bắt đầu tích hợp dữ liệu gen (Genomics) từ các ngân hàng sinh học (như UK Biobank) để cá nhân hóa điều trị.  
4. Rào cản kỹ thuật:  
   Sự phức tạp trong làm sạch dữ liệu (Data Cleaning): EHR chứa lượng lớn văn bản tự do phi cấu trúc. Để làm sạch và trích xuất ý nghĩa, các nhà nghiên cứu thường phải dùng môi trường SQL hoặc Python, điều này đòi hỏi kỹ năng lập trình chuyên sâu.    
   Khả năng tương tác (Interoperability): Dù có nỗ lực tạo ra các Mô hình Dữ liệu Chung (như PCORnet hay OMOP), việc thiếu chuẩn hóa cấu trúc dữ liệu giữa các hãng phần mềm EHR khác nhau vẫn làm cản trở quá trình gộp dữ liệu.    
   Thiết kế khác biệt: Hệ thống EHR dùng trong bệnh viện được tối ưu hóa cho "giao diện lâm sàng và thao tác nhanh", trong khi EHR dùng cho nghiên cứu lại yêu cầu "định dạng có cấu trúc và metadata mở rộng".    
   Pháp lý và Quyền riêng tư: Sự khác biệt về luật bảo vệ dữ liệu (như HIPAA ở Mỹ hay GDPR ở Châu Âu) tạo ra rào cản lớn khi muốn chia sẻ dữ liệu để huấn luyện các mô hình AI xuyên quốc gia.