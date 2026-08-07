# 07. Định vị đề tài: Khung hỗ trợ quyết định lâm sàng (Clinical Decision Support Framework)

> Mục đích của tài liệu này: xác lập một cách rõ ràng, nhất quán vị thế khoa học
> của đề tài — đặc biệt trả lời câu hỏi thường gặp của hội đồng: *"Đây có phải
> là một mô hình AI không?"* và *"Mô hình của nhóm so với Delphi/LSTM thì như
> thế nào?"*. Tài liệu dùng làm định hướng thống nhất khi viết các chương khác.

---

## 1. Tuyên bố định vị

> **Đề tài KHÔNG xây dựng một mô hình AI mới.**
>
> **Đề tài xây dựng một KHUNG hỗ trợ quyết định lâm sàng (Clinical Decision
> Support Framework — CDSF), trong đó mô hình học máy chỉ là MỘT thành phần
> có thể thay thế, được đưa vào (ép theo) một quy trình nhiều tầng có kiểm
> soát nhằm tạo ra cảnh báo sớm giải thích được và an toàn về mặt lâm sàng.**

Hai câu trên là câu trả lời cốt lõi cho mọi câu hỏi về định vị. Từ đó suy ra:

- Sản phẩm của đề tài là **khung** (framework/kiến trúc), không phải **mô hình**.
- Mô hình AI hiện tại trong khung là **LightGBM**, được chọn vì phù hợp dữ liệu
  dạng bảng, dữ liệu nhỏ và có khả năng giải thích (SHAP).
- LightGBM **hoàn toàn có thể được thay thế** bằng LSTM, GRU, hoặc các mô hình
  foundation model hàng triệu tham số (Chronos, TimesFM, Transformer tạo sinh)
  mà kiến trúc khung không hề thay đổi. Chính tính chất "cắm – thay được" này
  chứng minh giá trị của đề tài nằm ở khung, không nằm ở một mô hình cụ thể.

---

## 2. Đề tài là AI hay không? — Phân định phạm trù

Câu hỏi "có phải AI không" cần được trả lời theo từng tầng, không theo toàn hệ thống:

| Tầng | Nội dung | Có phải AI? |
|------|----------|-------------|
| Tầng 1 | Z-Score cá nhân hóa, EWMA/STL, Isolation Forest, sai số dự báo | Phần lớn là **thống kê** + ML vô giám sát; không phải mô hình dự đoán chính |
| Tầng 2 | Rule engine tri thức y khoa (JSON) | **Không phải AI học máy** — là hệ chuyên gia (expert system), dựa trên hướng dẫn lâm sàng |
| Tầng 3 | LightGBM + SHAP ước lượng điểm nguy cơ | **Đây là thành phần AI duy nhất** (Machine Learning) |

Kết luận:

- Đề tài không tuyên bố "xây dựng mô hình AI".
- Đề tài tuyên bố "xây dựng **hệ thống hỗ trợ quyết định lai (hybrid CDSS)**,
  trong đó AI (học máy) là một thành phần".
- Gọi cả hệ thống là "mô hình AI" là **sai phạm trù** và cần tránh trong báo cáo.

### 2.1. Gỡ AI ra, hệ thống vẫn hoạt động

Một điểm quan trọng để trả lời hội đồng: **nếu bỏ hoàn toàn mô hình AI đi, khung
của đề tài vẫn là một phần mềm ra quyết định hoàn chỉnh** — được xây trên toán
thống kê (Z-Score cá nhân hóa, EWMA/STL, Isolation Forest) kết hợp hệ luật y khoa
(if/else trên ngưỡng lâm sàng). Nghĩa là:

- Không AI → vẫn ra được cảnh báo (bằng thống kê + luật).
- Có AI (LightGBM) → thêm một **nguồn điểm nguy cơ học từ dữ liệu** được đối
  chiếu chéo với hai nguồn trên.

Từ đó suy ra hai tính chất của khung:

1. **AI-agnostic:** khung không phụ thuộc AI; AI chỉ là một thành phần nâng cấp.
2. **Điều kiện tối thiểu:** hệ thống luôn hoạt động kể cả khi chưa có dữ liệu
   huấn luyện hay khi dữ liệu quá nhỏ để học máy — bởi lõi quyết định là luật
   và thống kê, không phải mô hình.

### 2.2. "Ép AI theo khung" nghĩa là gì

Thay vì để một mô hình học máy tự phát ra kết quả rồi dừng lại, khung của đề tài
"ép" mô hình đó phải chạy bên trong một quy trình có kiểm soát:

```
Dữ liệu cá nhân (chuỗi thời gian chỉ số cơ thể)
        │
        ▼
Tầng 1  Phân tích thống kê theo chính cá nhân (baseline riêng, không theo quần thể)
        ▼
Tầng 2  Kiểm tra bằng luật tri thức y khoa (ngưỡng lâm sàng, hệ cơ quan)
        ▼
Tầng 3  Mô hình ML (LightGBM) ước lượng điểm nguy cơ → tổng hợp 4 nguồn điểm
        ▼
        Cảnh báo kèm GIẢI THÍCH từng bước (chỉ số nào, lệch bao nhiêu, luật nào kích hoạt)
```

Ý nghĩa: AI không quyết định một mình. Mọi quyết định cuối cùng đều phải đi qua
các lớp kiểm soát (thống kê + tri thức y khoa) và bắt buộc sinh ra lời giải thích
cho bác sĩ. Đó là cách khung "kìm" AI lại trong phạm vi an toàn lâm sàng — khác
hẳn việc để một mô hình hộp đen tự trả về xác suất.

Nói cách khác, khung **chuyển hóa AI từ mô hình sinh (generative) thành mô hình
luận giải dựa trên luật**: AI không tự sinh kết luận, mà chỉ đóng góp một giá trị
điểm nguy cơ (có kèm giải thích đặc trưng) vào một quyết định được lập luận bằng
luật và thống kê. Đầu ra cuối cùng luôn là một chuỗi lý do có thể kiểm chứng —
đúng chuẩn của một công cụ hỗ trợ quyết định, không phải của một mô hình hộp đen.

---

## 3. Phép so sánh ĐÚNG và SAI

### 3.1. Những cách so sánh SAI (tránh trong báo cáo)

| Cách so sánh sai | Vì sao sai |
|---|---|
| "So sánh mô hình của nhóm với Delphi" | Delphi là một **mô hình**; hệ thống của nhóm là một **khung nhiều tầng** → khác cấp độ, so được là phi phạm trù |
| "So sánh hệ thống của nhóm với LSTM" | LSTM chỉ là một lựa chọn ở Tầng 3, không phải là đối thủ của cả khung |
| "Mô hình chúng tôi chính xác hơn Delphi" | Chưa có benchmark trực tiếp cùng dữ liệu → không được khẳng định |

### 3.2. Cách so sánh ĐÚNG

So sánh phải **cùng cấp độ**:

| Cấp độ | So sánh hợp lệ |
|--------|-----------------|
| Cấp khung (framework) | Khung 3 tầng của nhóm vs các hệ thống CDSS khác — theo tiêu chí **kiến trúc**: cá nhân hóa, khả năng giải thích, chi phí, tính thay thế mô hình |
| Cấp mô hình ML (thành phần Tầng 3) | LightGBM vs LSTM/GRU vs Transformer thời gian — trên **cùng bộ dữ liệu**, cùng metric AUROC/AUPRC (đây là benchmark mà đề tài báo cáo được) |
| Cấp bài toán | Đề tài giải bài toán **cảnh báo sớm (early warning)**; Delphi/Foresight giải bài toán **dự đoán diễn tiến bệnh (disease trajectory prediction)** — đây là hai bài toán khác nhau, không thể đem so trực tiếp |

---

## 4. Đối chiếu với các mô hình tạo sinh (Delphi, Foresight)

| Tiêu chí | Delphi / Foresight | Đề tài này |
|----------|--------------------|------------|
| Bài toán | Dự đoán bệnh/biến cố *tiếp theo* từ lịch sử bệnh án | Cảnh báo sớm *thay đổi bất thường* trong chỉ số hằng ngày |
| Đầu vào | Hồ sơ bệnh án điện tử (EHR), triệu bệnh nhân | Chuỗi chỉ số cơ thể cá nhân, dữ liệu nhỏ |
| Loại mô hình | Transformer tạo sinh (GPT-style), hàng triệu tham số | Khung hybrid: thống kê + luật + LightGBM (module có thể thay) |
| Giải thích | Hạn chế (hộp đen) | Giải thích được theo thiết kế (reason chain từng tầng) |
| Dữ liệu cần | Rất lớn + GPU | Nhỏ, chạy được trên máy tính thường |
| Cá nhân hóa | Theo quần thể | Theo chính cá nhân (baseline riêng) |
| Phạm vi triển khai | Tập đoàn lớn, hạ tầng mạnh | Phòng khám nhỏ, thiết bị thường |

Điểm khác biệt cốt lõi: hai bên **không cùng bài toán** nên không so sánh "ai chính
xác hơn". Giá trị của đề tài không nằm ở chỗ vượt qua Delphi về accuracy, mà nằm ở
chỗ đề xuất một **khung cảnh báo sớm chi phí thấp, giải thích được, triển khai
được ở quy mô nhỏ** — một vùng mà các mô hình tạo sinh không phục vụ được.

---

## 5. Bốn lợi thế của cách định vị khung

1. **Cá nhân hóa**: mọi ngưỡng, Z-Score, xu hướng đều so với chính bệnh nhân,
   không so với quần thể (khác biệt với các mô hình học theo dân số).
2. **Giải thích được (XAI by design)**: cảnh báo luôn kèm lý do từng tầng —
   chỉ số nào, lệch bao nhiêu, luật nào kích hoạt, mô hình ML đóng góp bao nhiêu.
3. **Chi phí thấp, dữ liệu nhỏ**: chạy được trên dữ liệu hàng trăm bệnh nhân,
   không cần GPU, không cần EHR triệu hồ sơ — phù hợp với điều kiện đơn vị nghiên
   cứu và phòng khám nhỏ.
4. **Tính thay thế mô hình (model-agnostic)**: khung không ràng buộc vào một
   mô hình cụ thể. LightGBM hiện tại có thể thay bằng LSTM/GRU, rồi nâng lên
   time-series foundation model (Chronos, TimesFM), thậm chí Transformer tạo sinh
   khi đủ dữ liệu — kiến trúc khung giữ nguyên, chỉ thay module Tầng 3. Điều này
   chứng tỏ đóng góp của đề tài là **kiến trúc**, không phải một mô hình.

---

## 6. Nguyên tắc viết báo cáo

- KHÔNG viết: *"mô hình của đề tài là LightGBM"*, *"chúng tôi xây dựng mô hình AI"*.
- NÊN viết: *"hệ thống của đề tài là một khung hỗ trợ quyết định lâm sàng; mô hình
  học máy hiện tại trong khung là LightGBM..."*.
- Khi báo cáo AUC/AUPRC: nói rõ đó là kết quả của **thành phần ML (Tầng 3)**
  trên dữ liệu thật, không phải là "độ chính xác" của cả hệ thống.
- Khi so sánh: luôn nêu rõ so sánh ở cấp nào (khung / mô hình / bài toán).

### Câu trả lời mẫu cho hội đồng

> **Hỏi:** "Đây có phải là mô hình AI không?"
>
> **Trả lời:** "Đề tài của chúng tôi xây dựng một khung hỗ trợ quyết định lâm sàng.
> AI — cụ thể là mô hình học máy LightGBM — chỉ là một thành phần trong khung,
> đảm nhiệm việc ước lượng điểm nguy cơ. Quyết định cuối cùng được tổng hợp từ ba
> nguồn: phân tích thống kê cá nhân hóa, tri thức y khoa dạng luật, và mô hình học
> máy; mọi cảnh báo đều kèm giải thích từng bước."

> **Hỏi:** "So với Delphi/LSTM thì mô hình của nhóm thế nào?"
>
> **Trả lời:** "Delphi giải bài toán dự đoán bệnh tiếp theo từ bệnh án triệu bệnh
> nhân; LSTM là một lựa chọn mô hình học sâu cho chuỗi thời gian. Hai hệ thống này
> không cùng cấp với khung của chúng tôi. Chúng tôi so sánh ở hai cấp: ở cấp khung,
> so về khả năng cá nhân hóa, giải thích và chi phí triển khai; ở cấp mô hình, chúng
> tôi benchmark LightGBM với LSTM trên cùng bộ dữ liệu để chọn thành phần phù hợp.
> Khung của chúng tôi không phụ thuộc một mô hình cố định — LightGBM hôm nay hoàn
> toàn có thể thay bằng LSTM hoặc foundation model khi dữ liệu cho phép."

> **Hỏi:** "Nếu bỏ mô hình AI đi thì hệ thống còn gì?"
>
> **Trả lời:** "Hệ thống vẫn hoạt động đầy đủ như một phần mềm ra quyết định dựa
> trên toán thống kê và hệ luật y khoa. AI chỉ là một nguồn điểm nguy cơ bổ sung
> học từ dữ liệu; lõi quyết định là luật + thống kê, nên hệ thống luôn chạy được
> kể cả khi chưa có dữ liệu huấn luyện. Điều này chứng tỏ đóng góp của chúng tôi
> nằm ở kiến trúc khung, không phải ở một mô hình cụ thể."

---

*Tài liệu nội bộ. Không thay thế chẩn đoán của bác sĩ.*
