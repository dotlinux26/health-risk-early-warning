# 09. Thảo luận & chuẩn hóa định vị đề tài (ghi chép nội bộ)

---

## 1. Đề tài là gì — khẳng định dứt khoát

Đề tài **KHÔNG phải** "một mô hình AI mới".

Đề tài là:

> **AI-enabled Clinical Decision Support Framework** — khung hỗ trợ quyết định
> lâm sàng có tích hợp AI.

Nếu viết paper, contribution là *"A Hybrid Clinical Decision Support Framework"*,
**không phải** *"A Novel LightGBM Model"*.

**Từ cần dùng:** "Framework tích hợp AI".
**Từ cần tránh:** "Framework AI" (nghe sẽ bị hỏi "AI mới của anh đâu?").

---

## 2. Cấu trúc 3 tầng — tư duy FUSION, không phải phân cấp

Mỗi tầng giải quyết **một loại thông tin khác nhau**, bổ sung nhau — không tầng
nào "đè" tầng nào, không tầng nào là "chuẩn quyết định".

| Tầng | Câu hỏi tầng đó trả lời |
|---|---|
| Tầng 1 — thống kê cá nhân hóa | Có bất thường so với CHÍNH bệnh nhân không? |
| Tầng 2 — luật tri thức y khoa | Bất thường đó có ý nghĩa y khoa gì? |
| Tầng 3 — ML + tổng hợp | Với dữ liệu của hàng nghìn bệnh nhân, nguy cơ tổng thể bao nhiêu? |

**Lưu ý không được nói:** *"Luật là chuẩn quyết định."* — vì hội đồng sẽ hỏi
*"Luật đã luôn đúng thì cần AI làm gì?"* → toang.

**Cách nói đúng:** ba tầng là **fusion thông tin** (hợp nhất nhiều nguồn bằng
chứng thành một quyết định), không phải hierarchy (phân cấp quyền lực).

---

## 3. AI chỉ là plugin của framework — điểm chứng minh giá trị

Model ở Tầng 3 **thay thế được, không khóa cứng**:

- Hôm nay: LightGBM → năm sau: LSTM → Chronos / TimesFM / Transformer.
- Khi đổi model: **không cần sửa** API, rule, parser, report, dashboard.

Điều này chứng minh: **AI chỉ là một thành phần cắm vào framework.** Toàn bộ
phần còn lại của hệ thống độc lập với từng model cụ thể.

---

## 4. Cách gọi Tầng 1 trước hội đồng

Không gọi Tầng 1 là "AI". Nên gọi:

> **Data Analytics / Statistical Anomaly Detection**

Vì Z-Score, EWMA, STL là **thống kê**. Chỉ riêng Isolation Forest là ML không
giám sát — khi cần nói kỹ, trình bày theo dạng:

> "Tầng 1 là phân tích thống kê cá nhân hóa, trong đó Isolation Forest là một
> thuật toán Machine Learning không giám sát."

Cách nói này khó bị bắt bẻ hơn.

---

## 5. Contribution thật của đề tài: cơ chế FUSION

Không phải LightGBM, không phải rule — mà là:

> **Cách tổng hợp nhiều nguồn bằng chứng (rule, ML, trend, stat) thành một
> quyết định rủi ro cuối cùng.**

Ví dụ: rule 0.8, ML 0.2, trend 0.9, stat 0.6 → làm sao ra risk 0.74?

**Bài toán cốt lõi:** hợp nhất bằng chứng đa nguồn thành quyết định. Đây mới là
điểm reviewer / hội đồng quan tâm.

---

## 6. Ý nghĩa với hướng publish (nếu có)

Paper sẽ không phải *"LightGBM for Healthcare"* (loại này đã có rất nhiều), mà là:

> **Hybrid Clinical Decision Support Framework integrating Statistical
> Personalization, Medical Knowledge and Machine Learning.**

Lý do: LightGBM đổi được, rule đổi được, Chronos đổi được — nhưng **framework
giữ nguyên**. Chính sự ổn định của khung mới là giá trị nghiên cứu.

---

## 7. Nhắc việc cho Khánh & An

- [ ] Dùng đúng thuật ngữ: **"AI-enabled Clinical Decision Support Framework"** — không nói "Framework AI".
- [ ] Không nói "luật là chuẩn quyết định" — nói **ba tầng bổ sung nhau (fusion)**.
- [ ] Tầng 1 gọi là **Statistical Anomaly Detection / Data Analytics**; Isolation Forest nêu là ML không giám sát trong đó.
- [ ] Nhấn mạnh **model thay thế được** (LightGBM → LSTM → Chronos) để chứng minh framework.
- [ ] Contribution cốt lõi trình bày là **cơ chế fusion đa nguồn bằng chứng**, không phải model cụ thể.

---

*Tài liệu thảo luận nội bộ. Không thay thế chẩn đoán của bác sĩ.*
