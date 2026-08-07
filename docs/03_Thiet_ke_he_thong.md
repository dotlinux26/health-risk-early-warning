# 03. Thiết kế hệ thống đa tầng (System Architecture)

**Nguyên tắc cốt lõi:** *Không so sánh bệnh nhân với quần thể — so sánh họ với chính họ trong quá khứ, rồi chiếu lên tri thức y khoa, cuối cùng lượng hóa thành điểm rủi ro giải thích được.*

> **Định vị:** sơ đồ dưới đây là **khung hỗ trợ quyết định lâm sàng (CDSF)** của
> đề tài. Mô hình học máy (LightGBM) chỉ là một module trong Tầng 3 và **có thể
> thay thế** bằng LSTM/GRU hoặc foundation model mà kiến trúc khung không đổi.
> Xem thêm [docs/07](07_Dinh_vi_de_tai.md).

---

## 1. Sơ đồ tổng thể

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DỮ LIỆU ĐẦU VÀO                              │
│   Chuỗi thời gian chỉ số cơ thể (huyết áp, nhịp tim, đường huyết,   │
│   cân nặng, creatinine, SpO2, HbA1c...) + tuổi, giới, bệnh nền      │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ TẦNG 1 — PHÂN TÍCH BẤT THƯỜNG & XU HƯỚNG CÁ NHÂN                   │
│  • Z-Score cá nhân hóa:  Z = (X - μ_cá nhân) / σ_cá nhân            │
│  • Isolation Forest đa chiều (co-biến đổi: HA + nhịp tim + đường huyết)│
│  • Phân rã xu hướng EWMA / STL                                       │
│  ► Đầu ra: danh sách {chỉ số, Z, mức độ lệch, hướng, cửa sổ}        │
└───────────────────────────────┬─────────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ TẦNG 2 — ÁNH XẠ TRI THỨC Y KHOA                                     │
│  Cơ sở tri thức JSON (luật từ hướng dẫn lâm sàng):                  │
│   • HA tâm thu/tâm trương tăng đột biến      → Hệ tim mạch          │
│   • Glucose / HbA1c vượt ngưỡng tham chiếu  → Hệ nội tiết            │
│   • Creatinine máu tăng                     → Chức năng thận        │
│  ► Đầu ra: tập {hệ cơ quan, luật kích hoạt, mức nghiêm trọng}       │
└───────────────────────────────┬─────────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ TẦNG 3 — TỔNG HỢP RỦI RO & HỖ TRỢ QUYẾT ĐỊNH                        │
│  • Điểm rủi ro tổng hợp (điểm Tầng 1 × trọng số tri thức Tầng 2     │
│    + điểm mô hình ML (LightGBM + SHAP))                             │
│  • Phân loại: THẤP / TRUNG BÌNH / CAO                               │
│  ► ĐẦU RA CHUẨN (3 thành phần bắt buộc):                            │
│    1) Phân loại rủi ro    2) Hệ cơ quan ảnh hưởng                   │
│    3) Giải thích thông số + Khuyến nghị chuyên khoa                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Mô tả chi tiết từng tầng

### TẦNG 1 — Anomaly Detection cá nhân hóa

**Vai trò:** Bỏ qua "mốc" cộng đồng, chỉ đo độ lệch so với **đường cơ sở của chính cá nhân** trong một chu kỳ (ví dụ 90 ngày gần nhất).

**Các phương pháp:**

| Phương pháp | Công thức / Ý tưởng | Ứng dụng |
|-------------|---------------------|----------|
| Z-Score cá nhân hóa | `Z = (X - μ_window) / σ_window` | Đo độ lệch 1 chiều, dễ giải thích |
| Isolation Forest | Cô lập điểm dị trong không gian đa chiều | Phát hiện thay đổi *đồng thời* nhiều chỉ số |
| EWMA (kiểm soát quá trình) | `EWMA_t = λ·X_t + (1-λ)·EWMA_{t-1}` | Bắt xu hướng trượt nhẹ, nhiễu thấp |
| STL decomposition | Tách trend + seasonal + residual | Nhận diện xu hướng dài hạn (tăng cân, HA tăng dần) |
| LSTM Autoencoder (tùy chọn) | Học tái tạo chuỗi, lỗi tái tạo cao = bất thường | Đối chứng sâu, ít minh bạch hơn |

**Đầu ra Tầng 1:** `AnomalyRecord[]` với cấu trúc:
```json
{
  "metric": "systolic_bp",
  "current": 152.0,
  "baseline_mean": 128.0,
  "z_score": 2.6,
  "window_days": 90,
  "direction": "up",
  "trend": "rising",
  "flagged": true
}
```

### TẦNG 2 — Ánh xạ tri thức y khoa

**Vai trò:** Biến "bất thường" thành "ý nghĩa lâm sàng" bằng luật tĩnh từ hướng dẫn.

**Cấu trúc luật (JSON — dễ chỉnh sửa, versioning):**
```json
{
  "rule_id": "R_CV_01",
  "name": "Tăng huyết áp đột biến",
  "system": "tim_manh",
  "condition": {
    "metric": "systolic_bp",
    "op": ">",
    "threshold": 140,
    "logic": "and",
    "other": { "metric": "diastolic_bp", "op": ">", "threshold": 90 }
  },
  "severity": "high",
  "specialty": "Khoa Tim mạch",
  "evidence": "ESC/ESH 2018, JNC-8"
}
```

**Bảng ánh xạ chỉ số → hệ cơ quan (khởi đầu):**

| Chỉ số | Hệ cơ quan | Luật gợi ý |
|--------|-----------|------------|
| Huyết áp tâm thu/tâm trương | Tim mạch | Tăng đột biến / HA ≥ 140/90 |
| Nhịp tim | Tim mạch | Nhịp nhanh/chậm kéo dài |
| Glucose, HbA1c | Nội tiết | Vượt ngưỡng 7.0 / 6.5% |
| Creatinine, eGFR | Thận | Creatinine tăng / eGFR giảm |
| BMI, vòng bụng | Chuyển hóa | Tăng theo xu hướng |
| SpO2 | Hô hấp | Giảm < 94% |
| LDL/HDL/Triglyceride | Mỡ máu | Vượt ngưỡng |

### TẦNG 3 — Tổng hợp rủi ro & giải thích

**Công thức điểm rủi ro:**
```
Risk_Score = w_stat · Σ(Tầng1 z-score chuẩn hóa) + w_kb · Σ(mức nghiêm trọng Tầng2)
             + w_ml · Mô hình học máy (0–1)  +  w_trend · Độ dốc xu hướng
Phân loại:  Low < 0.33  |  0.33 ≤ Medium < 0.66  |  High ≥ 0.66
```

**Đầu ra chuẩn hóa (bắt buộc 3 thành phần):**
```json
{
  "risk_level": "CAO",
  "risk_score": 0.78,
  "affected_systems": ["tim_manh", "noi_tiet"],
  "evidence": [
    {"metric": "systolic_bp", "z_score": 2.6, "message": "Huyết áp tâm thu 3 tháng gần nhất lệch +2.6σ so với trung bình cá nhân"},
    {"metric": "hba1c", "value": 6.9, "message": "HbA1c vượt ngưỡng 6.5%"}
  ],
  "recommendations": ["Khoa Tim mạch", "Khoa Nội tiết"],
  "disclaimer": "Hệ thống chỉ HỖ TRỢ QUYẾT ĐỊNH, không thay thế chẩn đoán của bác sĩ."
}
```

---

## 3. Sơ đồ luồng dữ liệu & Xử lý lỗi

1. **Nạp & validate:** schema bắt buộc `patient_id, timestamp, metric, value` (+ `unit`).
2. **Làm sạch:** loại outlier thô (lỗi nhập), nội suy missing (≤ 30% cửa sổ), LOCF.
3. **Căn chỉnh thời gian:** resample về chuỗi đều (ngày/tuần) — quan trọng cho STL/LSTM.
4. **Tầng 1 → Tầng 2 → Tầng 3:** chạy tuyến tính; mỗi tầng ghi log để truy vết.
5. **Không đủ dữ liệu:** nếu < 7 điểm dữ liệu thì trả `INSUFFICIENT_DATA`, không đoán bừa (an toàn).
6. **Truy vết (XAI):** mỗi cảnh báo lưu `reason_chain` gồm (chỉ số, Z, luật, hệ cơ quan, trọng số) để bác sĩ kiểm tra ngược.

---

## 4. Bảo mật & Đạo đức

- Dữ liệu ẩn danh hóa (không lưu CCCD/HoTen), mã hóa khi lưu trữ.
- Tuân thủ GDPR/HIPAA nếu dùng dữ liệu thực; có phương án chạy hoàn toàn cục bộ.
- Disclaimer rõ ràng trong mọi đầu ra; không xuất khẩu ra bên ngoài nếu không đồng ý.
- Logging đầy đủ để đối chiếu khuyến nghị (audit trail).

---

## 5. Kiến trúc triển khai (deployment) đề xuất

```
[Client / Dashboard (Streamlit)]
        │  POST /assess  (JSON chuỗi chỉ số)
        ▼
[FastAPI Gateway] ──► [Pipeline 3 tầng (module Python)]
        │                        │
        ▼                        ▼
[Report generator (md/PDF)]  [Log store / tri thức JSON]
```

- **Đơn giản nhất:** chạy script `python -m src.main --input data.csv --output report/`.
- **Nâng cao:** FastAPI + Docker, tri thức JSON versioned trong Git.
