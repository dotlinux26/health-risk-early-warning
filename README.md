# Hệ thống cảnh báo sớm nguy cơ sức khỏe dựa trên phân tích chuỗi thời gian cá nhân hóa

## Đội ngũ thực hiện

| Thành viên | Vai trò |
|------------|---------|
| Nguyễn Đức Cảnh | Trưởng nhóm — Chủ trì toàn bộ phát triển code |
| Nguyễn Khắc Nam Khánh | Nghiên cứu, tổng hợp và viết báo cáo |
| Vũ Đình An | Nghiên cứu, tổng hợp và viết báo cáo |

## Mục lục tài liệu

- [01. Tổng hợp nghiên cứu (5 bài báo)](docs/01_Tong_hop_nghien_cuu.md)
- [02. Kế hoạch dự án](docs/02_Ke_hoach_du_an.md)
- [03. Thiết kế hệ thống 3 tầng](docs/03_Thiet_ke_he_thong.md)
- [04. Lựa chọn & so sánh mô hình](docs/04_Lua_chon_mo_hinh.md)
- [05. Nâng cấp mô hình & Mô hình con trích xuất dữ liệu](docs/05_Mo_hinh_nang_cap_va_ingest.md)
- [06. Báo cáo huấn luyện: kết quả, giới hạn, hướng chuẩn mực hóa](docs/06_Bao_cao_huong_train_va_gioi_han.md)

## Kiến trúc 3 tầng

```
Tầng 1  Phân tích bất thường & xu hướng cá nhân (Z-Score, Isolation Forest, EWMA/STL,
        sai số dự báo chuỗi thời gian — EWMA offline, sẵn sàng Chronos/TimesFM)
Tầng 2  Ánh xạ tri thức y khoa (rule engine trên JSON)
Tầng 3  Tổng hợp rủi ro & hỗ trợ quyết định (LightGBM + SHAP + báo cáo chi tiết từng chỉ số)
```

Mô hình ML hiện tại: LightGBM được huấn luyện trên dữ liệu tổng hợp có nhãn
(`data/models/risk_lgbm.joblib`, AUC 0.9874) và được đối chứng trên dữ liệu thật
Pima Diabetes (AUC 0.8116) và Cleveland Heart Disease (AUC 0.8906). Chi tiết
kết quả, giới hạn và lộ trình: [docs/06](docs/06_Bao_cao_huong_train_va_gioi_han.md).

## Cài đặt

```bash
pip install -r requirements.txt
```

## Vận hành

```bash
./run_api.sh start        # khởi động API + hộp thoại chat (http://127.0.0.1:8000)
python -m src.main --input <file.csv> --output report/
```

## Huấn luyện mô hình

```bash
python -m src.models.train --n-per-condition 120 --seed 42   # train trên dữ liệu tổng hợp
python scripts/download_datasets.py                           # tải dataset chính thống (UCI)
python scripts/train_real_datasets.py                         # đánh giá trên dữ liệu thật
```

## Schema dữ liệu đầu vào

```csv
patient_id,timestamp,metric,value
P001,2025-01-01,systolic_bp,128
P001,2025-01-01,diastolic_bp,82
P001,2025-01-02,systolic_bp,131
```

## Cấu trúc mã nguồn

```
src/
  config.py                 # tham số toàn cục
  main.py                   # pipeline chính
  api.py                    # API FastAPI
  data/                     # nạp, làm sạch, đặc trưng hóa
  tier1_anomaly/            # phát hiện bất thường cá nhân hóa
  tier2_knowledge/          # cơ sở tri thức y khoa (JSON + rule engine)
  tier3_risk/               # điểm rủi ro + báo cáo
  models/                   # LightGBM, SHAP, LSTM (đối chứng)
  ingest/                   # trích xuất dữ liệu từ PDF/DOCX
  chat/                     # hộp thoại tích lũy nhật ký sức khỏe
```
