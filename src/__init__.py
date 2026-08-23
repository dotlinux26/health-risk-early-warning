"""Hệ thống đánh giá nguy cơ sức khỏe cá nhân hóa (3 tầng).

Gói Python gồm:
  - data:        nạp, làm sạch, đặc trưng hóa dữ liệu chuỗi chỉ số cơ thể
  - tier1_anomaly: phát hiện bất thường & xu hướng cá nhân hóa
  - tier2_knowledge: ánh xạ tri thức y khoa dạng luật
  - tier3_risk:  tổng hợp điểm rủi ro và sinh báo cáo
  - models:      mô hình ML (LightGBM chính, LSTM đối chứng) + SHAP
"""

__version__ = "0.1.0"
