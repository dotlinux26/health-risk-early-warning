# P0.4 Độ nhạy risk score theo bộ trọng số

- Dữ liệu: `data/sample_long.csv` (5 bệnh nhân thật) + 150 bệnh nhân giả từ NHANES (pipeline 3 tầng đầy đủ).
- Lưu ý an toàn: luật severity ≥ 0.7 vẫn kích hoạt sàn TRUNG_BÌNH ở mọi bộ.

| Bộ trọng số | Phân bố mức | Score mean±std |
|---|---|---|
| v1 (hiện hành) | {'THAP': 115, 'TRUNG_BINH': 38, 'CAO': 2} | 0.254±0.146 |
| v2 | {'THAP': 120, 'TRUNG_BINH': 33, 'CAO': 2} | 0.258±0.156 |
| v3 | {'THAP': 114, 'TRUNG_BINH': 39, 'CAO': 2} | 0.283±0.140 |

| Cặp so sánh | Đồng thuận mức | Chuyển mức | std(Δscore) |
|---|---|---|---|
| v1 (hiện hành) vs v2 | 96.8% | {'TRUNG_BINH→THAP': 5} | 0.015 |
| v1 (hiện hành) vs v3 | 99.4% | {'THAP→TRUNG_BINH': 1} | 0.012 |
| v2 vs v3 | 96.1% | {'THAP→TRUNG_BINH': 6} | 0.025 |

> Kết luận áp dụng: nếu đồng thuận cao (≥95%) → risk level bền vững với
> thiết kế trọng số. Nếu thấp → risk score nhạy với trọng số; các trọng
> số hiện tại chỉ là thiết kế ban đầu và phải được ghi rõ trong báo cáo.
> Không hiệu chỉnh trọng số nhằm cải thiện số khi chưa có outcome thật.
