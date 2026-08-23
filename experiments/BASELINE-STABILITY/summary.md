# P0.3 Ổn định đường cơ sở cá nhân theo độ dài cửa sổ N

- Thiết kế: 150 bệnh nhân tổng hợp × 150 ngày × 4 chỉ số; seed=42.
- Tham chiếu: cửa sổ 90 ngày (giống production). So sánh μ, σ, z-band.

| N (ngày) | \|Δμ\| (đơn vị σ₉₀) | P95 \|Δμ\| | \|Δσ\| tương đối | Đổi vùng z | Đổi cờ z≥2σ | Số quan sát |
|---|---|---|---|---|---|---|
| 3 | 0.615 | 1.503 | 0.460 | 21.5% | 2.7% | 30022 |
| 5 | 0.527 | 1.282 | 0.363 | 19.0% | 2.8% | 32729 |
| 7 | 0.463 | 1.118 | 0.309 | 16.5% | 2.6% | 33178 |
| 14 | 0.331 | 0.796 | 0.215 | 11.9% | 2.3% | 33133 |
| 30 | 0.206 | 0.498 | 0.130 | 7.4% | 1.6% | 33233 |
| 90 | 0.000 | 0.000 | 0.000 | 0.0% | 0.0% | 33263 |

## Cách đọc

- `|Δμ|` nhỏ (≪1σ) nghĩa là mean baseline hội tụ sớm; nếu tăng nhanh khi
  N giảm, baseline ngắn dễ bị nhiễu kéo lệch.
- `Đổi cờ z≥2σ` = tỉ lệ ngày mà kết luận "bất thường rõ" đảo trạng thái
  so với tham chiếu 90 ngày — đây là con số ảnh hưởng trực tiếp đến UX:
  quyết định hiển thị trạng thái baseline (CHƯA ỔN ĐỊNH / ĐỦ ĐIỀU KIỆN)
  trong UI theo bảng này, không phải cảm tính.

> Giới hạn: dữ liệu tổng hợp phục vụ định lượng hành vi thống kê của
> cửa sổ trượt; khi có dữ liệu dọc thật sẽ lặp lại đúng protocol này.
