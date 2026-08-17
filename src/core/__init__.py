"""Core package — cấu hình, kiểu dữ liệu, pipeline điều phối 3 tầng.

Lưu ý: __init__ này cố ý KHÔNG import pipeline/config để tránh vòng lặp
(core → tier1 → core). Import trực tiếp theo module:
    from src.core.types import AnomalyRecord
    from src.core.pipeline import assess_patient
"""