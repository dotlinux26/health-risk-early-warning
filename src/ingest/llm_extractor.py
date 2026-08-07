"""Trích xuất cấu trúc bằng LLM (TÙY CHỌN) — dùng khi regex không đủ mạnh.

Hoạt động với bất kỳ server nào tương thích OpenAI API (mặc định Ollama tại
http://localhost:11434/v1, có thể đổi qua biến môi trường INGEST_LLM_URL).
Nếu không có server/API key, hệ thống tự rơi về regex (không lỗi).
"""
from __future__ import annotations

import json
import os

PROMPT = """Bạn là trợ lý trích xuất dữ liệu y tế. Đọc văn bản báo cáo sức khỏe bên dưới
và trả về JSON array duy nhất, mỗi phần tử có dạng:
{"metric": "<tên chỉ số chuẩn: systolic_bp|diastolic_bp|heart_rate|glucose|glucose_fasting|hba1c|creatinine|egfr|spo2|bmi|weight|height>",
 "value": <số>,
 "unit": "<đơn vị>",
 "date": "<YYYY-MM-DD hoặc rỗng>"}
Chỉ xuất các chỉ số xuất hiện rõ trong văn bản. Không thêm giải thích, không thêm markdown.

VĂN BẢN:
{text}"""


def extract_with_llm(text: str) -> list[dict]:
    """Gọi LLM (OpenAI-compatible) để lấy danh sách chỉ số. Lỗi -> trả [].

    Cấu hình: INGEST_LLM_URL (mặc định Ollama), INGEST_LLM_MODEL (mặc định llama3.1).
    """
    url = os.environ.get("INGEST_LLM_URL", "http://localhost:11434/v1/chat/completions")
    model = os.environ.get("INGEST_LLM_MODEL", "llama3.1")

    try:
        import requests
    except ImportError:
        return []

    try:
        resp = requests.post(
            url,
            json={
                "model": model,
                "messages": [{"role": "user", "content": PROMPT.format(text=text[:8000])}],
                "temperature": 0,
            },
            timeout=60,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        # Bóc JSON từ phản hồi (chống trường hợp mô hình thêm markdown)
        start, end = content.find("["), content.rfind("]")
        if start == -1 or end == -1:
            return []
        return json.loads(content[start : end + 1])
    except Exception:
        return []
