"""Mô hình con trích xuất dữ liệu sức khỏe từ báo cáo PDF/DOCX.

Luồng: PDF/DOCX -> text -> parser (regex + optional LLM) -> dataset chuẩn (long format).
Dataset sinh ra đúng schema đầu vào của pipeline chính:
    patient_id | timestamp | metric | value | unit
"""
