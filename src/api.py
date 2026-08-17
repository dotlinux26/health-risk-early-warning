"""API FastAPI cho frontend.

Chạy:
    uvicorn src.api:app --reload --port 8000
    hoặc  python -m src.api

Endpoints:
    GET  /                       trang demo nhỏ (upload + đánh giá)
    GET  /api/health             kiểm tra dịch vụ
    GET  /api/kb                 cơ sở tri thức y khoa (cho frontend hiển thị)
    POST /api/assess             đánh giá từ JSON danh sách chỉ số
    POST /api/assess_docs        upload PDF/DOCX -> ingest -> đánh giá
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from src.config import CONFIG
from src.core.pipeline import VALUE_COLUMNS, assess_patient
from src.data.loader import load_long_df
from src.data.preprocess import build_baseline, impute_missing, resample_to_daily
from src.ingest.pipeline import ingest_file
from src.tier1_anomaly.detector import run_tier1, tier1_summary
from src.tier2_knowledge.rules import KnowledgeBase
from src.tier3_risk.scoring import RiskScorer

from src.chat.agent import ChatAgent
from src.chat.store import ChatStore

app = FastAPI(
    title="HealthRisk API — Cảnh báo sớm nguy cơ sức khỏe (3 tầng)",
    version="0.3.0",
    description="Hệ thống hỗ trợ quyết định. Không thay thế chẩn đoán của bác sĩ.",
)

kb = KnowledgeBase()
scorer = RiskScorer(CONFIG, kb=kb)
chat_store = ChatStore()
chat_agent = ChatAgent(store=chat_store, config=CONFIG, scorer=scorer)

TMP_DIR = Path("data/uploads")
CHAT_PAGE = Path(__file__).parent / "chat" / "static" / "index.html"
RULES_PAGE = Path(__file__).parent / "chat" / "static" / "rules.html"
BENCH_PAGE = Path(__file__).parent / "chat" / "static" / "benchmark.html"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Trang demo tối giản cho frontend (upload file hoặc dán JSON)."""
    return """
    <!doctype html><html lang="vi"><head><meta charset="utf-8">
    <title>HealthRisk Demo</title></head><body style="font-family:sans-serif">
    <h2>Cảnh báo sớm nguy cơ sức khỏe</h2>
    <form method="post" action="/api/assess_docs" enctype="multipart/form-data">
      <label>Bệnh nhân: <input name="patient_id" value="P001"></label><br><br>
      <label>File PDF/DOCX: <input type="file" name="file" accept=".pdf,.docx,.doc,.txt"></label><br><br>
      <button>Đánh giá từ file</button>
    </form>
    <p>Hoặc dùng Swagger: <a href="/docs">/docs</a></p>
    </body></html>
    """


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": "0.3.0"}


@app.get("/chat", response_class=HTMLResponse)
def chat_page() -> str:
    """Hộp thoại trò chuyện tích lũy dữ liệu sức khỏe."""
    return CHAT_PAGE.read_text(encoding="utf-8")


@app.post("/api/chat")
async def chat(payload: dict[str, Any]) -> JSONResponse:
    """Body: {"patient_id": "P001", "message": "Huyết áp 135/85, nhịp tim 80", "date": "2026-08-17"}"""
    pid = str(payload.get("patient_id", "P001"))
    message = str(payload.get("message", ""))
    day = payload.get("date")
    return JSONResponse(chat_agent.handle(pid, message, day=str(day) if day else None))


@app.post("/api/chat_file")
async def chat_file(file: UploadFile = File(...), patient_id: str = Form("P001"),
                    date: str = Form("")) -> JSONResponse:
    """Upload PDF/DOCX/TXT qua chat -> ingest -> tích lũy -> phản hồi.

    date (tùy chọn): ngày ghi nhận do người dùng chọn, ghi đè ngày trong file.
    """
    if file.filename is None or file.filename.split(".")[-1].lower() not in {"pdf", "docx", "doc", "txt"}:
        return JSONResponse({"error": "Chỉ hỗ trợ pdf, docx, doc, txt"}, status_code=400)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = TMP_DIR / file.filename
    tmp_path.write_bytes(await file.read())
    return JSONResponse(chat_agent.handle(patient_id, "", file_path=tmp_path, day=date or None))


@app.get("/api/chat/patients")
def chat_patients() -> dict:
    """Danh sách mã bệnh nhân đang có dữ liệu trong data/chat/."""
    return {"patients": chat_store.list_patients()}


@app.get("/api/chat/status")
def chat_status(patient_id: str = "P001") -> dict:
    return chat_store.status(patient_id, CONFIG.min_points)


@app.post("/api/chat/reset")
def chat_reset(patient_id: str = "P001") -> JSONResponse:
    chat_store.reset(patient_id)
    return JSONResponse({"ok": True, "message": f"Đã xóa dữ liệu {patient_id}"})


@app.get("/api/kb")
def knowledge_base() -> dict:
    return kb.to_dict()


# --------------------------------------------------------------------------- #
# Quản lý cơ sở tri thức (CRUD) — cho bác sĩ thêm luật / hệ / chỉ số qua giao diện
# --------------------------------------------------------------------------- #
@app.get("/rules", response_class=HTMLResponse)
def rules_page() -> str:
    """Giao diện quản lý luật lâm sàng cho bác sĩ."""
    return RULES_PAGE.read_text(encoding="utf-8")


@app.post("/api/kb/rules")
def kb_add_rule(rule: dict[str, Any]) -> JSONResponse:
    try:
        saved = kb.add_rule(rule)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return JSONResponse({"ok": True, "rule": saved})


@app.put("/api/kb/rules/{rule_id}")
def kb_update_rule(rule_id: str, patch: dict[str, Any]) -> JSONResponse:
    try:
        saved = kb.update_rule(rule_id, patch)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return JSONResponse({"ok": True, "rule": saved})


@app.delete("/api/kb/rules/{rule_id}")
def kb_delete_rule(rule_id: str) -> JSONResponse:
    try:
        kb.delete_rule(rule_id)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return JSONResponse({"ok": True, "rule_id": rule_id})


@app.post("/api/kb/systems")
def kb_add_system(payload: dict[str, Any]) -> JSONResponse:
    try:
        system_key = str(payload.get("system_key", "")).strip()
        label = str(payload.get("label", "")).strip()
        saved = kb.add_system(system_key, label)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return JSONResponse({"ok": True, "system": saved})


@app.post("/api/kb/metrics")
def kb_add_metric(payload: dict[str, Any]) -> JSONResponse:
    try:
        metric = str(payload.get("metric", "")).strip()
        saved = kb.add_metric(metric, payload.get("info", {}))
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return JSONResponse({"ok": True, "metric": {metric: saved}})


# --------------------------------------------------------------------------- #
# Benchmark đa mô hình — xem kết quả thực nghiệm + so sánh luận giải
# --------------------------------------------------------------------------- #
@app.get("/benchmark", response_class=HTMLResponse)
def benchmark_page() -> str:
    """Giao diện xem kết quả benchmark + so sánh các mô hình."""
    return BENCH_PAGE.read_text(encoding="utf-8")


@app.get("/api/benchmark")
def benchmark_summary() -> dict:
    from src.experiments.view import (
        available_input_metrics,
        build_summary,
        list_experiments,
    )

    return {
        "summary": build_summary(),
        "experiments": list_experiments(),
        "input_metrics": available_input_metrics(),
    }


@app.get("/api/benchmark/exp/{experiment_id}/curves")
def benchmark_curves(experiment_id: str):
    """Ảnh ROC/PR/Calibration của một experiment (evidence package)."""
    from fastapi.responses import FileResponse

    import src.experiments.view as v

    path = v.EXPERIMENTS_DIR / experiment_id / "curves.png"
    if not path.exists():
        return JSONResponse({"error": "Không tìm thấy curves.png"}, status_code=404)
    return FileResponse(path, media_type="image/png")


@app.post("/api/benchmark/explain")
def benchmark_explain(payload: dict[str, Any]) -> JSONResponse:
    """Body: {"metrics": {"systolic_bp": 202, ...}} -> so sánh luận giải các model."""
    from src.experiments.view import clinical_context, explain_patient

    values = payload.get("metrics") or {}
    model_keys = payload.get("models") or None
    if not isinstance(values, dict) or not values:
        return JSONResponse({"error": "Cần truyền metrics"}, status_code=400)
    try:
        values = {k: float(v) for k, v in values.items()}
    except (TypeError, ValueError):
        return JSONResponse({"error": "Giá trị metrics phải là số"}, status_code=400)
    context = clinical_context(values)
    score_context = payload.get("score_context") or None
    results = explain_patient(values, model_keys=model_keys, rules=context.get("rules"),
                              score_context=score_context)
    return JSONResponse({"results": results, "clinical": context})


def _assess_json(records: list[dict[str, Any]], patient_id: str, modes: list[str] | None = None) -> JSONResponse:
    """Đánh giá từ danh sách bản ghi chuẩn (long format).

    modes: chế độ chẩn đoán chuyên biệt, vd ["htn"] chỉ xét luật tăng huyết áp.
    """
    import pandas as pd

    df = pd.DataFrame(records)
    if not {"timestamp", "metric", "value"}.issubset(df.columns):
        return JSONResponse({"error": "Thiếu cột (cần: timestamp, metric, value)"}, status_code=400)
    df["patient_id"] = patient_id
    wide = load_long_df(df)[patient_id]
    result = assess_patient(wide, CONFIG, scorer, modes=modes)
    return JSONResponse({"patient_id": patient_id, "modes": modes, **result})


@app.post("/api/assess")
async def assess(payload: dict[str, Any]) -> JSONResponse:
    """Body: {"patient_id": "P001", "mode": "htn", "records": [{"timestamp": "...", "metric": "...", "value": 128}, ...]}"""
    patient_id = str(payload.get("patient_id", "P001"))
    records = payload.get("records", [])
    if not records:
        return JSONResponse({"error": "records rỗng"}, status_code=400)
    modes = payload.get("mode")
    if isinstance(modes, str) and modes.strip():
        modes = [m.strip() for m in modes.split(",") if m.strip()]
    elif isinstance(modes, list):
        modes = [str(m) for m in modes]
    else:
        modes = None
    return _assess_json(records, patient_id, modes=modes)


@app.post("/api/assess_docs")
async def assess_docs(
    file: UploadFile = File(...),
    patient_id: str = Form("P001"),
    use_llm: bool = Form(False),
) -> JSONResponse:
    """Upload PDF/DOCX -> ingest -> dataset chuẩn -> đánh giá 3 tầng."""
    if file.filename is None or file.filename.split(".")[-1].lower() not in {"pdf", "docx", "doc", "txt"}:
        return JSONResponse({"error": "Chỉ hỗ trợ pdf, docx, doc, txt"}, status_code=400)

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = TMP_DIR / file.filename
    tmp_path.write_bytes(await file.read())

    df = ingest_file(tmp_path, patient_id=patient_id, use_llm=use_llm)
    if df.empty:
        return JSONResponse(
            {"error": "Không trích xuất được chỉ số nào từ file", "patient_id": patient_id},
            status_code=422,
        )

    # dataset chuẩn -> CSV để lưu trữ / đưa vào pipeline
    dataset_csv = TMP_DIR / f"{patient_id}_ingested.csv"
    df.to_csv(dataset_csv, index=False)

    ingested = df.copy()
    ingested["timestamp"] = ingested["timestamp"].astype(str)

    wide = load_long_df(df)[patient_id]
    result = assess_patient(wide, CONFIG, scorer)
    return JSONResponse(
        {
            "patient_id": patient_id,
            "ingested": ingested.to_dict(orient="records"),
            "dataset_csv": str(dataset_csv),
            **result,
        }
    )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
