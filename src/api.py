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
    title="HealthRisk API — Đánh giá nguy cơ sức khỏe cá nhân hóa (3 tầng)",
    version="0.3.0",
    description="Hệ thống hỗ trợ quyết định. Không thay thế chẩn đoán của bác sĩ.",
)

kb = KnowledgeBase()
scorer = RiskScorer(CONFIG, kb=kb)
chat_store = ChatStore()
chat_agent = ChatAgent(store=chat_store, config=CONFIG, scorer=scorer)

TMP_DIR = Path("data/uploads")
APP_PAGE = Path(__file__).parent / "chat" / "static" / "app.html"
CHAT_PAGE = Path(__file__).parent / "chat" / "static" / "index.html"
RULES_PAGE = Path(__file__).parent / "chat" / "static" / "rules.html"
BENCH_PAGE = Path(__file__).parent / "chat" / "static" / "benchmark.html"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Ứng dụng dạng biểu mẫu (P1): Đánh giá · Bản ghi · Luật · Benchmark."""
    return APP_PAGE.read_text(encoding="utf-8")


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
    actor = str(rule.pop("actor", "unknown") or "unknown")
    try:
        saved = kb.add_rule(rule, actor=actor)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return JSONResponse({"ok": True, "rule": saved})


@app.put("/api/kb/rules/{rule_id}")
def kb_update_rule(rule_id: str, patch: dict[str, Any]) -> JSONResponse:
    actor = str(patch.pop("actor", "unknown") or "unknown")
    try:
        saved = kb.update_rule(rule_id, patch, actor=actor)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return JSONResponse({"ok": True, "rule": saved})


@app.delete("/api/kb/rules/{rule_id}")
def kb_delete_rule(rule_id: str, actor: str = "unknown") -> JSONResponse:
    try:
        kb.delete_rule(rule_id, actor=actor)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return JSONResponse({"ok": True, "rule_id": rule_id})


# --------------------------------------------------------------------------- #
# P1 Governance — workflow DRAFT → REVIEW → APPROVED → ACTIVE + audit trail
# --------------------------------------------------------------------------- #
@app.post("/api/kb/rules/{rule_id}/transition")
def kb_transition(rule_id: str, payload: dict[str, Any]) -> JSONResponse:
    """Body: {"to": "review|approved|active|rejected", "actor": "...", "note": "..."}"""
    from src.tier2_knowledge.governance import STATUSES

    to_status = str(payload.get("to", "")).strip().lower()
    if to_status not in STATUSES:
        return JSONResponse(
            {"error": f"Trạng thái '{to_status}' không hợp lệ. "
                      f"Cho phép: {', '.join(STATUSES)}"},
            status_code=400,
        )
    actor = str(payload.get("actor", "unknown") or "unknown")
    note = str(payload.get("note", ""))
    try:
        updated = kb.transition(rule_id, to_status, actor=actor, note=note)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    kb.reload()
    scorer_kb_reload()
    return JSONResponse({"ok": True, "rule": updated})


@app.get("/api/kb/audit")
def kb_audit(limit: int = 200) -> dict:
    """Audit trail quản trị tri thức (mới nhất trước)."""
    from src.tier2_knowledge.governance import read_audit

    return {"entries": read_audit(limit=limit)}


def scorer_kb_reload() -> None:
    """Nạp lại KB cho scorer sau khi luật đổi trạng thái."""
    global scorer
    kb.reload()
    scorer.kb = kb


# --------------------------------------------------------------------------- #
# P1 U10 — Quản lý bản ghi cá nhân theo ngày (CRUD từng ô dữ liệu)
# --------------------------------------------------------------------------- #
@app.get("/api/records/{patient_id}")
def records_table(patient_id: str) -> dict:
    """Bảng dữ liệu cá nhân theo ngày: mỗi hàng một ngày, cột là chỉ số."""
    return chat_store.table_by_date(patient_id)


@app.put("/api/records/{patient_id}")
async def record_upsert(patient_id: str, payload: dict[str, Any]) -> JSONResponse:
    """Body: {"timestamp": "YYYY-MM-DD", "metric": "...", "value": 120 hoặc null, "unit": ""}.

    value=null nghĩa là xóa ô; ô nào chưa có dữ liệu cứ để trống hoàn toàn.
    """
    ts = str(payload.get("timestamp", "")).strip()
    metric = str(payload.get("metric", "")).strip()
    if not ts or not metric:
        return JSONResponse({"error": "Cần timestamp và metric"}, status_code=400)
    value = payload.get("value", None)
    if value is not None:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return JSONResponse({"error": "value phải là số hoặc null"}, status_code=400)
    try:
        saved = chat_store.upsert(patient_id, ts, metric, value,
                                  unit=str(payload.get("unit", "")))
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return JSONResponse({"ok": True, **saved})


@app.delete("/api/records/{patient_id}")
def record_delete(patient_id: str, timestamp: str, metric: str = "") -> JSONResponse:
    try:
        if metric:
            chat_store.delete_value(patient_id, timestamp, metric)
        else:
            chat_store.delete_day(patient_id, timestamp)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    return JSONResponse({"ok": True, "deleted": {"timestamp": timestamp,
                                                 "metric": metric or "*"}})


# --------------------------------------------------------------------------- #
# P1 U9 — Render Markdown chuẩn hóa bằng python-markdown
# --------------------------------------------------------------------------- #
@app.post("/api/render_markdown")
def render_markdown(payload: dict[str, Any]) -> JSONResponse:
    """Body: {"markdown": "# Tiêu đề\\n| a | b |..."} -> {"html": "..."}.

    Extension: tables + fenced_code — đủ cho bảng số liệu và khối mã trong
    evidence/báo cáo, giữ đầu ra thống nhất giữa các trang.
    """
    import markdown as _md

    text = str(payload.get("markdown", ""))
    if not text.strip():
        return JSONResponse({"html": ""})
    html = _md.markdown(text, extensions=["tables", "fenced_code", "nl2br"])
    return JSONResponse({"html": html})


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


@app.get("/api/benchmark/research")
def benchmark_research() -> JSONResponse:
    """Dữ liệu trang nghiên cứu: robustness K2-K4 + provenance + evidence status."""
    import json as _json
    from pathlib import Path as _Path

    def _load(rel: str) -> dict[str, Any]:
        p = _Path("experiments") / rel
        return _json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    summary = _load("summary.json")
    label = _load("LABEL-SENSITIVITY/summary.json")
    baseline = _load("BASELINE-STABILITY/summary.json")
    weight = _load("WEIGHT-SENSITIVITY/summary.json")

    dc = summary.get("data_completeness") or {}
    cal_ok = any((_Path("experiments") / f"EXP-{m}-42" / "calibration.json").exists()
                 for m in ("LGBM", "XGB", "LR"))
    evidence_status = [
        {"item": "Phát triển & kiểm định nội bộ (phân tầng, không rò rỉ)",
         "state": "done",
         "note": "Imputer fit trên train; split 70/15/15 theo seed cố định."},
        {"item": "Baseline cá nhân (z-score theo cửa sổ)",
         "state": "done",
         "note": "Tầng 1 dùng baseline N ngày của chính người dùng."},
        {"item": "Đánh giá độ nhạy định nghĩa nhãn (K2)",
         "state": "done",
         "note": "AUC nhãn B=1.00 vs A=0.935 — vòng lặp nhãn–đầu vào được chứng minh."},
        {"item": "Hiệu chỉnh xác suất (calibration)",
         "state": "done" if cal_ok else "partial",
         "note": "Isotonic fit trên validation cho mọi model; ECE test ≤ 1.7%."},
        {"item": "Ổn định baseline theo cửa sổ (K3)",
         "state": "done",
         "note": "N<7 ngày: |Δμ|≈0.6σ, lật band 21.5% → UI yêu cầu tối thiểu 7–14 ngày."},
        {"item": "Nhạy cảm trọng số fusion (K4)",
         "state": "done",
         "note": "Đồng thuận mức giữa các bộ trọng số ≥ 96%."},
        {"item": "Kiểm định thời gian (temporal / prospective)",
         "state": "partial",
         "note": "NHANES-LMF: AUC tử vong ≤12m 0.821 (split theo thời gian), "
                 "lead time trung vị 9 tháng — cấp cohort/horizon tháng. "
                 "Cửa sổ 30/90 ngày chờ dữ liệu dọc theo ngày."},
        {"item": "Kiểm định ngoài (external dataset khác NHANES)",
         "state": "partial",
         "note": "Hold-out theo thời gian 2017-18 đạt (AUC drop 0.02 ≤ 0.05); "
                 "external theo địa lý/quần thể chưa có."},
        {"item": "Complete-case check khuyết dữ liệu glucose 52%",
         "state": "done",
         "note": "LightGBM/XGB ổn định (|ΔAUC| < 0.01) → impute vô hại cho "
                 "production; LR lệch +0.016 — ghi chú khi báo cáo baseline."},
        {"item": "Thử nghiệm lâm sàng có kiểm soát",
         "state": "todo",
         "note": "Ngoài phạm vi đồ án; ghi nhận là hạn chế."},
    ]
    temporal = {}
    _temporal_path = _Path("experiments") / "EXP-TEMPORAL-LMF" / "summary.json"
    if _temporal_path.exists():
        try:
            t = json.loads(_temporal_path.read_text(encoding="utf-8"))
            a = t.get("tasks", {}).get("death_within_12m", {})
            lt = t.get("tasks", {}).get("lead_time", {})
            surv = t.get("tasks", {}).get("survival_c_index", {})
            lr, lg = a.get("lr", {}), a.get("lgbm", {})
            temporal = {
                "source": t.get("data", {}).get("source"),
                "n_train": t.get("data", {}).get("n_train"),
                "n_test": t.get("data", {}).get("n_test"),
                "lr": {
                    "auc_temporal": lr.get("temporal_split_test", {}).get("roc_auc"),
                    "auc_random": lr.get("random_split_test", {}).get("roc_auc"),
                    "c_index": surv.get("lr", {}).get("c_index_test_full_followup"),
                },
                "lgbm": {
                    "auc_temporal": lg.get("temporal_split_test", {}).get("roc_auc"),
                    "auc_random": lg.get("random_split_test", {}).get("roc_auc"),
                    "c_index": surv.get("lgbm", {}).get("c_index_test_full_followup"),
                },
                "lead_time": lt,
            }
        except Exception:
            temporal = {}
    return JSONResponse({
        "label_sensitivity": {
            "overlap": label.get("labels_overlap"),
            "aggregate": label.get("aggregate"),
            "oracle_note": label.get("oracle_note"),
        },
        "baseline_stability": {
            "design": baseline.get("design"),
            "results": baseline.get("results", []),
        },
        "weight_sensitivity": {
            "versions": weight.get("versions"),
            "pairs": weight.get("pairs"),
        },
        "provenance": {
            "dataset": summary.get("dataset"),
            "n": summary.get("n"),
            "positive": summary.get("positive"),
            "seeds": summary.get("seeds"),
            "missing_rates": dc.get("missing_rates"),
            "completeness_confidence": dc.get("confidence"),
        },
        "temporal_validation": temporal,
        "evidence_status": evidence_status,
    })


@app.get("/api/evidence/ml")
def evidence_ml(score: float | None = None) -> JSONResponse:
    """Bằng chứng tầng ML dạng Markdown (U4/U9).

    score (tùy chọn): áp calibrator production (isotonic, fit trên val của
    EXP-ML-LGBM-42) lên một điểm raw -> hiển thị cặp Raw vs Calibrated.
    """
    import json as _json

    from src.experiments.view import EXPERIMENTS_DIR

    exp_dir = EXPERIMENTS_DIR / "EXP-ML-LGBM-42"
    lines = ["**LightGBM** — mô hình production (NHANES merged, n=16314)"]
    try:
        cal = _json.loads((exp_dir / "calibration.json").read_text(encoding="utf-8"))
        m_none = cal["methods"]["none"]
        m_iso = cal["methods"][cal.get("selected", "isotonic")]
        lines += [
            f"- Đầu ra THÔ của model: **{score:.3f}**" if score is not None
            else "- Đầu ra thô của model: *xem ô Đánh giá*",
            f"- Phương pháp hiệu chỉnh: **{cal.get('selected', 'isotonic')}** "
            "(fit trên validation; test chỉ đánh giá)",
            f"- Brier test: raw {m_none['brier_test']:.4f} → "
            f"calibrated {m_iso['brier_test']:.4f}",
            f"- ECE test: raw {m_none['ece_test']:.1%} → "
            f"calibrated {m_iso['ece_test']:.1%}",
            "- ⚠️ Đầu ra chưa hiệu chỉnh **không được diễn giải là xác suất bệnh**.",
        ]
        if score is not None and cal.get("selected") != "none":
            import joblib as _jl

            cal_path = exp_dir / f"calibrator_{cal['selected']}.joblib"
            if cal_path.exists():
                from src.experiments.calibration import apply_calibration

                p_cal = float(apply_calibration(_jl.load(cal_path), [score])[0])
                lines.insert(2, f"- Sau hiệu chỉnh ({cal['selected']}): **{p_cal:.3f}**")
    except FileNotFoundError:
        lines.append("- Chưa có evidence package calibration — chạy benchmark trước.")
    try:
        met = _json.loads((exp_dir / "metrics.json").read_text(encoding="utf-8"))
        lines.append(f"- ROC-AUC (test): {met['roc_auc']:.4f} · "
                     "phân tầng cắt ngang, không phải dự báo biến cố.")
    except FileNotFoundError:
        pass
    return JSONResponse({"html_md": "\n".join(lines)})


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
