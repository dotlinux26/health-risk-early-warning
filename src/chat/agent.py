"""Chat agent: tích lũy dữ liệu nhật ký sức khỏe, đánh giá khi đủ dữ liệu, yêu cầu bổ sung khi thiếu."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from src.chat.parser import RECOMMENDED_METRICS, ChatParser
from src.chat.store import ChatStore
from src.config import Config, CONFIG
from src.data.loader import load_long_df
from src.ingest.pipeline import ingest_file
from src.main import assess_patient
from src.tier2_knowledge.rules import KnowledgeBase
from src.tier3_risk.scoring import RiskScorer

# Mức rủi ro hiển thị bằng tiếng Việt (frontend không dùng mã).
_LEVEL_VI = {"THAP": "THẤP", "TRUNG_BINH": "TRUNG BÌNH", "CAO": "CAO"}

COMMANDS = {
    "trạng thái": "status",
    "trang thai": "status",
    "tình trạng": "status",
    "tinh trang": "status",
    "status": "status",
    "báo cáo": "report",
    "bao cao": "report",
    "report": "report",
    "xóa dữ liệu": "reset",
    "xoa du lieu": "reset",
    "reset": "reset",
}

# Nhận diện chế độ chẩn đoán chuyên biệt từ câu hỏi tự nhiên.
# Ví dụ "đánh giá nguy cơ tăng huyết áp" -> modes=["htn"].
MODE_KEYWORDS: dict[str, tuple[str, list[str]]] = {
    "tăng huyết áp": ("htn", ["htn"]),
    "tang huyet ap": ("htn", ["htn"]),
    "huyết áp": ("htn", ["htn"]),
    "huyet ap": ("htn", ["htn"]),
    "tim": ("cv", ["cv"]),
    "tim mạch": ("cv", ["cv"]),
    "tim mach": ("cv", ["cv"]),
    "tiểu đường": ("dm", ["dm"]),
    "tieu duong": ("dm", ["dm"]),
    "đái tháo đường": ("dm", ["dm"]),
    "dai thao duong": ("dm", ["dm"]),
    "đường huyết": ("dm", ["dm"]),
    "duong huyet": ("dm", ["dm"]),
    "thận": ("ckd", ["ckd"]),
    "than": ("ckd", ["ckd"]),
    "hô hấp": ("resp", ["resp"]),
    "ho hap": ("resp", ["resp"]),
    "spo2": ("resp", ["resp"]),
    "cân nặng": ("met", ["met"]),
    "can nang": ("met", ["met"]),
    "cân": ("met", ["met"]),
    "bmi": ("met", ["met"]),
}


class ChatAgent:
    def __init__(
        self,
        store: ChatStore | None = None,
        config: Config = CONFIG,
        scorer: RiskScorer | None = None,
    ) -> None:
        self.store = store or ChatStore()
        self.config = config
        self.parser = ChatParser()
        self.scorer = scorer or RiskScorer(config, kb=KnowledgeBase())
        self._last_message = ""

    # ------------------------------------------------------------------ #
    def handle(self, patient_id: str, message: str, file_path: str | Path | None = None,
               day: str | None = None) -> dict:
        """Xử lý một tin nhắn chat. Trả về dict chứa `reply` và dữ liệu cấu trúc.

        day: ngày ghi nhận (YYYY-MM-DD) do người dùng chọn trên giao diện. Nếu
        không truyền, dùng ngày trong tin nhắn (nếu có) hoặc ngày hiện tại.
        """
        message = (message or "").strip()
        pid = (patient_id or "P001").strip()
        self._last_message = message
        record_day: date | None = None
        if day:
            try:
                record_day = pd.Timestamp(day).date()
            except Exception:
                record_day = None

        if file_path is not None:
            return self._handle_file(pid, file_path, record_day)

        # Lệnh điều khiển
        for kw, action in COMMANDS.items():
            if kw in message.lower():
                return self._run_command(pid, action)

        # Nếu không nhận được chỉ số nào -> trả lời hướng dẫn
        records = self.parser.parse(message, day=record_day)
        if not records:
            return self._guide(pid)

        added = self.store.append(pid, records)
        status = self.store.status(pid, self.config.min_points)
        return self._reply_after_record(pid, records, added, status, message)

    # ------------------------------------------------------------------ #
    def _handle_file(self, pid: str, file_path: str | Path, day: date | None = None) -> dict:
        df = ingest_file(file_path, patient_id=pid)
        if df.empty:
            return {
                "reply": "Không trích xuất được chỉ số nào từ file này.\n"
                         "Nhập trực tiếp, ví dụ: \"Huyết áp 135/85, nhịp tim 78\".",
                "ready": self.store.status(pid, self.config.min_points)["ready"],
                "collected": self.store.status(pid, self.config.min_points),
            }
        from src.ingest.parsers import ParsedRecord

        records = [
            ParsedRecord(metric=r.metric, value=float(r.value), unit=str(r.unit),
                         date=day or pd.Timestamp(r.timestamp).date())
            for r in df.itertuples()
        ]
        added = self.store.append(pid, records)
        status = self.store.status(pid, self.config.min_points)
        return self._reply_after_record(pid, records, added, status,
                                        f"[file {Path(file_path).name}]")

    # ------------------------------------------------------------------ #
    def _snapshot(self, pid: str) -> dict[str, float]:
        """Snapshot chỉ số hiện tại (giá trị mới nhất mỗi chỉ số) của bệnh nhân."""
        df = self.store.load(pid)
        if df.empty:
            return {}
        wide = load_long_df(df)[pid]
        last = wide.iloc[-1]
        return {c: float(last[c]) for c in wide.columns
                if c != "timestamp" and not pd.isna(last[c])}

    def _detect_modes(self, message: str) -> list[str] | None:
        """Nhận diện chế độ chẩn đoán chuyên biệt từ câu hỏi tự nhiên."""
        if not message:
            return None
        msg = message.lower()
        found: list[str] = []
        for kw, (_label, modes) in MODE_KEYWORDS.items():
            if kw in msg:
                for m in modes:
                    if m not in found:
                        found.append(m)
        return found or None

    def _run_command(self, pid: str, action: str) -> dict:
        if action == "status":
            st = self.store.status(pid, self.config.min_points)
            lines = [self._fmt_status(st)]
            modes = self._detect_modes(self._last_message or "")
            quick = self._quick_snapshot(pid, modes=modes)
            if quick["risk_level"] != "KHONG_PHAT_HIEN":
                lines.append(f"Đánh giá sơ bộ hiện tại: **{quick['risk_level']}** — {quick['summary']}")
            if st["ready"]:
                lines.append("Gửi lệnh \"báo cáo\" để thực hiện đánh giá.")
            return {"reply": "\n".join(lines), "ready": st["ready"], "collected": st, "snapshot": self._snapshot(pid)}

        if action == "report":
            st = self.store.status(pid, self.config.min_points)
            df = self.store.load(pid)
            if df.empty:
                return {"reply": "Chưa có dữ liệu. Gửi nhật ký sức khỏe hàng ngày để tích lũy.",
                        "ready": False, "collected": st}
            modes = self._detect_modes(self._last_message) if self._last_message else None
            result = self._assess_df(df, modes=modes)
            result["reply"] = self._fmt_assessment(pid, result, ready=st["ready"], forced=True)
            result["ready"] = st["ready"]
            result["collected"] = st
            result["snapshot"] = self._snapshot(pid)
            return result

        if action == "reset":
            self.store.reset(pid)
            return {"reply": f"Đã xóa toàn bộ dữ liệu của {pid}.",
                    "ready": False, "collected": self.store.status(pid, self.config.min_points)}

        return {"reply": "Không nhận diện được yêu cầu.", "ready": False}

    # ------------------------------------------------------------------ #
    def _reply_after_record(self, pid: str, records, added: int, status: dict, source: str) -> dict:
        lines: list[str] = []
        if added == 0:
            lines.append("Các chỉ số này đã được ghi nhận trong ngày (bỏ qua trùng lặp).")
        elif records:
            names = ", ".join(dict.fromkeys(r.metric for r in records))
            lines.append(f"Đã ghi nhận ({source}): {names}.")

        lines.append(self._fmt_status(status))

        # Đánh giá ngay theo tri thức y khoa từ snapshot hiện tại — dù chỉ có
        # 1 chỉ số (máy đo HA ở nhà, cân tự đo). Không bắt buộc đủ 7 ngày.
        df = self.store.load(pid)
        modes = self._detect_modes(self._last_message or "") if not status["ready"] else None
        if status["ready"]:
            result = self._assess_df(df)
            lines.append(self._fmt_assessment(pid, result, ready=True))
            result["reply"] = "\n".join(lines)
            result["ready"] = True
            result["snapshot"] = self._snapshot(pid)
            return result

        # Chưa đủ chuỗi thời gian nhưng vẫn đánh giá tức thì theo luật.
        result = self._assess_df(df, modes=modes)
        if result["risk_level"] != "INSUFFICIENT_DATA":
            lines.append(self._fmt_assessment(pid, result, ready=False))
            lines.append(self._ask_more(status))
            result["reply"] = "\n".join(lines)
            result["ready"] = False
            result["collected"] = status
            result["snapshot"] = self._snapshot(pid)
            return result

        quick = self._quick_snapshot(pid, modes=modes)
        if quick["risk_level"] != "KHONG_PHAT_HIEN":
            lines.append(f"Đánh giá sơ bộ hôm nay: **{quick['risk_level']}** — {quick['summary']}")
        lines.append(self._ask_more(status))
        return {
            "reply": "\n".join(lines),
            "ready": False,
            "risk_level": quick["risk_level"] if quick["risk_level"] != "KHONG_PHAT_HIEN" else None,
            "collected": status,
        }

    def _quick_snapshot(self, pid: str, modes: list[str] | None = None) -> dict:
        df = self.store.load(pid)
        if df.empty:
            return {"risk_level": "KHONG_PHAT_HIEN", "summary": ""}
        wide = load_long_df(df)[pid]
        last = wide.iloc[-1]
        snapshot = {c: float(last[c]) for c in wide.columns
                    if c != "timestamp" and not pd.isna(last[c])}
        res = self.scorer.score([], snapshot=snapshot, modes=modes)
        if res.affected_systems:
            return {"risk_level": res.risk_level, "summary": ", ".join(res.affected_systems)}
        return {"risk_level": "KHONG_PHAT_HIEN", "summary": ""}

    def _assess_df(self, df: pd.DataFrame, modes: list[str] | None = None) -> dict:
        wide = load_long_df(df)[df["patient_id"].iloc[0]]
        result = assess_patient(wide, self.config, self.scorer, modes=modes)
        if modes:
            result["modes"] = modes
        # Luật kích hoạt (đầy đủ metadata) — để frontend cấu hình suy luận
        # từng model giống tab "So sánh luận giải" của /benchmark.
        rules = [e for e in result.get("evidence", []) if e.get("rule_id")]
        if rules:
            result["rules"] = rules
        result["score_weights"] = self.config.risk_weights
        # Mỗi model ML cho một điểm tổng hợp 3 tầng RIÊNG (stat + knowledge
        # + model đó + trend) — để so sánh mức khác biệt giữa các model.
        try:
            from src.experiments.view import explain_patient
            snap = self._snapshot(df["patient_id"].iloc[0])
            critical = any(e.get("severity", 0) >= self.config.critical_rule_severity
                           for e in result.get("evidence", []) if e.get("rule_id"))
            score_context = {
                "components": result.get("components") or {},
                "weights": self.config.risk_weights,
                "critical": critical,
                "floor": self.config.critical_rule_floor,
            }
            result["ml_all"] = [
                {"key": r["key"], "name": r["name"], "family": r["family"],
                 "ml_score": r["risk_score"], "ml_level": r["level"],
                 "total_score": r.get("total_score"), "total_level": r.get("total_level")}
                for r in explain_patient(snap, rules=rules, score_context=score_context)
            ]
        except Exception:
            pass
        return result

    # ------------------------------------------------------------------ #
    def _fmt_status(self, st: dict) -> str:
        if not st["has_data"]:
            return "Chưa có dữ liệu. Gửi dòng nhật ký, ví dụ: \"Huyết áp 125/80, nhịp tim 76\"."
        metrics = ", ".join(f"{m['metric']} ({m['days']} ngày)" for m in st["metrics"]) or "chưa có chỉ số nào"
        need = st["needed_dates"] - st["unique_dates"]
        line = f"Đã có **{st['unique_dates']}/{st['needed_dates']}** ngày đo. Các chỉ số: {metrics}."
        if st["ready"]:
            line += " Đủ dữ liệu để phân tích chuỗi thời gian."
        else:
            line += f" (cần thêm ~{max(need, 1)} ngày đo để cá nhân hóa)."
        return line

    def _ask_more(self, st: dict) -> str:
        hints = ", ".join(desc for _, desc in RECOMMENDED_METRICS)
        return (
            "Gửi thêm các lần đo — mỗi ngày một dòng, ví dụ:\n"
            "• " + "Huyết áp 128/82, nhịp tim 74, cân nặng 78" + "\n"
            "• " + "Đường huyết lúc đói 6.8" + "\n\n"
            "Chế độ đánh giá chuyên biệt (gõ đầu câu): đánh giá nguy cơ tăng huyết "
            "áp / tiểu đường / thận / hô hấp / cân nặng.\n\n"
            f"Ưu tiên theo dõi: {hints}."
        )

    def _fmt_assessment(self, pid: str, result: dict, ready: bool, forced: bool = False) -> str:
        head = f"**BÁO CÁO ĐẦY ĐỦ** ({pid})" if ready else f"**BÁO CÁO SƠ BỘ** ({pid})"
        lines = [head, f"- Mức rủi ro: **{_LEVEL_VI.get(result['risk_level'], result['risk_level'])}** (điểm {result['risk_score']:.3f})"]
        detail = self._fmt_score_explain(result)
        if detail:
            lines.append("\n<details><summary>🧮 Chi tiết cách tính điểm</summary>")
            lines.extend(detail)
            lines.append("</details>")
        if result.get("modes"):
            labels = {
                "htn": "tăng huyết áp", "dm": "đái tháo đường", "ckd": "suy thận mạn",
                "resp": "hô hấp", "met": "chuyển hóa", "cv": "tim mạch", "endo": "nội tiết",
            }
            modes_str = ", ".join(labels.get(m, m) for m in result["modes"])
            lines.append(f"- Chế độ đánh giá: **{modes_str}**")
        lines.append(f"- Hệ cơ quan có khả năng ảnh hưởng: "
                     f"{', '.join(result['affected_systems']) if result['affected_systems'] else 'chưa xác định'}")

        # 1) Luật lâm sàng được kích hoạt — quyết định theo tri thức y khoa.
        rule_ev = [e for e in result.get("evidence", []) if e.get("rule_id")]
        if rule_ev:
            lines.append("\n*Cảnh báo theo luật lâm sàng:*")
            for e in rule_ev[:5]:
                url = e.get("source_url")
                src = f" (nguồn: {e.get('rule_id')} — {url})" if url else ""
                meta = []
                if e.get("source_page"):
                    meta.append(f"trang {e['source_page']}")
                if e.get("source_section"):
                    meta.append(e["source_section"])
                detail = f", {', '.join(meta)}" if meta else ""
                excerpt = e.get("source_excerpt")
                ex = f" — trích: “{excerpt}”" if excerpt else ""
                lines.append(f"  • {e['message']}{src}{detail}{ex}")

        # 2) Chỉ số theo dõi.
        detail = [r for r in result.get("metrics_detail", [])
                  if r["range_status"] in ("CAO", "THẤP") or r["flagged"]]
        if detail:
            lines.append("\n*Chỉ số theo dõi:*")
            for r in detail:
                unit = f" {r['unit']}" if r["unit"] else ""
                seg = f"{r['name']}: **{r['current']:.1f}{unit}** ({r['range_status']}"
                if r["deviation"] is not None:
                    seg += f", vượt {r['deviation']:.1f}{unit})"
                else:
                    seg += ")"
                if r["baseline_mean"] is not None and r["delta"] is not None:
                    seg += f" — {r['delta']:+.1f}{unit} so với đường cơ sở {r['baseline_mean']:.1f}, xu hướng {r['trend_name']}"
                lines.append(f"  • {seg}")

        # 3) Minh chứng thống kê còn lại.
        stat_ev = [e for e in result.get("evidence", []) if not e.get("rule_id")]
        if stat_ev:
            lines.append("\n*Minh chứng dữ liệu:*")
            for e in stat_ev[:4]:
                lines.append(f"  • {e['message']}")

        # 4) Điểm cuối theo từng mô hình ML — mỗi model cho điểm tổng hợp
        # 3 tầng riêng, để so sánh mức khác biệt (gom trong <details>).
        ml_all = result.get("ml_all")
        if isinstance(ml_all, list) and ml_all:
            lines.append("\n<details><summary>📊 Điểm tổng hợp theo từng mô hình ML</summary>")
            lines.append("Mỗi mô hình cho một điểm cuối riêng (tổng hợp 3 tầng: thống kê + "
                         "tri thức y khoa + model đó + xu hướng):")
            for m in ml_all:
                lines.append(f"- {m['name']}: **{m['total_score']:.3f}** → "
                             f"**{_LEVEL_VI.get(m['total_level'], m['total_level'])}** "
                             f"(ML {m['ml_score']:.3f} → {_LEVEL_VI.get(m['ml_level'], m['ml_level'])})")
            lines.append("_Tham khảo bổ sung; kết luận cuối do bác sĩ xác nhận._</details>")

        # 5) Mức độ đầy đủ dữ liệu — đánh dấu rõ ca thiếu chỉ số, không để ngầm.
        suf = result.get("data_sufficiency")
        if isinstance(suf, dict) and suf.get("needed"):
            flag = {
                "CAO": "đầy đủ",
                "TRUNG_BINH": "tương đối đủ",
                "THAP": "THIẾU DỮ LIỆU",
            }.get(suf.get("level"), "chưa xác định")
            lines.append(f"\n*Độ đầy đủ dữ liệu:* **{flag}** — {suf.get('note', '')}")
            if suf.get("missing_metrics"):
                lines.append(f"  _Thiếu: {', '.join(suf['missing_metrics'])} — gửi thêm các chỉ số này "
                             "để đánh giá đầy đủ hơn._")

        if not ready:
            lines.append("\n_Đánh giá theo tri thức y khoa; chưa đủ lịch sử để cá nhân hóa._")
        return "\n".join(lines)

    def _fmt_score_explain(self, result: dict) -> list[str]:
        """Giải thích điểm rủi ro: thành phần + trọng số + ngưỡng xếp loại."""
        comp = result.get("components")
        if not isinstance(comp, dict) or not comp:
            return []
        w = self.config.risk_weights
        total = result.get("risk_score")
        level = result.get("risk_level")

        labels = {"stat": "thống kê", "knowledge": "tri thức y khoa", "ml": "mô hình ML", "trend": "xu hướng"}
        parts = " + ".join(
            f"{labels.get(k, k)} {comp.get(k, 0.0):.2f}×{w.get(k, 0.0):.2f}"
            for k in w
            if comp.get(k, 0.0) is not None
        )
        low, high = self.config.risk_level_thresholds

        if not parts:
            return []
        lines = [f"- Cách tính điểm: **{total:.3f}** = {parts}"]
        lines.append(
            f"  _Chuẩn xếp loại: THẤP < {low:.2f} · TRUNG BÌNH {low:.2f}–{high:.2f} · CAO ≥ {high:.2f}. "
            f"Kết quả: **{_LEVEL_VI.get(level, level)}**._"
        )

        # An toàn lâm sàng: luật nghiêm trọng đẩy điểm sàn.
        critical = any(e.get("rule_id") and e.get("severity", 0) >= self.config.critical_rule_severity
                      for e in result.get("evidence", []))
        if critical:
            lines.append(
                f"  _Có luật lâm sàng nghiêm trọng kích hoạt → điểm được nâng sàn lên "
                f"tối thiểu {self.config.critical_rule_floor:.2f} (an toàn lâm sàng)._"
            )
        return lines

    def _guide(self, pid: str) -> dict:
        """Không khớp lệnh cũng không khớp chỉ số nào -> KHÔNG phản hồi gì."""
        return {
            "reply": "",
            "silent": True,
            "ready": False,
            "collected": self.store.status(pid, self.config.min_points),
        }
