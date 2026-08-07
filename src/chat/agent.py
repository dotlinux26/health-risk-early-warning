"""Chat agent: tích lũy dữ liệu nhật ký sức khỏe, đánh giá khi đủ dữ liệu, yêu cầu bổ sung khi thiếu."""
from __future__ import annotations

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

COMMANDS = {
    "trạng thái": "status",
    "status": "status",
    "trang thai": "status",
    "báo cáo": "report",
    "bao cao": "report",
    "report": "report",
    "xóa dữ liệu": "reset",
    "xoa du lieu": "reset",
    "reset": "reset",
}


class ChatAgent:
    def __init__(self, store: ChatStore | None = None, config: Config = CONFIG) -> None:
        self.store = store or ChatStore()
        self.config = config
        self.parser = ChatParser()
        self.scorer = RiskScorer(config, kb=KnowledgeBase())

    # ------------------------------------------------------------------ #
    def handle(self, patient_id: str, message: str, file_path: str | Path | None = None) -> dict:
        """Xử lý một tin nhắn chat. Trả về dict chứa `reply` và dữ liệu cấu trúc."""
        message = (message or "").strip()
        pid = (patient_id or "P001").strip()

        if file_path is not None:
            return self._handle_file(pid, file_path)

        # Lệnh điều khiển
        for kw, action in COMMANDS.items():
            if kw in message.lower():
                return self._run_command(pid, action)

        # Nếu không nhận được chỉ số nào -> trả lời hướng dẫn
        records = self.parser.parse(message)
        if not records:
            return self._guide(pid)

        added = self.store.append(pid, records)
        status = self.store.status(pid, self.config.min_points)
        return self._reply_after_record(pid, records, added, status, message)

    # ------------------------------------------------------------------ #
    def _handle_file(self, pid: str, file_path: str | Path) -> dict:
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
                         date=pd.Timestamp(r.timestamp).date())
            for r in df.itertuples()
        ]
        added = self.store.append(pid, records)
        status = self.store.status(pid, self.config.min_points)
        return self._reply_after_record(pid, records, added, status,
                                        f"[file {Path(file_path).name}]")

    # ------------------------------------------------------------------ #
    def _run_command(self, pid: str, action: str) -> dict:
        if action == "status":
            st = self.store.status(pid, self.config.min_points)
            lines = [self._fmt_status(st)]
            if st["ready"]:
                lines.append("Gửi lệnh \"báo cáo\" để thực hiện đánh giá.")
            return {"reply": "\n".join(lines), "ready": st["ready"], "collected": st}

        if action == "report":
            df = self.store.load(pid)
            if df.empty:
                return {"reply": "Chưa có dữ liệu. Gửi nhật ký sức khỏe hàng ngày để tích lũy.",
                        "ready": False, "collected": self.store.status(pid, self.config.min_points)}
            result = self._assess_df(df)
            result["reply"] = self._fmt_assessment(pid, result["risk_level"],
                                                   result["risk_score"],
                                                   result["affected_systems"],
                                                   result["evidence"],
                                                   ready=False, forced=True)
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

        if status["ready"]:
            df = self.store.load(pid)
            result = self._assess_df(df)
            lines.append(self._fmt_assessment(pid, result["risk_level"],
                                              result["risk_score"],
                                              result["affected_systems"],
                                              result["evidence"], ready=True))
            result["reply"] = "\n".join(lines)
            result["ready"] = True
            return result

        # Chưa đủ dữ liệu -> phản hồi sơ bộ theo luật (nếu có) và yêu cầu bổ sung
        quick = self._quick_snapshot(pid)
        if quick["risk_level"] != "KHONG_PHAT_HIEN":
            lines.append(f"Đánh giá sơ bộ hôm nay: **{quick['risk_level']}** — {quick['summary']}")
        lines.append(self._ask_more(status))
        return {
            "reply": "\n".join(lines),
            "ready": False,
            "risk_level": quick["risk_level"] if quick["risk_level"] != "KHONG_PHAT_HIEN" else None,
            "collected": status,
        }

    # ------------------------------------------------------------------ #
    def _quick_snapshot(self, pid: str) -> dict:
        df = self.store.load(pid)
        if df.empty:
            return {"risk_level": "KHONG_PHAT_HIEN", "summary": ""}
        wide = load_long_df(df)[pid]
        last = wide.iloc[-1]
        snapshot = {c: float(last[c]) for c in wide.columns
                    if c != "timestamp" and not pd.isna(last[c])}
        res = self.scorer.score([], snapshot=snapshot)
        if res.affected_systems:
            return {"risk_level": res.risk_level, "summary": ", ".join(res.affected_systems)}
        return {"risk_level": "KHONG_PHAT_HIEN", "summary": ""}

    def _assess_df(self, df: pd.DataFrame) -> dict:
        wide = load_long_df(df)[df["patient_id"].iloc[0]]
        return assess_patient(wide, self.config, self.scorer)

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
            f"Ưu tiên theo dõi: {hints}."
        )

    def _fmt_assessment(self, pid, level, score, systems, evidence, ready, forced=False) -> str:
        head = f"**BÁO CÁO ĐẦY ĐỦ** ({pid})" if ready else f"**BÁO CÁO SƠ BỘ** ({pid})"
        lines = [head, f"- Mức rủi ro: **{level}** (điểm {score:.3f})"]
        lines.append(f"- Hệ cơ quan có khả năng ảnh hưởng: {', '.join(systems) if systems else 'chưa xác định'}")
        lines.append("\n*Minh chứng:*")
        for e in evidence[:6]:
            lines.append(f"  • {e['message']}")
        if not ready:
            lines.append("\n_Đánh giá theo tri thức y khoa; chưa đủ lịch sử để cá nhân hóa._")
        return "\n".join(lines)

    def _guide(self, pid: str) -> dict:
        st = self.store.status(pid, self.config.min_points)
        return {
            "reply": (
                "Không phát hiện chỉ số nào trong tin nhắn.\n"
                "Định dạng nhật ký sức khỏe (mỗi ngày một dòng), ví dụ:\n"
                "• \"Huyết áp 135/85, nhịp tim 80\"\n"
                "• \"Đường huyết lúc đói 6.9, cân nặng 77\"\n"
                "• Hoặc đính kèm file PDF/DOCX báo cáo khám.\n\n"
                "Lệnh: \"trạng thái\", \"báo cáo\", \"xóa dữ liệu\"."
            ),
            "ready": st["ready"],
            "collected": st,
        }
