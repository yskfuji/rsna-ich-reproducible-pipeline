"""Simple audit logger writing JSONL."""
import json
from pathlib import Path
from datetime import datetime


def log_action(event: str, user_id: str = "local", details: dict | None = None, log_path: str = "logs/audit_log.jsonl"):
    record = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "event": event,
        "user": user_id,
        "details": details or {},
    }
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")
