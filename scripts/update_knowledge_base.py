"""
Automated Knowledge Base Update Pipeline
==========================================
Runs nightly to keep SPL chatbot up to date.

Windows: Schedule with Task Scheduler
Linux/VT ARC: Schedule with cron

Usage:
    python scripts/update_knowledge_base.py
"""

import subprocess
import shutil
import os
import json
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE = os.path.join(BASE, "data", "update_log.json")


def log(message: str, status: str = "info"):
    """Write timestamped log entry."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "status": status,
        "message": message,
    }
    print(f"[{entry['timestamp']}] {message}")

    # Load existing log
    try:
        with open(LOG_FILE, encoding="utf-8") as f:
            logs = json.load(f)
    except FileNotFoundError:
        logs = []

    logs.append(entry)

    # Keep last 100 entries only
    logs = logs[-100:]

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)


def run_step(name: str, command: list) -> bool:
    """Run a command and return True if successful."""
    log(f"Starting: {name}")
    try:
        result = subprocess.run(
            command,
            cwd=BASE,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            log(f"✅ Done: {name}", "success")
            return True
        else:
            log(f"❌ Failed: {name}\n{result.stderr}", "error")
            return False
    except Exception as e:
        log(f"❌ Error: {name} — {e}", "error")
        return False


def backup_chunks():
    """Backup existing chunks before overwriting."""
    src = os.path.join(BASE, "data", "spl_chunks.json")
    dst = os.path.join(BASE, "data", f"spl_chunks_backup_{datetime.now().strftime('%Y%m%d')}.json")
    if os.path.exists(src):
        shutil.copy(src, dst)
        log(f"Backed up chunks to {dst}")


def compare_chunks() -> dict:
    """Compare new vs old chunks and report changes."""
    try:
        with open(os.path.join(BASE, "data", "spl_chunks.json"), encoding="utf-8") as f:
            new = json.load(f)

        # Find most recent backup
        backups = [f for f in os.listdir(os.path.join(BASE, "data"))
                   if f.startswith("spl_chunks_backup_")]
        if not backups:
            return {"new_count": len(new), "old_count": 0, "change": len(new)}

        latest = sorted(backups)[-1]
        with open(os.path.join(BASE, "data", latest), encoding="utf-8") as f:
            old = json.load(f)

        return {
            "new_count": len(new),
            "old_count": len(old),
            "change": len(new) - len(old),
        }
    except Exception as e:
        return {"error": str(e)}


def update():
    log("=" * 50)
    log("Starting SPL Knowledge Base Update")
    log("=" * 50)

    # Step 1 — Backup existing data
    backup_chunks()

    # Step 2 — Rescrape website
    success = run_step(
        "Scraping SPL website",
        ["python", "scripts/scrape_spl_semantic.py"]
    )
    if not success:
        log("Stopping — scrape failed", "error")
        return False

    # Step 3 — Compare changes
    diff = compare_chunks()
    log(f"Chunks: {diff.get('old_count', 0)} → {diff.get('new_count', 0)} ({diff.get('change', 0):+d} change)")

    # Step 4 — Rebuild vector store
    chroma_dir = os.path.join(BASE, "data", "chroma_db")
    if os.path.exists(chroma_dir):
        shutil.rmtree(chroma_dir)
        log("Deleted old vector store")

    # Step 5 — Run eval to verify quality
    success = run_step(
        "Running evaluation suite",
        ["python", "scripts/run_eval.py"]
    )
    if not success:
        log("Warning — eval failed after update", "warning")

    log("=" * 50)
    log("✅ Knowledge base update complete")
    log("=" * 50)
    return True


if __name__ == "__main__":
    update()