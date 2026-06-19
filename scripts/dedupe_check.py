"""
Deduplication checker and fixer for SPL knowledge base.

Checks all 3 data files for exact-text duplicates and removes them.
Run this any time you want to verify or clean up duplicates —
including right after a GitHub Actions run, or before any fine-tuning.

Usage:
    python scripts/dedupe_check.py            # just reports duplicates
    python scripts/dedupe_check.py --fix       # reports AND removes them
"""
import json
import sys
from collections import Counter


def check_file(path: str, text_key: str = "text", fix: bool = False) -> int:
    """
    Check one JSON file for duplicate entries based on text_key.
    Returns the number of duplicates found (and removed if fix=True).
    """
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"{path}: not found, skipping")
        return 0

    print(f"\n{path}")
    print(f"  Total entries: {len(data)}")

    # Build the text to compare — different files use different keys
    if text_key == "text":
        texts = [d.get("text", "") for d in data]
    else:  # auto_qa_dataset.json uses question+answer
        texts = [d.get("question", "") + " " + d.get("answer", "") for d in data]

    counts = Counter(texts)
    dupes = {t: c for t, c in counts.items() if c > 1}

    total_waste = sum(dupes.values()) - len(dupes)
    print(f"  Unique entries: {len(counts)}")
    print(f"  Duplicate groups: {len(dupes)}")
    print(f"  Wasted duplicate copies: {total_waste}")

    if dupes and total_waste > 0:
        for t in list(dupes.keys())[:3]:
            print(f"    Example (x{dupes[t]}): {t[:100]}...")

    if fix and total_waste > 0:
        seen = set()
        deduped = []
        for d, t in zip(data, texts):
            if t not in seen:
                seen.add(t)
                deduped.append(d)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(deduped, f, indent=2, ensure_ascii=False)

        print(f"  FIXED — removed {len(data) - len(deduped)} duplicates")
        print(f"  New total: {len(deduped)}")

    return total_waste


def main():
    fix = "--fix" in sys.argv

    print("=" * 60)
    print("SPL Knowledge Base — Duplicate Check" + (" + Fix" if fix else ""))
    print("=" * 60)

    total_waste = 0
    total_waste += check_file("data/spl_chunks.json", "text", fix)
    total_waste += check_file("data/manual_chunks.json", "text", fix)
    total_waste += check_file("data/auto_qa_dataset.json", "qa", fix)

    print("\n" + "=" * 60)
    if total_waste == 0:
        print("CLEAN — no duplicates found across any file")
    elif fix:
        print(f"FIXED — removed {total_waste} duplicate entries total")
        print("Remember to also sync to backend/data/ and push")
    else:
        print(f"FOUND {total_waste} duplicate entries — run with --fix to remove them")
    print("=" * 60)


if __name__ == "__main__":
    main()