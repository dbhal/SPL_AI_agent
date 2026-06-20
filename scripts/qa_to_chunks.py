"""
Converts auto-generated Q&A pairs into chunks
that get added to the vector store.

IMPORTANT — file responsibilities:
  curated_chunks.json  — hand-written, permanent, NEVER touched by this script
  auto_qa_chunks.json  — auto-generated from QA, rebuilt fresh every workflow run
  manual_chunks.json   — DO NOT USE (kept for legacy compat only)
"""
import json


def convert():
    with open("data/auto_qa_dataset.json", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    # Convert each Q&A into a chunk — fresh every time, no accumulation
    new_chunks = []
    for i, pair in enumerate(qa_pairs):
        chunk = {
            "text": f"{pair['question']} {pair['answer']}",
            "source_url": pair.get("source_url", "https://www.spl.ise.vt.edu"),
            "chunk_index": i,
            "word_count": len(pair["answer"].split()),
        }
        new_chunks.append(chunk)

    # Write to its OWN file — never touches curated_chunks.json
    with open("data/auto_qa_chunks.json", "w", encoding="utf-8") as f:
        json.dump(new_chunks, f, indent=2, ensure_ascii=False)

    print(f"✅ Written {len(new_chunks)} auto QA chunks to data/auto_qa_chunks.json")
    print(f"   (curated_chunks.json is untouched)")


if __name__ == "__main__":
    convert()
