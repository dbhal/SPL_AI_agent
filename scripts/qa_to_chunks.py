"""
Converts auto-generated Q&A pairs into chunks
that get added to the vector store.
Each Q&A becomes a chunk so the bot can find
answers to indirect or vague questions.
"""
import json

def convert():
    with open("data/auto_qa_dataset.json", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    # Load existing manual chunks
    try:
        with open("data/manual_chunks.json", encoding="utf-8") as f:
            manual = json.load(f)
    except:
        manual = []

    # Convert each Q&A into a chunk
    new_chunks = []
    for i, pair in enumerate(qa_pairs):
        chunk = {
            "text": f"{pair['question']} {pair['answer']}",
            "source_url": pair.get("source_url", "https://www.spl.ise.vt.edu"),
            "chunk_index": len(manual) + i,
            "word_count": len(pair['answer'].split()),
        }
        new_chunks.append(chunk)

    # Merge with existing manual chunks
    all_chunks = manual + new_chunks

    with open("data/manual_chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"✅ Added {len(new_chunks)} Q&A chunks to manual_chunks.json")
    print(f"   Total manual chunks: {len(all_chunks)}")

if __name__ == "__main__":
    convert()