"""
Automatically generates Q&A pairs from scraped chunks
using Groq API. Run once when you have new chunks.
"""
import json
import groq
import os

client = groq.Groq(api_key=os.environ["GROQ_API_KEY"])

def generate_qa_from_chunk(chunk_text: str, source_url: str) -> list[dict]:
    prompt = f"""You are helping build a Q&A dataset for the SPL chatbot.

Given this text from the SPL website:
{chunk_text}

Generate 3-5 natural questions a website visitor might ask,
along with accurate answers based only on the text above.

Return ONLY a JSON array like this:
[
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}}
]
No other text."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1000,
    )

    text = response.choices[0].message.content.strip()
    try:
        qa_pairs = json.loads(text)
        for pair in qa_pairs:
            pair["source_url"] = source_url
        return qa_pairs
    except:
        return []


def generate_all():
    with open("data/spl_chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)

    all_qa = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}...")
        qa_pairs = generate_qa_from_chunk(chunk["text"], chunk["source_url"])
        all_qa.extend(qa_pairs)
        print(f"  Generated {len(qa_pairs)} Q&A pairs")

    with open("data/auto_qa_dataset.json", "w", encoding="utf-8") as f:
        json.dump(all_qa, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Generated {len(all_qa)} Q&A pairs automatically")
    print(f"   Saved to data/auto_qa_dataset.json")


if __name__ == "__main__":
    generate_all()