"""
Run this after every change to check chatbot quality automatically.
Tests against the same combined chunk sources as the live bot.
"""
import json
import os
from rag_pipeline import load_chunks, build_vector_store, query

TEST_CASES = [
    {
        "question": "Who leads SPL?",
        "must_contain": ["Triantis", "Godfrey"],
    },
    {
        "question": "How many PhD students?",
        "must_contain": ["4", "Tatiana", "Yasmin"],
    },
    {
        "question": "post doc?",
        "must_contain": ["Leon", "Maria"],
    },
    {
        "question": "lab address",
        "must_contain": ["Alexandria", "3625"],
    },
    {
        "question": "tell about all partners",
        "must_contain": ["Azist", "MedStar", "INFRABEL"],
    },
    {
        "question": "director email",
        "must_contain": ["triantis@vt.edu", "j.godfrey@vt.edu"],
    },
    {
        "question": "What is production pressure?",
        "must_contain": ["do more with less", "safety"],
    },
    {
        "question": "Tell me about LEAP HI project",
        "must_contain": ["mental workload", "situational awareness", "NSF"],
    },
    {
        "question": "What kinds of data and methods does SPL use?",
        "must_contain": ["410,269", "DEA", "LSTM"],
    },
    {
        "question": "Do you work only in rail and transportation?",
        "must_contain": ["healthcare", "infrastructure"],
    },
]


def load_all_chunks():
    """Load all three chunk sources — same as app.py does at startup."""
    # 1. Scraped website chunks
    chunks = load_chunks("data/spl_chunks.json")
    print(f"  Scraped chunks:  {len(chunks)}")

    # 2. Curated hand-written chunks
    curated_path = "data/curated_chunks.json"
    if os.path.exists(curated_path):
        with open(curated_path, encoding="utf-8") as f:
            curated = json.load(f)
        chunks = chunks + curated
        print(f"  Curated chunks:  {len(curated)}")
    else:
        print("  Curated chunks:  NOT FOUND — curated_chunks.json missing!")

    # 3. Auto QA chunks
    auto_qa_path = "data/auto_qa_chunks.json"
    if os.path.exists(auto_qa_path):
        with open(auto_qa_path, encoding="utf-8") as f:
            auto_qa = json.load(f)
        chunks = chunks + auto_qa
        print(f"  Auto QA chunks:  {len(auto_qa)}")
    else:
        print("  Auto QA chunks:  not found — skipping (workflow not yet run)")

    print(f"  Total:           {len(chunks)} chunks")
    return chunks


def run_eval():
    print("Loading all chunk sources...")
    chunks = load_all_chunks()

    print("\nBuilding vector store for eval...")
    collection = build_vector_store(chunks)

    passed = 0
    failed = []

    print("\nRunning test cases...\n")
    for test in TEST_CASES:
        result = query(test["question"], collection)
        answer = result["answer"].lower()

        missing = [kw for kw in test["must_contain"]
                   if kw.lower() not in answer]

        if not missing:
            print(f"✅ PASS — {test['question']}")
            passed += 1
        else:
            print(f"❌ FAIL — {test['question']}")
            print(f"   Missing: {missing}")
            failed.append(test["question"])

    print(f"\n{'='*50}")
    print(f"Score: {passed}/{len(TEST_CASES)}")
    if failed:
        print("Failed questions:")
        for q in failed:
            print(f"  - {q}")
    else:
        print("All tests passed! ✓")


if __name__ == "__main__":
    run_eval()
