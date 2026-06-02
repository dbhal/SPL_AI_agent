"""
Run this after every change to check chatbot quality automatically.
"""
import json
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
]

def run_eval():
    chunks = load_chunks("data/spl_chunks.json")
    collection = build_vector_store(chunks)

    passed = 0
    failed = []

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
        print(f"Failed questions:")
        for q in failed:
            print(f"  - {q}")

if __name__ == "__main__":
    run_eval()