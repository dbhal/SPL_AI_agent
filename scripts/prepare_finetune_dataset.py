"""
Converts auto_qa_dataset.json + manual_chunks.json into Llama 3 chat format
ready for fine-tuning on Kaggle.
"""
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def prepare():
    # Load auto-generated Q&A pairs
    with open(os.path.join(BASE, "data", "auto_qa_dataset.json"), encoding="utf-8") as f:
        qa_pairs = json.load(f)

    # Load manual chunks
    with open(os.path.join(BASE, "data", "manual_chunks.json"), encoding="utf-8") as f:
        manual = json.load(f)

    formatted = []

    # ── 1. Convert manual chunks into training examples ───────────────────────
    for chunk in manual:
        text = chunk.get("text", "").strip()
        if len(text.split()) > 20:  # skip tiny chunks
            # Split chunks that start with a question (Q&A format)
            if "?" in text[:100]:
                parts = text.split("?", 1)
                if len(parts) == 2:
                    q = parts[0].strip() + "?"
                    a = parts[1].strip()
                    if len(a) > 20:
                        formatted.append({
                            "instruction": q,
                            "response": a,
                            "source": chunk.get("source_url", "https://www.spl.ise.vt.edu")
                        })
            else:
                # Use chunk as factual context
                formatted.append({
                    "instruction": "Tell me about SPL based on this information.",
                    "response": text,
                    "source": chunk.get("source_url", "https://www.spl.ise.vt.edu")
                })

    # ── 2. Format auto-generated Q&A pairs ───────────────────────────────────
    for pair in qa_pairs:
        q = pair.get("question", "").strip()
        a = pair.get("answer", "").strip()
        if q and a and len(a) > 20:
            formatted.append({
                "instruction": q,
                "response": a,
                "source": pair.get("source_url", "https://www.spl.ise.vt.edu")
            })

    # ── 3. Add high quality handcrafted Q&A pairs ────────────────────────────
    manual_qa = [
        {
            "instruction": "Who are the current SPL team members?",
            "response": "SPL current team members:\n\n**Faculty:**\n- Dr. Konstantinos P. Triantis (Director, triantis@vt.edu)\n- Dr. Joseph Godfrey (Co-Director, j.godfrey@vt.edu)\n\n**Postdoctoral Associates:**\n- Leon Sobrie (leonsobrie@vt.edu)\n- Maria Tomai (tomai@vt.edu)\n\n**PhD Students:**\n- Tatiana Daychman (tvdgrad1@vt.edu)\n- Yasmin Mashayekhy Fard (yas@vt.edu)\n- Juan Carlos Franco R. (jcfranco@vt.edu)\n- Ibrahim Oguz Cetinkaya (ibrahimoguz@vt.edu)",
            "source": "https://www.spl.ise.vt.edu/about/people.html"
        },
        {
            "instruction": "What is the SPL lab address?",
            "response": "The System Performance Laboratory is located at **3625 Potomac Ave, Room 5732, Alexandria, VA 22305**. This is the Virginia Tech Innovation Campus in Alexandria, Virginia, near the Potomac Yards Metro station.",
            "source": "https://www.spl.ise.vt.edu/about.html"
        },
        {
            "instruction": "Tell me about all SPL partners",
            "response": "SPL has the following partners:\n\n1. **MedStar Health** — exploring human-technology interaction in healthcare operating rooms\n2. **Carilion Clinic/VT Carilion School of Medicine** — AI and system simulation in healthcare\n3. **INFRABEL** (Belgian National Railway) — sharing operational data since 2019 for railway research\n4. **Azist Inc.** — developing a technology innovation incubator\n5. **Urban Alliance** and **Chrysalis Development Group** — community outreach partners",
            "source": "https://www.spl.ise.vt.edu/partnership.html"
        },
        {
            "instruction": "How many PhD students does SPL have?",
            "response": "SPL currently has **4 PhD students**:\n- Tatiana Daychman\n- Yasmin Mashayekhy Fard\n- Juan Carlos Franco R.\n- Ibrahim Oguz Cetinkaya",
            "source": "https://www.spl.ise.vt.edu/about/people.html"
        },
        {
            "instruction": "How many postdoctoral associates does SPL have?",
            "response": "SPL currently has **2 postdoctoral associates**:\n- **Leon Sobrie** — researches Business Analytics, Machine Learning, Decision Support\n- **Maria Tomai** — researches Socio-technical systems, System Dynamics, Sustainability Transitions",
            "source": "https://www.spl.ise.vt.edu/about/people.html"
        },
        {
            "instruction": "What is the SPL research funding?",
            "response": "SPL has participated in more than **$6.6 million** in funded research projects. The current NSF LEAP-HI grant is **$2 million** funded for four years.",
            "source": "https://www.ise.vt.edu/people/faculty/triantis.html"
        },
        {
            "instruction": "What is the director email of SPL?",
            "response": "The SPL directors can be contacted at:\n- **Dr. Konstantinos P. Triantis** (Director): triantis@vt.edu\n- **Dr. Joseph Godfrey** (Co-Director): j.godfrey@vt.edu",
            "source": "https://www.spl.ise.vt.edu/about/people.html"
        },
        {
            "instruction": "What research areas does SPL focus on?",
            "response": "SPL focuses on the following research areas:\n\n1. Enterprise Performance Measurement\n2. **Data Envelopment Analysis (DEA)**\n3. Fuzzy Sets and Logic\n4. **System Dynamics Modeling**\n5. Design of Quality Management Systems\n6. Engineering Administration\n7. Systems Engineering Management\n8. Influential Observation Identification\n9. Process Definition and Re-engineering\n10. **Infrastructure Systems** (Resilience and Sustainability)\n11. Evaluation and Assessment of R&D Enterprises\n12. **Human-Automation Interaction**",
            "source": "https://www.spl.ise.vt.edu/about.html"
        },
        {
            "instruction": "How can I join SPL as a student?",
            "response": "To join SPL as a student you can:\n\n1. **Apply to VT ISE graduate program** and work with SPL faculty\n2. **Volunteer** — fill out the form at https://forms.gle/SHCK9ce6t63zKDrA8\n3. **Contact Dr. Godfrey** directly at j.godfrey@vt.edu\n4. **Attend SPL events** like Teen Science Cafés and NAPW workshops\n\nVisit https://www.spl.ise.vt.edu for more information.",
            "source": "https://www.spl.ise.vt.edu/about/people.html"
        },
        {
            "instruction": "What is the NSF funded project at SPL?",
            "response": "The NSF-funded project is called **LEAP-HI** (Leading Engineering for America's Prosperity, Health, and Infrastructure). It is a **$2 million grant** funded for four years, focusing on human-automation interaction in safety-critical systems.",
            "source": "https://www.spl.ise.vt.edu/research/funded/active/reports.html"
        },
    ]
    formatted.extend(manual_qa)

    # ── 4. Save in two formats ────────────────────────────────────────────────
    output_path = os.path.join(BASE, "data", "finetune_dataset_final.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(formatted, f, indent=2, ensure_ascii=False)

    jsonl_path = os.path.join(BASE, "data", "finetune_dataset_final.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in formatted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Total training examples: {len(formatted)}")
    print(f"  - From manual chunks:  {len([x for x in formatted if x in formatted[:len(manual)]])}")
    print(f"  - From auto Q&A:       {len(qa_pairs)}")
    print(f"  - Handcrafted Q&A:     {len(manual_qa)}")
    print(f"Saved to: {output_path}")
    print(f"Saved to: {jsonl_path}")
    return formatted


if __name__ == "__main__":
    prepare()