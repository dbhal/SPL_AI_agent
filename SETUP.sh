# SPL AI Chatbot — Setup Guide
# ==============================
# Run these commands step by step. Each phase builds on the last.

# ── Phase 1: Install dependencies ────────────────────────────────────────────

pip install requests beautifulsoup4           # for scraping
pip install chromadb sentence-transformers    # for RAG / vector store
pip install chainlit                          # for chat UI
pip install transformers peft bitsandbytes datasets accelerate trl  # for fine-tuning on GPU


# ── Phase 2: Scrape the SPL website ──────────────────────────────────────────
# This creates data/spl_chunks.json

cd spl_chatbot
python scraper/scrape_spl.py


# ── Phase 3: Test the RAG pipeline locally ────────────────────────────────────
# Option A: Install Ollama first (easiest)
#   Mac/Linux: curl -fsSL https://ollama.ai/install.sh | sh
#   Then:      ollama pull llama3
# Then run:

python rag/rag_pipeline.py


# ── Phase 4: Launch the chat UI ──────────────────────────────────────────────

chainlit run ui/chat_app.py
# Open http://localhost:8000 in your browser


# ── Phase 5: Fine-tune on VT ARC (when ready) ────────────────────────────────
# 1. Request VT ARC access: https://arc.vt.edu
# 2. SSH into TinkerCliffs: ssh <your_pid>@tinkercliffs1.arc.vt.edu
# 3. Upload your project:   scp -r spl_chatbot/ <pid>@tinkercliffs1.arc.vt.edu:~/
# 4. Create a Python venv:
#       module load Python/3.11
#       python -m venv ~/venv/spl_llm
#       source ~/venv/spl_llm/bin/activate
#       pip install transformers peft bitsandbytes datasets accelerate trl
# 5. Submit the job:
#       cd ~/spl_chatbot
#       mkdir -p logs
#       sbatch rag/submit_finetune.sh
# 6. Check job status:
#       squeue -u <your_pid>
#       cat logs/finetune_<job_id>.out
