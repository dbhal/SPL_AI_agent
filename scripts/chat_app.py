"""
SPL Chatbot — Chainlit Chat Interface
"""

import asyncio
import groq
import chainlit as cl
import sys
import os

sys.path.append(os.path.dirname(__file__))
from rag_pipeline import load_chunks, build_vector_store, retrieve, build_prompt

print("Loading SPL knowledge base...")
BASE = os.path.dirname(os.path.dirname(__file__))
chunks = load_chunks(os.path.join(BASE, "data", "spl_chunks.json"))
collection = build_vector_store(chunks)
print("Ready!")


@cl.on_chat_start
async def start():
    await cl.Message(
        content=(
            "👋 Hi! I'm the **SPL Assistant** at Virginia Tech.\n\n"
            "I can help you with:\n"
            "- 🔬 SPL's research areas and projects\n"
            "- 👥 Meet the SPL team\n"
            "- 🤝 Partnership and collaboration opportunities\n"
            "- 📚 Publications and education programs\n"
            "- 📅 News and upcoming events\n\n"
            "What would you like to know?"
        )
    ).send()


@cl.on_message
async def handle_message(message: cl.Message):
    loop = asyncio.get_event_loop()
    context_chunks = await loop.run_in_executor(
        None, retrieve, message.content, collection
    )
    prompt = build_prompt(message.content, context_chunks)

    msg = cl.Message(content="")
    await msg.send()

    client = groq.Groq(api_key=os.environ["GROQ_API_KEY"])
    stream = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=512,
        stream=True,
    )

    for chunk in stream:
        token = chunk.choices[0].delta.content or ""
        if token:
            await msg.stream_token(token)

    source_links = "\n".join(
        f"- [{c['source_url']}]({c['source_url']})"
        for c in context_chunks
    )
    if source_links:
        await msg.stream_token(f"\n\n**Sources:**\n{source_links}")

    await msg.update()