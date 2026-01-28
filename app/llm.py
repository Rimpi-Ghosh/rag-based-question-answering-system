import os
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

def generate_answer(chunks, question: str):
    """
    Generate answer using Groq LLM with retrieved context.
    Tracks latency as a metric.
    """

    # Create client INSIDE function (fixes reload issue)
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    context = "\n".join(chunks)

    prompt = f"""
You are an AI assistant.
Answer the question ONLY using the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
"""

    start_time = time.time()

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    latency = round(time.time() - start_time, 3)

    answer = response.choices[0].message.content.strip()

    return f"{answer}\n\n[Latency: {latency} seconds]"
