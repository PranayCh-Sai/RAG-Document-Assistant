import os
from dotenv import load_dotenv
from anthropic import Anthropic
from retriever import retrieve_chunks

load_dotenv()

client = Anthropic()

def ask_question(question, k=5):
    # Step 1: Retrieve relevant chunks
    print(f"\nSearching for relevant chunks...")
    chunks = retrieve_chunks(question, k=k)

    # Step 2: Build context string from chunks
    context_parts = []
    for i, chunk in enumerate(chunks):
        company = chunk.metadata.get("company", "Unknown")
        filing_id = chunk.metadata.get("filing_id", "Unknown")
        context_parts.append(
            f"[Source {i+1} | Company: {company} | Filing: {filing_id}]\n{chunk.page_content}"
        )

    context = "\n\n---\n\n".join(context_parts)

    # Step 3: Build the prompt
    system_prompt = """You are a financial analyst assistant. 
You answer questions about company SEC 10-K filings strictly based on the provided context.
Rules:
- Always cite your sources using [Source N] notation
- If the context doesn't contain enough information, say so clearly
- Never make up numbers or facts
- Be concise and professional"""

    user_prompt = f"""Context from SEC filings:

{context}

Question: {question}

Answer based only on the context above, citing sources."""

    # Step 4: Call Claude
    print("Asking Claude...\n")
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )

    return response.content[0].text

if __name__ == "__main__":
    questions = [
        "What was Apple's total revenue in 2023?",
        "How many vehicles did Tesla deliver?",
        "What are Microsoft's main business segments?"
    ]

    for question in questions:
        print(f"\n{'='*60}")
        print(f"Q: {question}")
        print('='*60)
        answer = ask_question(question)
        print(answer)