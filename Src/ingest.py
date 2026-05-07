import os
import re
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

def clean_text(text):
    # Parse HTML and extract plain text
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(separator=" ")
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ── 1. Load and clean all filings ──────────────────────────────
print("Loading and cleaning documents...")

docs = []
base_path = "data/sec-edgar-filings"

for company in ["AAPL", "MSFT", "TSLA"]:
    company_path = os.path.join(base_path, company, "10-K")
    for filing_id in os.listdir(company_path):
        file_path = os.path.join(company_path, filing_id, "full-submission.txt")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                raw_text = f.read()
            cleaned = clean_text(raw_text)
            doc = Document(
                page_content=cleaned,
                metadata={"company": company, "filing_id": filing_id}
            )
            docs.append(doc)
            print(f"  Loaded {company} - {filing_id}")

print(f"\nTotal documents loaded: {len(docs)}")

# ── 2. Chunk ────────────────────────────────────────────────────
print("\nChunking documents...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

chunks = splitter.split_documents(docs)
print(f"Total chunks created: {len(chunks)}")

# ── 3. Embed and store ─────────────────────────────────────────
print("\nEmbedding and storing in ChromaDB...")
print("(This may take a while — deleting old DB first)")

import shutil
if os.path.exists("data/chroma_db"):
    shutil.rmtree("data/chroma_db")
    print("Old ChromaDB deleted.")

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="data/chroma_db"
)

print(f"\nDone! {len(chunks)} chunks stored in ChromaDB")
print("Vector database saved to data/chroma_db")