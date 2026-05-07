import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings 

load_dotenv()

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory="data/chroma_db",
        embedding_function=embeddings
    )
    return vectorstore

def retrieve_chunks(question, k=5):
    vectorstore = load_vectorstore()
    results = vectorstore.similarity_search(question, k=k)
    return results

if __name__ == "__main__":
    question = "What was Apple's total revenue?"
    print(f"Question: {question}\n")
    chunks = retrieve_chunks(question)
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i+1} ({chunk.metadata.get('company')}) ---")
        print(chunk.page_content[:300])
        print()