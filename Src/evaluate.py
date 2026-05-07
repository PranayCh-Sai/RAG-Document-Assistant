import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from retriever import retrieve_chunks
from generator import ask_question

load_dotenv()

# ── Evaluation questions with ground truth answers ─────────────
eval_questions = [
    {
        "question": "What was Apple's total revenue in 2023?",
        "ground_truth": "Apple's total net sales in fiscal year 2023 were $383,285 million."
    },
    {
        "question": "What are Microsoft's three main business segments?",
        "ground_truth": "Microsoft's three main business segments are Productivity and Business Processes, Intelligent Cloud, and More Personal Computing."
    },
    {
        "question": "What was Tesla's automotive revenue in 2024?",
        "ground_truth": "Tesla's automotive sales revenue in 2024 was $72,480 million."
    },
    {
        "question": "What products does Apple's Mac segment include?",
        "ground_truth": "Apple's Mac segment includes desktop and laptop computers such as MacBook Air, MacBook Pro, iMac, Mac mini, Mac Studio and Mac Pro."
    },
    {
        "question": "What is Microsoft's Intelligent Cloud segment?",
        "ground_truth": "Microsoft's Intelligent Cloud segment includes server products and cloud services such as Azure, SQL Server, Windows Server, and GitHub."
    },
]

# ── Build evaluation dataset ───────────────────────────────────
print("Building evaluation dataset...")

questions, answers, contexts, ground_truths = [], [], [], []

for item in eval_questions:
    q = item["question"]
    print(f"  Processing: {q[:50]}...")

    chunks = retrieve_chunks(q, k=5)
    ctx = [chunk.page_content for chunk in chunks]
    ans = ask_question(q)

    questions.append(q)
    answers.append(ans)
    contexts.append(ctx)
    ground_truths.append(item["ground_truth"])

dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths,
})

# ── Run RAGAS evaluation ───────────────────────────────────────
print("\nRunning RAGAS evaluation...")

llm = LangchainLLMWrapper(ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=2048))
emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=llm,
    embeddings=emb,
)

# ── Print scorecard ────────────────────────────────────────────
print("\n" + "="*50)
print("RAGAS EVALUATION SCORECARD")
print("="*50)
df = results.to_pandas()
print(f"Faithfulness:      {df['faithfulness'].mean():.3f}")
print(f"Answer Relevancy:  {df['answer_relevancy'].mean():.3f}")
print(f"Context Precision: {df['context_precision'].mean():.3f}")
print(f"Context Recall:    {df['context_recall'].mean():.3f}")
print("="*50)
print("\nAll scores are 0–1. Higher is better.")