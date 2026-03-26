import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import shutil
import re
from dotenv import load_dotenv

import pdfplumber
import pytesseract
import cv2
import numpy as np
from PIL import Image

from langdetect import detect
from sentence_transformers import CrossEncoder

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever

from langchain.schema import Document
from langchain_groq import ChatGroq

# Multi LLM APIs
from anthropic import Anthropic
from openai import OpenAI
from transformers import pipeline


# ---------------- CONFIG ----------------

PDF_PATH = r"C:\Users\ASUS\Desktop\rag\state law.pdf"
CHROMA_PATH = "chroma_db"

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ---------------- ENV ----------------

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    print("❌ GROQ_API_KEY missing")
    sys.exit()


# ---------------- LOAD MODELS ----------------

anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

groq_llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

hf_model = pipeline(
    "text-generation",
    model="google/flan-t5-base",
    max_length=512
)


# ---------------- MULTI LLM ----------------

def multi_llm(prompt):

    # Claude
    try:
        res = anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}]
        )
        print("🤖 Model: Claude")
        return res.content[0].text
    except:
        pass

    # ChatGPT
    try:
        res = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        print("🤖 Model: ChatGPT")
        return res.choices[0].message.content
    except:
        pass

    # Groq
    try:
        res = groq_llm.invoke(prompt)
        print("🤖 Model: Groq")
        return res.content
    except:
        pass

    # HuggingFace
    print("⚠️ Using HuggingFace fallback")
    return hf_model(prompt)[0]["generated_text"]


# ---------------- UTIL ----------------

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"


def clean_text(text):
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def preprocess_image(img):
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )
    return Image.fromarray(thresh)


# ---------------- LOAD PDF ----------------

print("📄 Loading PDF...")

documents = []

with pdfplumber.open(PDF_PATH) as pdf:
    for i, page in enumerate(pdf.pages):

        text = page.extract_text()

        if not text or len(text.strip()) < 50:
            try:
                img = page.to_image(resolution=300).original
                img = preprocess_image(img)
                text = pytesseract.image_to_string(img)
            except:
                text = ""

        text = clean_text(text)

        if len(text) > 50:
            documents.append(
                Document(
                    page_content=text,
                    metadata={"page": i + 1}
                )
            )

print("✅ Pages:", len(documents))


# ---------------- CHUNK ----------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=300
)

chunks = splitter.split_documents(documents)

print("Chunks:", len(chunks))


# ---------------- RETRIEVAL ----------------

if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"
)

vectordb = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory=CHROMA_PATH
)

vector_retriever = vectordb.as_retriever(search_kwargs={"k": 40})

bm25 = BM25Retriever.from_documents(chunks)
bm25.k = 25

retriever = EnsembleRetriever(
    retrievers=[bm25, vector_retriever],
    weights=[0.5, 0.5]
)


# ---------------- RERANK ----------------

print("⚖️ Loading reranker...")

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def rerank(query, docs, top_k=8):
    pairs = [[query, d.page_content] for d in docs]
    scores = reranker.predict(pairs)
    scored = list(zip(docs, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in scored[:top_k]]


# ---------------- CHAT LOOP ----------------

print("\n🚀 LEGAL RAG READY\n")

while True:

    question = input("\nQuestion: ")

    if question.lower() == "exit":
        break

    lang = detect_language(question)

    if lang == "te":
        question = multi_llm(f"Translate Telugu to English:\n{question}")

    elif lang == "hi":
        question = multi_llm(f"Translate Hindi to English:\n{question}")

    print("🔎 Query:", question)

    queries_text = multi_llm(f"Generate 5 search queries:\n{question}")
    queries = [q.strip() for q in queries_text.split("\n") if q.strip()]
    queries.append(question)

    all_docs = []
    for q in queries:
        all_docs.extend(retriever.invoke(q))

    docs = list({d.page_content: d for d in all_docs}.values())
    docs = rerank(question, docs)

    if not docs:
        print("❌ Not enough information")
        continue

    context = ""
    for d in docs:
        context += f"\n--- PAGE {d.metadata.get('page')} ---\n{d.page_content}\n"

    prompt = f"""
Use ONLY the context. Quote law. Show page.

Context:
{context}

Question:
{question}

Answer:
"""

    answer = multi_llm(prompt)

    print("\n📜 Answer:\n", answer)

    print("\n1 Telugu\n2 Hindi\n3 Skip")
    choice = input("Choice: ")

    if choice == "1":
        print("\n📜 Telugu:\n",
              multi_llm(f"Translate to Telugu:\n{answer}"))

    elif choice == "2":
        print("\n📜 Hindi:\n",
              multi_llm(f"Translate to Hindi:\n{answer}"))

    print("\n" + "-"*50)