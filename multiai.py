import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import tempfile
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

from anthropic import Anthropic
from openai import OpenAI


# ---------------- TESSERACT FIX ----------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ---------------- ENV ----------------
load_dotenv()


# ---------------- SAFE API INIT ----------------

anthropic_key = os.getenv("ANTHROPIC_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")

anthropic_available = False
openai_available = False

if anthropic_key:
    anthropic_client = Anthropic(api_key=anthropic_key)
    anthropic_available = True

if openai_key:
    openai_client = OpenAI(api_key=openai_key)
    openai_available = True

if "groq" not in st.session_state:
    st.session_state.groq = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )


# ---------------- MULTI LLM ----------------

def multi_llm(prompt):

    # ---------------- CLAUDE (1st Priority) ----------------
    if anthropic_available:
        try:
            st.info("🔵 Trying Claude...")

            response = anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=800,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            result = response.content[0].text

            if result.strip():
                st.session_state["model_used"] = "Claude"
                return result

        except Exception as e:
            st.warning(f"⚠️ Claude failed: {str(e)}")

    # ---------------- CHATGPT (2nd) ----------------
    if openai_available:
        try:
            st.info("🟢 Trying ChatGPT...")

            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            result = response.choices[0].message.content

            if result and result.strip():
                st.session_state["model_used"] = "ChatGPT"
                return result

        except Exception as e:
            st.warning(f"⚠️ ChatGPT failed: {str(e)}")

    # ---------------- GROQ (3rd) ----------------
    try:
        st.info("🟠 Using Groq...")

        response = st.session_state.groq.invoke(prompt)

        if response and response.content.strip():
            st.session_state["model_used"] = "Groq"
            return response.content

    except Exception as e:
        st.warning(f"⚠️ Groq failed: {str(e)}")

    # ---------------- FINAL ----------------
    st.session_state["model_used"] = "None"
    return "❌ All AI APIs failed. Check API keys / quota."

# ---------------- SESSION ----------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "reranker" not in st.session_state:
    st.session_state.reranker = None


# ---------------- FUNCTIONS ----------------

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
        cv2.THRESH_BINARY, 11, 2
    )
    return Image.fromarray(thresh)


def process_pdf(uploaded_file):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    docs = []

    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):

            text = page.extract_text()

            if not text or len(text.strip()) < 20:
                try:
                    img = page.to_image(resolution=300).original
                    img = preprocess_image(img)
                    text = pytesseract.image_to_string(img)
                except Exception as e:
                    print(f"OCR failed on page {i+1}: {e}")
                    text = ""

            text = clean_text(text)

            # ✅ RELAXED FILTER
            if text and len(text.strip()) > 10:
                docs.append(Document(page_content=text, metadata={"page": i+1}))

    st.write(f"📄 Documents extracted: {len(docs)}")

    if not docs:
        st.error("❌ No text extracted from PDF. Try another file.")
        return None, None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=300
    )

    chunks = splitter.split_documents(docs)

    st.write(f"✂️ Chunks created: {len(chunks)}")

    if not chunks:
        st.error("❌ Chunking failed. No usable content.")
        return None, None

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5"
    )

    vectordb = Chroma.from_documents(chunks, embeddings)

    vector_retriever = vectordb.as_retriever(search_kwargs={"k": 40})

    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 25

    retriever = EnsembleRetriever(
        retrievers=[bm25, vector_retriever],
        weights=[0.5, 0.5]
    )

    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    return retriever, reranker


def rerank(query, docs, reranker, top_k=8):
    pairs = [[query, d.page_content] for d in docs]
    scores = reranker.predict(pairs)
    scored = list(zip(docs, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in scored[:top_k]]


# ---------------- UI ----------------

st.title("⚖️ Legal RAG System")

file = st.file_uploader("Upload PDF", type="pdf")

if file:

    if st.session_state.retriever is None:
        with st.spinner("Processing PDF..."):
            r, rr = process_pdf(file)

            if r is None:
                st.stop()

            st.session_state.retriever = r
            st.session_state.reranker = rr

        st.success("✅ PDF processed")

    q = st.chat_input("Ask your question...")

    if q:

        lang = detect_language(q)

        if lang == "te":
            q = multi_llm(f"Translate Telugu to English:\n{q}")
        elif lang == "hi":
            q = multi_llm(f"Translate Hindi to English:\n{q}")

        queries_text = multi_llm(f"Generate 5 queries:\n{q}")
        queries = [x.strip() for x in queries_text.split("\n") if x.strip()]
        queries.append(q)

        all_docs = []
        for query in queries:
            all_docs.extend(st.session_state.retriever.invoke(query))

        docs = list({d.page_content: d for d in all_docs}.values())
        docs = rerank(q, docs, st.session_state.reranker)

        if docs:
            context = ""
            for d in docs:
                context += f"\n--- PAGE {d.metadata.get('page')} ---\n{d.page_content}\n"

            prompt = f"""
Use ONLY the context. Quote law. Show page.

Context:
{context}

Question:
{q}

Answer:
"""
            ans = multi_llm(prompt)
        else:
            ans = "Not enough information"

        st.session_state.chat_history.append(("user", q))
        st.session_state.chat_history.append(("bot", ans))

    for role, msg in st.session_state.chat_history:
        st.chat_message("user" if role=="user" else "assistant").write(msg)

    if "model_used" in st.session_state:
        st.info(f"🤖 Model Used: {st.session_state['model_used']}")