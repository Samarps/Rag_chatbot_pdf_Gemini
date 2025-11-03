import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai
from config import GEMINI_API_KEY
import pdfplumber, os

# --- Load FAISS index and metadata ---
VECTOR_DIR = "vectorstore"
INDEX_FILE = f"{VECTOR_DIR}/index.faiss"
META_FILE = f"{VECTOR_DIR}/metadata.pkl"

@st.cache_resource
def load_resources():
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "rb") as f:
        metadata = pickle.load(f)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    client = genai.Client(api_key=GEMINI_API_KEY)
    return index, metadata, embedder, client

index, metadata, embedder, client = load_resources()
model_name = "models/gemini-2.5-flash"

# --- Helper to re-read PDF pages ---
@st.cache_resource
def load_pdfs():
    data_dir = "data"
    docs = {}
    for file in os.listdir(data_dir):
        if file.endswith(".pdf"):
            docs[file] = []
            with pdfplumber.open(os.path.join(data_dir, file)) as pdf:
                for page in pdf.pages:
                    docs[file].append(page.extract_text() or "")
    return docs

docs = load_pdfs()

# --- RAG function ---
def retrieve_and_answer(query, top_k=5):
    query_emb = embedder.encode([query])
    distances, indices = index.search(query_emb, top_k)

    retrieved_texts = []
    for idx in indices[0]:
        meta = metadata[idx]
        src = meta["source"]
        page = meta["page"]
        text = docs[src][page - 1]
        retrieved_texts.append(f"From {src} (page {page}):\n{text}")

    context = "\n\n".join(retrieved_texts[:3])
    prompt = f"""
You are a helpful research assistant.
Use the following context to answer the question.

Context:
{context}

Question: {query}

Answer concisely and in an academic tone.
"""

    response = client.models.generate_content(model=model_name, contents=prompt)
    return response.text

# --- Streamlit UI ---
st.set_page_config(page_title="RAGBot - Gemini", layout="wide")
st.title("RAG Research Chatbot (Gemini)")

user_query = st.text_input("Ask a question about your research papers:")

if st.button("Search") and user_query.strip():
    with st.spinner("Retrieving information..."):
        answer = retrieve_and_answer(user_query)
    st.markdown("### Answer:")
    st.write(answer)
