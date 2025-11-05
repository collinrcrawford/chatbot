# streamlit_rag.py
# Required packages:
# pip install streamlit openai numpy PyPDF2

import streamlit as st
import openai
import numpy as np
from typing import List, Tuple
from io import StringIO
from PyPDF2 import PdfReader
import math
import time

st.set_page_config(page_title="Streamlit RAG Chatbot", layout="wide")

# -----------------------
# Helper utilities
# -----------------------
EMBED_MODEL = "text-embedding-3-small"  # OpenAI embeddings model
CHAT_MODEL = "gpt-3.5-turbo"            # Chat completions model

def set_api_key(key: str):
    openai.api_key = key.strip()

def get_text_from_pdf(uploaded_file) -> str:
    try:
        reader = PdfReader(uploaded_file)
        text = []
        for page in reader.pages:
            text.append(page.extract_text() or "")
        return "\n".join(text)
    except Exception as e:
        st.error(f"Failed to read PDF: {e}")
        return ""

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    if not text:
        return []
    text = text.strip()
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
        if start < 0:
            start = 0
        if start >= length:
            break
    return chunks

def embed_texts(texts: List[str]) -> List[List[float]]:
    # OpenAI embeddings in batch
    if len(texts) == 0:
        return []
    try:
        # API supports batching via input list
        resp = openai.Embedding.create(model=EMBED_MODEL, input=texts)
        return [r["embedding"] for r in resp["data"]]
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return []

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def retrieve(query: str, embeddings: np.ndarray, docs: List[dict], top_k: int = 4) -> List[Tuple[dict, float]]:
    if embeddings.size == 0 or len(docs) == 0:
        return []
    q_emb = np.array(embed_texts([query])[0], dtype=np.float32)
    # compute similarities
    sims = []
    for i, e in enumerate(embeddings):
        sims.append((i, cosine_sim(q_emb, e)))
    sims.sort(key=lambda x: x[1], reverse=True)
    results = []
    for idx, score in sims[:top_k]:
        results.append((docs[idx], score))
    return results

def build_prompt(question: str, retrieved: List[Tuple[dict,float]]) -> str:
    header = "You are an expert assistant. Use the provided context to answer the user's question. If the answer is not contained in the context, say you don't know and try to provide helpful guidance without making up facts.\n\n"
    context_parts = []
    for i, (doc, score) in enumerate(retrieved, start=1):
        context_parts.append(f"---\nSource {i} (score={score:.4f})\n{doc['text']}\n---\n")
    context = "\n".join(context_parts)
    prompt = f"{header}Context:\n{context}\nUser question: {question}\n\nAnswer concisely and cite the source numbers where appropriate (e.g., [Source 1])."
    return prompt

def ask_openai_chat(prompt: str, temperature: float = 0.0) -> str:
    try:
        messages = [
            {"role": "system", "content": "You are a helpful, concise assistant that cites sources from provided context."},
            {"role": "user", "content": prompt}
        ]
        resp = openai.ChatCompletion.create(model=CHAT_MODEL, messages=messages, temperature=temperature, max_tokens=800)
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        st.error(f"OpenAI chat error: {e}")
        return ""

# -----------------------
# Streamlit UI & session state
# -----------------------
if "documents" not in st.session_state:
    st.session_state["documents"] = []  # list of dicts: {'id':..., 'text':..., 'source':...}
if "embeddings" not in st.session_state:
    st.session_state["embeddings"] = np.zeros((0,))  # numpy array of embeddings (n, dim)
if "dim" not in st.session_state:
    st.session_state["dim"] = None

st.title("📚 Streamlit RAG Chatbot — Single File")
st.markdown(
    "Upload documents (TXT or PDF) or paste text, click *Index documents*, then ask questions. "
    "Enter your OpenAI API key below. This app stores embeddings in-memory for this session only."
)

# --- OpenAI key input
with st.sidebar:
    st.header("Settings & API Key")
    key = st.text_input("OpenAI API key", type="password", placeholder="sk-...", help="Your OpenAI API key. Stored only in memory this session.")
    if key:
        set_api_key(key)
        st.success("OpenAI key set (in memory).")
    st.write("---")
    top_k = st.number_input("Retrieve top K chunks", min_value=1, max_value=10, value=4, step=1)
    chunk_size = st.number_input("Chunk size (characters)", min_value=200, max_value=4000, value=1000, step=100)
    overlap = st.number_input("Chunk overlap (characters)", min_value=0, max_value=1000, value=200, step=50)
    st.write("---")
    st.write("Notes:")
    st.caption("• This app uses OpenAI Embeddings + ChatCompletion (gpt-3.5-turbo).")
    st.caption("• Uploaded docs are indexed into memory (session). Re-run or reload to clear.")
    st.write("---")

# --- Document upload / paste
col1, col2 = st.columns([2,1])
with col1:
    st.subheader("1) Upload or paste documents to index")
    uploaded = st.file_uploader("Upload TXT or PDF files (you can upload multiple)", type=["txt","pdf"], accept_multiple_files=True)
    txt_area = st.text_area("Or paste plain text here (optional)", height=150, placeholder="Paste document text to index...")

    index_button = st.button("Index documents")

with col2:
    st.subheader("Indexed documents")
    st.write(f"Indexed chunks: {len(st.session_state.documents)}")
    if len(st.session_state.documents) > 0:
        # show small preview
        for i, d in enumerate(st.session_state.documents[:10], start=1):
            st.write(f"• [{i}] {d['source']} — {len(d['text'])} chars")
    clear = st.button("Clear indexed documents")
    if clear:
        st.session_state.documents = []
        st.session_state.embeddings = np.zeros((0,))
        st.session_state.dim = None
        st.success("Cleared indexed documents and embeddings (in-memory).")

# Indexing logic
if index_button:
    if not key:
        st.error("Please enter your OpenAI API key in the sidebar before indexing.")
    else:
        new_chunks = []
        # from uploaded files
        for f in uploaded or []:
            fname = f.name
            if fname.lower().endswith(".pdf"):
                text = get_text_from_pdf(f)
            else:
                try:
                    raw = f.read()
                    if isinstance(raw, bytes):
                        text = raw.decode("utf-8", errors="ignore")
                    else:
                        text = str(raw)
                except Exception as e:
                    st.warning(f"Can't read {fname}: {e}")
                    text = ""
            chunks = chunk_text(text, chunk_size=int(chunk_size), overlap=int(overlap))
            for idx, c in enumerate(chunks, start=1):
                new_chunks.append({"text": c, "source": f"{fname} [chunk {idx}]"})
        # from pasted text
        if txt_area and txt_area.strip():
            chunks = chunk_text(txt_area, chunk_size=int(chunk_size), overlap=int(overlap))
            for idx, c in enumerate(chunks, start=1):
                new_chunks.append({"text": c, "source": f"pasted_text [chunk {idx}]"})
        if not new_chunks:
            st.warning("No documents or text found to index.")
        else:
            with st.spinner("Computing embeddings for chunks..."):
                texts = [c["text"] for c in new_chunks]
                embs = embed_texts(texts)
                if not embs:
                    st.error("Failed to compute embeddings.")
                else:
                    embs_arr = np.array(embs, dtype=np.float32)
                    # append to existing store
                    if st.session_state["embeddings"].size == 0:
                        st.session_state.embeddings = embs_arr
                    else:
                        try:
                            st.session_state.embeddings = np.vstack([st.session_state.embeddings, embs_arr])
                        except Exception:
                            # if dims mismatch, re-create store (unlikely)
                            st.session_state.embeddings = embs_arr
                    st.session_state.documents.extend(new_chunks)
                    st.session_state.dim = st.session_state.embeddings.shape[1]
                    st.success(f"Indexed {len(new_chunks)} chunks. Total chunks: {len(st.session_state.documents)}")

# --- Q&A UI
st.write("---")
st.subheader("2) Ask questions")
question = st.text_input("Ask", placeholder="Type your question here...", key="question_input")
ask_btn = st.button("Get Answer")

if ask_btn:
    if not key:
        st.error("Please enter your OpenAI API key in the sidebar before asking.")
    elif not question or question.strip() == "":
        st.warning("Please enter a question.")
    else:
        # If no documents indexed, prompt user to index, but we can still answer general questions via the model without context.
        if len(st.session_state.documents) == 0:
            st.info("No indexed documents found — answering from the model without retrieval (no sources).")
            prompt = f"You are an assistant. Answer the question:\n\n{question}"
            with st.spinner("Contacting OpenAI..."):
                answer = ask_openai_chat(prompt)
            st.markdown("### Answer")
            st.write(answer)
            st.markdown("*(No sources — no documents were indexed.)*")
        else:
            with st.spinner("Retrieving relevant chunks and contacting OpenAI..."):
                retrieved = retrieve(question, st.session_state.embeddings, st.session_state.documents, top_k=int(top_k))
                if not retrieved:
                    st.error("Retrieval failed or no embeddings available.")
                else:
                    prompt = build_prompt(question, retrieved)
                    answer = ask_openai_chat(prompt)
                    st.markdown("### Answer")
                    st.write(answer)
                    st.markdown("### Retrieved sources")
                    for i, (doc, score) in enumerate(retrieved, start=1):
                        st.markdown(f"**Source {i} — {doc['source']} — similarity {score:.4f}**")
                        snippet = doc['text']
                        # show short snippet
                        display_snip = snippet if len(snippet) < 500 else snippet[:500] + "…"
                        st.code(display_snip, language=None)
                    st.success("Done.")

# Footer
st.write("---")
st.caption("This is a local, in-memory Streamlit RAG demo. For production: persist vectors, secure keys, rate limit, and add proper error handling.")
