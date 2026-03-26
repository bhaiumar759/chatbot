from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel
import google.generativeai as genai
import os, shutil, logging, asyncio, json, pickle
from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi  
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Improved Gemini RAG Agent")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
VECTOR_DB_PATH = "vectorstore"
BM25_PATH = os.path.join(VECTOR_DB_PATH, "bm25.pkl")
PARENT_CHUNKS_PATH = os.path.join(VECTOR_DB_PATH, "parent_chunks.pkl")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_PATH, exist_ok=True)

GEMINI_API_KEY = "AIzaSyBLcZrR0KVR1C7MRdp-R1_mAmRZh4BoeBM"   
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set.")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-3-flash-preview")  # ✅ FIX 2: Valid model name

CHAT_MEMORY: Dict[str, List[Dict[str, str]]] = {}

# -----------------------------------------------
# Gemini Embeddings (unchanged, works well)
# -----------------------------------------------
class GeminiEmbedding(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        result = genai.embed_content(model="gemini-embedding-001", content=text)
        return result["embedding"]

embedding = GeminiEmbedding()

# -----------------------------------------------
# ✅ FIX 3: Parent-Child Chunk Store
# Small child chunks → embedded for precision retrieval
# Large parent chunks → fed to LLM for richer context
# -----------------------------------------------
parent_chunks: Dict[str, str] = {}   # child_id → parent text
bm25_index: Optional[object] = None
bm25_corpus: List[str] = []
vectorstore: Optional[FAISS] = None


def save_bm25():
    with open(BM25_PATH, "wb") as f:
        pickle.dump((bm25_index, bm25_corpus), f)

def load_bm25():
    global bm25_index, bm25_corpus
    if os.path.exists(BM25_PATH):
        with open(BM25_PATH, "rb") as f:
            bm25_index, bm25_corpus = pickle.load(f)

def save_parent_chunks():
    with open(PARENT_CHUNKS_PATH, "wb") as f:
        pickle.dump(parent_chunks, f)

def load_parent_chunks():
    global parent_chunks
    if os.path.exists(PARENT_CHUNKS_PATH):
        with open(PARENT_CHUNKS_PATH, "rb") as f:
            parent_chunks = pickle.load(f)

def load_vectorstore():
    global vectorstore
    index_path = os.path.join(VECTOR_DB_PATH, "index.faiss")
    if os.path.exists(index_path):
        vectorstore = FAISS.load_local(
            VECTOR_DB_PATH, embedding, allow_dangerous_deserialization=True
        )

load_vectorstore()
load_bm25()
load_parent_chunks()


# -----------------------------------------------
# ✅ FIX 4: HyDE — Hypothetical Document Embeddings
# Generate a "fake ideal answer" and embed THAT
# instead of the raw user question. Retrieves far
# more relevant chunks because the vector space
# better matches answer-shaped text.
# -----------------------------------------------
def generate_hypothetical_answer(query: str) -> str:
    try:
        hyde_prompt = (
            f"Write a concise, factual paragraph that directly answers: '{query}'. "
            "Use specific details. This will be used for document retrieval only."
        )
        resp = model.generate_content(hyde_prompt)
        return resp.text.strip()
    except Exception as e:
        logger.warning(f"HyDE failed, falling back to raw query: {e}")
        return query


# -----------------------------------------------
# ✅ FIX 5: Hybrid Retrieval (Dense + BM25 Sparse)
# BM25 catches exact keyword/acronym matches that
# semantic search misses. RRF fuses both rankings.
# -----------------------------------------------
def reciprocal_rank_fusion(
    dense_ids: List[str],
    sparse_ids: List[str],
    k: int = 60
) -> List[str]:
    scores: Dict[str, float] = {}
    for rank, doc_id in enumerate(dense_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    for rank, doc_id in enumerate(sparse_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


def hybrid_search(query: str, top_k: int = 8) -> List[str]:
    """Returns list of parent chunk texts ranked by RRF score."""
    if vectorstore is None:
        return []

    # 1. HyDE: embed hypothetical answer instead of raw query
    hyde_text = generate_hypothetical_answer(query)

    # 2. Dense retrieval (semantic)
    dense_results = vectorstore.similarity_search(hyde_text, k=top_k * 2)
    dense_ids = [d.metadata.get("child_id", d.page_content[:40]) for d in dense_results]

    # 3. Sparse retrieval (BM25 keyword)
    sparse_ids: List[str] = []
    if bm25_index and bm25_corpus:
        tokenized = query.lower().split()
        bm25_scores = bm25_index.get_scores(tokenized)
        top_indices = np.argsort(bm25_scores)[::-1][:top_k * 2]
        sparse_ids = [bm25_corpus[i]["id"] for i in top_indices]

    # 4. Fuse rankings
    fused_ids = reciprocal_rank_fusion(dense_ids, sparse_ids)[:top_k]

    # 5. ✅ FIX 6: Return PARENT chunks (not child chunks)
    # Child was retrieved for precision; parent gives LLM full context
    seen, contexts = set(), []
    for doc_id in fused_ids:
        text = parent_chunks.get(doc_id, "")
        if text and text not in seen:
            seen.add(text)
            contexts.append(text)

    return contexts[:top_k]


# -----------------------------------------------
# Request model
# -----------------------------------------------
class AskRequest(BaseModel):
    query: str


# -----------------------------------------------
# ✅ FIX 7: Structured prompt with citation guidance
# Explicit instructions reduce hallucination and
# force grounding. Few-shot style "RULES" help
# the model follow the no-hallucination constraint.
# -----------------------------------------------
SYSTEM_PROMPT = """You are a precise document assistant. Your job is to answer questions
based ONLY on the provided document context.

STRICT RULES:
1. Answer ONLY from the CONTEXT section below. Never use outside knowledge.
2. If context is insufficient, say exactly: "I can not answer this question based on the provided context."
3. For numerical data, dates, or names: quote directly from context.
4. Keep answers focused — no padding or generic introductions.
5. If asked to summarize, cover all key points found in the context.
"""

def build_prompt(context_chunks: List[str], history: str, query: str) -> str:
    context = "\n\n---\n\n".join(
        f"[Chunk {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
    )
    return f"""{SYSTEM_PROMPT}

CONTEXT (from uploaded document):
{context}

CONVERSATION HISTORY (last 5 turns):
{history}

USER QUESTION:
{query}

ANSWER:"""


# -----------------------------------------------
# Streaming generator (unchanged logic, cleaner)
# -----------------------------------------------
async def stream_gemini_response(
    prompt: str,
    memory_list: List[Dict],
    query: str,
):
    full_response = []
    try:
        response = model.generate_content(prompt, stream=True)
        for chunk in response:
            if chunk.text:
                full_response.append(chunk.text)
                yield chunk.text.encode("utf-8")
                await asyncio.sleep(0)
        if full_response:
            memory_list.append({
                "user": query,
                "assistant": "".join(full_response)
            })
    except Exception as e:
        yield f"\n[Error: {e}]".encode("utf-8")


# -----------------------------------------------
# Routes
# -----------------------------------------------
@app.get("/")
def home():
    return {"message": "Improved Gemini RAG Agent"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global vectorstore, bm25_index, bm25_corpus

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDFs only.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        # ✅ FIX 3a: Parent splitter — larger, for LLM context
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=150
        )
        # ✅ FIX 3b: Child splitter — smaller, for precise embedding
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200, chunk_overlap=30
        )

        parent_docs = parent_splitter.split_documents(docs)
        child_langchain_docs = []
        new_bm25_items = []

        for p_idx, parent_doc in enumerate(parent_docs):
            # Stable parent ID based on file + chunk position
            p_id = f"{file.filename}::parent::{p_idx}"
            parent_chunks[p_id] = parent_doc.page_content

            # Derive child chunks from this parent
            children = child_splitter.split_text(parent_doc.page_content)
            for c_idx, child_text in enumerate(children):
                c_id = f"{p_id}::child::{c_idx}"
                # Each child points back to its parent
                child_langchain_docs.append(
                    Document(
                        page_content=child_text,
                        metadata={**parent_doc.metadata, "child_id": p_id}
                    )
                )
                new_bm25_items.append({"id": p_id, "text": child_text})

        # Update FAISS
        if vectorstore is None:
            vectorstore = FAISS.from_documents(child_langchain_docs, embedding)
        else:
            vectorstore.add_documents(child_langchain_docs)
        vectorstore.save_local(VECTOR_DB_PATH)

        # Update BM25 index (rebuild from scratch to stay consistent)
        bm25_corpus.extend(new_bm25_items)
        tokenized_corpus = [item["text"].lower().split() for item in bm25_corpus]
        bm25_index = BM25Okapi(tokenized_corpus)
        save_bm25()
        save_parent_chunks()

        logger.info(f"Indexed {len(parent_docs)} parent / {len(child_langchain_docs)} child chunks")
        return {
            "message": "PDF uploaded and indexed ✅",
            "parent_chunks": len(parent_docs),
            "child_chunks": len(child_langchain_docs),
            "filename": file.filename
        }

    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
async def ask(request: Request, body: AskRequest):
    session_id = request.client.host if request.client else "default"
    if session_id not in CHAT_MEMORY:
        CHAT_MEMORY[session_id] = []

    if vectorstore is None:
        raise HTTPException(status_code=400, detail="Upload a PDF first.")

    query = body.query
    context_chunks = hybrid_search(query, top_k=6)

    if not context_chunks:
        raise HTTPException(status_code=404, detail="No relevant context found.")

    history = "\n".join(
        f"User: {m['user']}\nAssistant: {m['assistant']}"
        for m in CHAT_MEMORY[session_id][-5:]
    )

    prompt = build_prompt(context_chunks, history, query)

    return StreamingResponse(
        stream_gemini_response(prompt, CHAT_MEMORY[session_id], query),
        media_type="text/plain"
    )