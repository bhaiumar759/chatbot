from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel
from collections.abc import AsyncGenerator   # ✅ Python 3.13 style
import google.generativeai as genai
import os, shutil, logging, asyncio, pickle
from typing import Any
from rank_bm25 import BM25Okapi
import numpy as np

# ──────────────────────────────────────────
# Logging
# ──────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────
# App
# ──────────────────────────────────────────
app = FastAPI(title="Gemini RAG Agent – Python 3.13")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

UPLOAD_DIR     = "uploads"
VECTOR_DB_PATH = "vectorstore"
BM25_PATH      = os.path.join(VECTOR_DB_PATH, "bm25.pkl")
PARENT_PATH    = os.path.join(VECTOR_DB_PATH, "parent_chunks.pkl")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_PATH, exist_ok=True)

# ──────────────────────────────────────────
# Gemini config
# ──────────────────────────────────────────
GEMINI_API_KEY = "AIzaSyCFft1Yvo9s2uCzwh-eutm9fje-QNaZe58"
genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL = genai.GenerativeModel("gemini-3-flash-preview")

# ──────────────────────────────────────────
# ✅ FIX: AskRequest defined at top level
#    — before any function or route uses it
# ──────────────────────────────────────────
class AskRequest(BaseModel):
    query: str

# ──────────────────────────────────────────
# In-memory stores
# ──────────────────────────────────────────
CHAT_MEMORY:   dict[str, list[dict[str, str]]] = {}
parent_chunks: dict[str, str] = {}
bm25_corpus:   list[dict]     = []
bm25_index:    Any | None     = None
vectorstore:   FAISS | None   = None


# ──────────────────────────────────────────
# Embeddings
# ──────────────────────────────────────────
class GeminiEmbedding(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        result = genai.embed_content(
            model="gemini-embedding-001",
            content=text,
            task_type="retrieval_document",
        )
        return result["embedding"]

embedding_fn = GeminiEmbedding()


# ──────────────────────────────────────────
# Persistence helpers
# ──────────────────────────────────────────
def _save(path: str, obj: object) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def _load(path: str) -> Any | None:
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

def load_all() -> None:
    global vectorstore, bm25_index, bm25_corpus, parent_chunks

    if os.path.exists(os.path.join(VECTOR_DB_PATH, "index.faiss")):
        try:
            vectorstore = FAISS.load_local(
                VECTOR_DB_PATH, embedding_fn,
                allow_dangerous_deserialization=True,
            )
            logger.info("FAISS loaded ✅")
        except Exception as e:
            logger.error(f"FAISS load failed: {e}")

    data = _load(BM25_PATH)
    if data:
        bm25_index, bm25_corpus = data
        logger.info("BM25 loaded ✅")

    data = _load(PARENT_PATH)
    if data:
        parent_chunks = data
        logger.info(f"Parent chunks loaded: {len(parent_chunks)} ✅")

load_all()


# ──────────────────────────────────────────
# HyDE
# ──────────────────────────────────────────
def generate_hypothetical_answer(query: str) -> str:
    try:
        resp = GEMINI_MODEL.generate_content(
            f"Write one concise factual paragraph that directly answers: '{query}'. "
            f"Use specific details. For document retrieval only."
        )
        return resp.text.strip()
    except Exception as e:
        logger.warning(f"HyDE failed, using raw query: {e}")
        return query


# ──────────────────────────────────────────
# Reciprocal Rank Fusion
# ──────────────────────────────────────────
def reciprocal_rank_fusion(
    dense_ids: list[str],
    sparse_ids: list[str],
    k: int = 60,
) -> list[str]:
    scores: dict[str, float] = {}
    for rank, doc_id in enumerate(dense_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, doc_id in enumerate(sparse_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda x: scores[x], reverse=True)


# ──────────────────────────────────────────
# Hybrid search
# ──────────────────────────────────────────
def hybrid_search(query: str, top_k: int = 6) -> list[str]:
    if vectorstore is None:
        return []

    hyde_text   = generate_hypothetical_answer(query)
    dense_docs  = vectorstore.similarity_search(hyde_text, k=top_k * 2)
    dense_ids   = [d.metadata.get("child_id", d.page_content[:40]) for d in dense_docs]

    sparse_ids: list[str] = []
    if bm25_index is not None and bm25_corpus:
        tokens      = query.lower().split()
        bm25_scores = bm25_index.get_scores(tokens)
        top_idx     = np.argsort(bm25_scores)[::-1][: top_k * 2]
        sparse_ids  = [bm25_corpus[i]["id"] for i in top_idx]

    fused_ids = reciprocal_rank_fusion(dense_ids, sparse_ids)[:top_k]

    seen: set[str] = set()
    contexts: list[str] = []
    for doc_id in fused_ids:
        text = parent_chunks.get(doc_id, "")
        if text and text not in seen:
            seen.add(text)
            contexts.append(text)
    return contexts


# ──────────────────────────────────────────
# Prompt builder
# ──────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a precise document assistant. Answer ONLY from the CONTEXT below.\n\n"
    "RULES:\n"
    "1. Never use outside knowledge.\n"
    "2. If context is insufficient say: "
    "'The document does not contain enough information to answer this.'\n"
    "3. Quote dates, numbers, and names directly from context.\n"
    "4. No padding or filler sentences.\n"
    "5. For summaries, cover every key point in the context.\n"
)

def build_prompt(chunks: list[str], history: str, query: str) -> str:
    context = "\n\n---\n\n".join(
        f"[Chunk {i+1}]\n{c}" for i, c in enumerate(chunks)
    )
    return (
        f"{SYSTEM_PROMPT}\n"
        f"CONTEXT:\n{context}\n\n"
        f"CONVERSATION HISTORY:\n{history}\n\n"
        f"USER QUESTION:\n{query}\n\n"
        f"ANSWER:"
    )


# ──────────────────────────────────────────
# Streaming generator
# ──────────────────────────────────────────
async def stream_response(
    prompt: str,
    memory: list[dict[str, str]],
    query: str,
) -> AsyncGenerator[bytes, None]:
    tokens: list[str] = []
    try:
        response = GEMINI_MODEL.generate_content(prompt, stream=True)
        for chunk in response:
            if chunk.text:
                tokens.append(chunk.text)
                yield chunk.text.encode()
                await asyncio.sleep(0)
        if tokens:
            memory.append({"user": query, "assistant": "".join(tokens)})
    except Exception as e:
        err = f"\n[Stream error: {e}]"
        logger.error(err)
        yield err.encode()


# ──────────────────────────────────────────
# Routes
# ──────────────────────────────────────────
@app.get("/")
def home() -> dict:
    return {"status": "ok", "message": "Gemini RAG Agent running"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)) -> dict:
    global vectorstore, bm25_index, bm25_corpus

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted.")

    path = os.path.join(UPLOAD_DIR, file.filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        docs = PyPDFLoader(path).load()

        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=150,
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200, chunk_overlap=30,
        )

        parent_docs                    = parent_splitter.split_documents(docs)
        child_lc_docs:  list[Document] = []
        new_bm25_items: list[dict]     = []

        for p_idx, p_doc in enumerate(parent_docs):
            p_id                = f"{file.filename}::p::{p_idx}"
            parent_chunks[p_id] = p_doc.page_content

            for c_idx, child_text in enumerate(
                child_splitter.split_text(p_doc.page_content)
            ):
                child_lc_docs.append(Document(
                    page_content=child_text,
                    metadata={**p_doc.metadata, "child_id": p_id},
                ))
                new_bm25_items.append({"id": p_id, "text": child_text})

        if vectorstore is None:
            vectorstore = FAISS.from_documents(child_lc_docs, embedding_fn)
        else:
            vectorstore.add_documents(child_lc_docs)
        vectorstore.save_local(VECTOR_DB_PATH)

        bm25_corpus.extend(new_bm25_items)
        bm25_index = BM25Okapi([item["text"].lower().split() for item in bm25_corpus])
        _save(BM25_PATH,   (bm25_index, bm25_corpus))
        _save(PARENT_PATH, parent_chunks)

        logger.info(f"{file.filename}: {len(parent_docs)} parents, {len(child_lc_docs)} children")
        return {
            "message":       "Uploaded and indexed ✅",
            "filename":      file.filename,
            "parent_chunks": len(parent_docs),
            "child_chunks":  len(child_lc_docs),
        }

    except Exception as e:
        logger.exception("Upload failed")
        raise HTTPException(500, str(e))


@app.post("/ask")
async def ask(request: Request, body: AskRequest) -> StreamingResponse:
    sid = request.client.host if request.client else "default"
    CHAT_MEMORY.setdefault(sid, [])

    if vectorstore is None:
        raise HTTPException(400, "No documents indexed yet — upload a PDF first.")

    chunks = hybrid_search(body.query, top_k=6)
    if not chunks:
        raise HTTPException(404, "No relevant content found for that query.")

    history = "\n".join(
        f"User: {m['user']}\nAssistant: {m['assistant']}"
        for m in CHAT_MEMORY[sid][-5:]
    )
    prompt = build_prompt(chunks, history, body.query)

    return StreamingResponse(
        stream_response(prompt, CHAT_MEMORY[sid], body.query),
        media_type="text/plain",
    )