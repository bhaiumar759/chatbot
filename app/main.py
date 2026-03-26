from fastapi import FastAPI, UploadFile, File, HTTPException
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
import os
import shutil
import logging
import requests
from sentence_transformers import SentenceTransformer
from typing import Optional

# -------------------------------
# Logging
# -------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------
# App Config
# -------------------------------
app = FastAPI(title="RAG API (100% Working)")

VECTOR_DB_PATH = "vectorstore"
UPLOAD_DIR = "uploads"

OLLAMA_API_KEY = "9a44b57de4c8429e8e5a58ac98dd39be.FRNk-Vay6gnHVUdnQkee5Y3V"
OLLAMA_URL = "https://ollama.com/api/generate"

# ✅ WORKING MODEL
OLLAMA_MODEL = "gpt-oss:120b"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_PATH, exist_ok=True)

# -------------------------------
# Embeddings (FAST + ACCURATE)
# -------------------------------
class MyEmbeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()


embeddings = MyEmbeddings()

# -------------------------------
# Ollama Call (FIXED)
# -------------------------------
def ollama_generate(prompt: str):
    headers = {
        "Authorization": f"Bearer {OLLAMA_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 800
        }
    }

    try:
        res = requests.post(OLLAMA_URL, headers=headers, json=payload, timeout=120)

        if res.status_code != 200:
            logger.error(res.text)
            return f"Error: {res.text}"

        data = res.json()

        return data.get("response", "").strip() or "Empty response from model."

    except Exception as e:
        logger.error(str(e))
        return f"Connection error: {str(e)}"


# -------------------------------
# Vector Store
# -------------------------------
vectorstore: Optional[FAISS] = None


@app.get("/")
def home():
    return {"message": "RAG API is running 🚀"}


# -------------------------------
# Upload PDF
# -------------------------------
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global vectorstore

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF allowed")

    path = os.path.join(UPLOAD_DIR, file.filename)

    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        loader = PyPDFLoader(path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )

        chunks = splitter.split_documents(docs)

        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(VECTOR_DB_PATH)

        return {
            "message": "PDF processed successfully",
            "chunks": len(chunks)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------
# Ask Question (HIGH ACCURACY)
# -------------------------------
@app.post("/ask")
async def ask(query: str):
    global vectorstore

    if not query:
        raise HTTPException(status_code=400, detail="Query required")

    # Load vectorstore if needed
    if vectorstore is None:
        if os.path.exists(VECTOR_DB_PATH):
            vectorstore = FAISS.load_local(
                VECTOR_DB_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            return {"answer": "Upload a PDF first"}

    # Retrieve context
    docs = vectorstore.similarity_search(query, k=5)

    context = "\n\n".join([d.page_content for d in docs])

    # ✅ STRONG PROMPT (VERY IMPORTANT)
    prompt = f"""
You are a highly accurate AI assistant.

RULES:
- Answer ONLY from the provided context
- DO NOT make up information
- If answer not found, say: "I don't have enough information in the document"
- Keep answer clear and professional

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    answer = ollama_generate(prompt)

    return {
        "question": query,
        "answer": answer,
        "chunks_used": len(docs),
        "model": OLLAMA_MODEL
    }