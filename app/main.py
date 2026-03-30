from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.auth import router as auth_router
from app.api.routes.bots import router as bots_router
from app.api.routes.chat import router as chat_router
from app.api.routes.pdf_upload import router as pdf_router
from app.core.config import settings
from app.db.database import init_db


logging.basicConfig(level=getattr(logging, str(settings.log_level).upper(), logging.INFO))
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(settings.uploads_root, exist_ok=True)
    os.makedirs(settings.vectorstore_root, exist_ok=True)
    await init_db()
    yield


app = FastAPI(title="Multi-tenant Gemini RAG Backend", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home() -> dict:
    return {"status": "ok", "message": "Gemini RAG backend is running"}


app.include_router(auth_router)
app.include_router(bots_router)
app.include_router(pdf_router)
app.include_router(chat_router)

