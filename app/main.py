from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import select

from app.core.config import get_settings
from app.core.database import AsyncSessionLocal
from app.models.chunk import DocumentChunk
from app.routers import auth, workspaces, documents, search, chat
from app.services.storage import StorageService
from app.services.vector_store import vector_store

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"✅ Starting {settings.app_name}")

    # MinIO bucket
    try:
        storage = StorageService()
        storage.ensure_bucket_exists()
        print("✅ MinIO bucket ready")
    except Exception as e:
        print(f"⚠️  MinIO bucket setup failed: {e}")

    # Qdrant collection
    try:
        vector_store.ensure_collection_exists()
        print("✅ Qdrant collection ready")
    except Exception as e:
        print(f"⚠️  Qdrant setup failed: {e}")

    # BM25 index — load all existing chunks from PostgreSQL on startup
    try:
        from app.services.retrieval.bm25_index import bm25_index

        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(DocumentChunk).order_by(DocumentChunk.created_at)
            )
            all_chunks = result.scalars().all()

        chunk_dicts = [
            {
                "id": c.id,
                "workspace_id": c.workspace_id,
                "document_id": c.document_id,
                "chunk_text": c.chunk_text,
                "page_num": c.page_num,
                "chunk_index": c.chunk_index,
            }
            for c in all_chunks
        ]

        if chunk_dicts:
            bm25_index.build(chunk_dicts)
            print(f"✅ BM25 index loaded: {len(chunk_dicts)} chunks")
        else:
            print("ℹ️  BM25 index empty — no chunks in database yet")

    except Exception as e:
        print(f"⚠️  BM25 index build failed: {e}")

    yield
    print("Shutting down")


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    debug=settings.debug,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Router registration — each router registered exactly once
app.include_router(auth.router)
app.include_router(workspaces.router)
app.include_router(documents.router)
app.include_router(search.router)
app.include_router(chat.router)


@app.get("/health", tags=["Health"])
async def health_check():
    return JSONResponse(
        status_code=200,
        content={
            "status": "OK",
            "app": settings.app_name,
        },
    )