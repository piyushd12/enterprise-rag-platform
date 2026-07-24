from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.routers import auth, workspaces

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Starting {settings.app_name}")
    yield
    print("Shutting down")

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    debug=settings.debug,
    lifespan=lifespan
)

app.include_router(auth.router)
app.include_router(workspaces.router)   

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(auth.router)

@app.get("/health",tags=["Health"])
async def health_check():
    return JSONResponse(
        status_code=200,
        content={
            "status" : "OK",
            "app" : settings.app_name
        }
    )