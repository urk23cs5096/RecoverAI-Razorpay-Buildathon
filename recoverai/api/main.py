"""FastAPI Application Entrypoint for RecoverAI."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from recoverai.api.routes import router
from recoverai.storage.database import init_db
from recoverai.config.settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("recoverai.api.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle event handler for database initialization and warmup."""
    logger.info("Initializing RecoverAI Database Schema...")
    init_db()
    logger.info("RecoverAI API Server started successfully.")
    yield
    logger.info("Shutting down RecoverAI API Server...")


app = FastAPI(
    title="RecoverAI — AI Revenue Recovery Agent",
    description="Intelligent, cost-aware autonomous revenue recovery agent for Razorpay merchants.",
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for local Streamlit / Frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "recoverai.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
    )
