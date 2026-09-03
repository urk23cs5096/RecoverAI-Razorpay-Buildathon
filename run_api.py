"""Quickstart script to launch RecoverAI FastAPI REST Backend."""

import uvicorn
from recoverai.config.settings import settings
from recoverai.storage.database import init_db

if __name__ == "__main__":
    init_db()
    print(f"Starting RecoverAI FastAPI Backend on http://{settings.API_HOST}:{settings.API_PORT}")
    print(f"Interactive API Docs available at http://{settings.API_HOST}:{settings.API_PORT}/docs")
    uvicorn.run(
        "recoverai.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=False,
    )
