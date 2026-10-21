from fastapi import FastAPI

from app.core import tenancy as _tenancy  # noqa: F401  (importing registers the per-transaction tenant-context hook)
from app.core.errors import AppError, app_error_handler
from app.routers import (
    agent, ask, ask_langchain, ask_structured, audit, auth, documents, lab, retrieval, users,
)

app = FastAPI(title="RAG Pipeline")

app.add_exception_handler(AppError, app_error_handler)

app.include_router(lab.router)
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(audit.router)
app.include_router(documents.router)
app.include_router(retrieval.router)
app.include_router(ask.router)
app.include_router(ask_langchain.router)
app.include_router(ask_structured.router)
app.include_router(agent.router)


@app.get("/ping")
async def ping():
    return {"status": "ok", "message": "pong"}
