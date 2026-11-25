"""AI Results Assistant chat routes."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from services.chat import ask_ai_assistant, detect_provider, get_active_model, load_env_file

HERE = Path(__file__).resolve().parents[1]
router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.get("/status")
async def chat_status() -> dict[str, Any]:
    """Return whether an AI assistant API key is configured in .env."""
    load_env_file(HERE / ".env")
    provider, _ = detect_provider()
    return {
        "configured": provider is not None,
        "provider": provider,
        "model": get_active_model(provider),
    }


@router.post("")
async def chat_endpoint(request: Request) -> dict[str, Any]:
    """Explain recombination results or selected graph regions using the configured AI key."""
    load_env_file(HERE / ".env")
    body = await request.json()
    question = str(body.get("question", "")).strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    history = body.get("history", [])
    run_context = body.get("run_context")
    return ask_ai_assistant(question=question, history=history, run_context=run_context)
