"""
FastAPI server for your agentic AI system.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Union, Optional
import uuid
from datetime import datetime

from src.utils.cache import LLMCache
from src.utils.config import Config
from src.agents.gemini_agent import GeminiAgent
from src.agents.xai_agent import XAIAgent

# Initialize FastAPI
app = FastAPI(
    title="Agentic AI API",
    description="Multi-agent AI system with caching",
    version="1.0.0"
)

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize cache and agents
cache = LLMCache(cache_dir=str(Config.CACHE_DIR))
gemini = GeminiAgent()
xai = XAIAgent()


# Request/Response models
class Message(BaseModel):
    role: str
    content: str


class GenerateRequest(BaseModel):
    query: Union[str, List[Message]]
    model: str = "gemini"
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class GenerateResponse(BaseModel):
    text: str
    model: str
    cached: bool
    metadata: Dict


class CompareRequest(BaseModel):
    query: str
    models: List[str] = ["gemini", "xai"]


class CompareResponse(BaseModel):
    query: str
    results: List[Dict[str, str]]


# Endpoints
@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Agentic AI API",
        "version": "1.0.0"
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate response from a single AI model.

    Example:
        POST /generate
        {
            "query": "What is quantum computing?",
            "model": "gemini",
            "temperature": 0.7
        }
    """
    # Select agent
    if request.model == "gemini":
        agent = gemini
    elif request.model == "xai":
        agent = xai
    else:
        raise HTTPException(status_code=400, detail=f"Unknown model: {request.model}")

    # Prepare kwargs
    kwargs = {}
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature
    if request.max_tokens is not None:
        kwargs["max_tokens"] = request.max_tokens

    # Check if cached
    cached_result = cache.get(agent.model_name, request.query, **kwargs)
    is_cached = cached_result is not None

    # Generate or retrieve from cache
    result = cache.cached_api_call(
        model_name=agent.model_name,
        query=request.query,
        api_function=agent.generate,
        **kwargs
    )

    return GenerateResponse(
        text=result["text"],
        model=result["model"],
        cached=is_cached,
        metadata=result.get("metadata", {})
    )


@app.post("/compare", response_model=CompareResponse)
async def compare(request: CompareRequest):
    """
    Compare responses from multiple models.

    Example:
        POST /compare
        {
            "query": "Explain AI",
            "models": ["gemini", "xai"]
        }
    """
    results = []

    for model_name in request.models:
        if model_name == "gemini":
            agent = gemini
        elif model_name == "xai":
            agent = xai
        else:
            continue

        result = cache.cached_api_call(
            model_name=agent.model_name,
            query=request.query,
            api_function=agent.generate
        )

        results.append({
            "model": model_name,
            "response": result["text"]
        })

    return CompareResponse(
        query=request.query,
        results=results
    )


@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics."""
    return cache.get_cache_size()


@app.delete("/cache/clear")
async def clear_cache():
    """Clear the entire cache."""
    cache.clear()
    return {"status": "cache cleared"}

# Run with: uvicorn api_server:app --reload --port 8000