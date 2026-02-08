"""
Startup script for the API server.
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "src.apis.api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )