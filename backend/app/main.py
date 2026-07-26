from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Advanced AI Medical Intelligence Platform",
    description="AI-powered medical image analysis with Explainable AI and LLM integration.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # We'll tighten this for production later.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "message": "Advanced AI Medical Intelligence Platform API",
        "status": "running"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }