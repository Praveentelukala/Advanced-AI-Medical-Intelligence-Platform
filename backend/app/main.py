from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Import Prediction Router
from app.api.predict import router as predict_router

# Create FastAPI app
app = FastAPI(
    title="Advanced AI Medical Intelligence Platform",
    description="AI-powered medical image analysis with Explainable AI and LLM integration.",
    version="1.0.0",
)

# Mount static heatmaps folder
app.mount("/heatmaps", StaticFiles(directory="heatmaps"), name="heatmaps")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # We'll tighten this for production later.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(predict_router)


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