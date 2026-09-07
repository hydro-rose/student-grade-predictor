"""FastAPI Application Server for Student Grade Prediction System.
Provides RESTful APIs for real-time inference, batch analysis,
model metadata, and serves the frontend dashboard.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import io
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predictor import GradePredictor
from src.data_loader import load_raw_data, CORE_FEATURES

app = FastAPI(
    title="SynapseGrade AI - Neural Academic Intelligence Platform",
    description="Next-generation multi-model performance forecasting, explainable AI (XAI), and continuous academic evaluation.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

APP_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = APP_DIR / "templates"
STATIC_DIR = APP_DIR / "static"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Initialize predictor singleton
predictor = GradePredictor()


class StudentPredictionRequest(BaseModel):
    studytime: float = Field(2.0, ge=1.0, le=4.0, description="Weekly study time (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h)")
    failures: int = Field(0, ge=0, le=4, description="Number of past class failures (0 to 4)")
    absences: int = Field(4, ge=0, le=93, description="Number of school absences (0 to 93)")
    g1: float = Field(12.0, ge=0.0, le=20.0, description="First period grade (0 to 20)")
    g2: float = Field(13.0, ge=0.0, le=20.0, description="Second period grade (0 to 20)")
    model_name: Optional[str] = Field("Linear Regression", description="Selected AI/ML regression model")


class GoalOptimizationRequest(BaseModel):
    target_g3: float = Field(14.0, ge=0.0, le=20.0, description="Desired final grade target (0 to 20)")
    studytime: float = Field(2.0, ge=1.0, le=4.0)
    failures: int = Field(0, ge=0, le=4)
    absences: int = Field(4, ge=0, le=93)
    g1: float = Field(11.0, ge=0.0, le=20.0)
    g2: float = Field(12.0, ge=0.0, le=20.0)


@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    """Serve main interactive dashboard."""
    metadata = predictor.metadata
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "metadata": metadata,
            "available_models": predictor.available_models,
        },
    )


@app.post("/api/predict")
async def predict_grade(payload: StudentPredictionRequest) -> Dict[str, Any]:
    """Generate real-time G3 grade prediction using selected AI/ML model."""
    try:
        result = predictor.predict_single(
            studytime=payload.studytime,
            failures=payload.failures,
            absences=payload.absences,
            g1=payload.g1,
            g2=payload.g2,
            model_name=payload.model_name or "Linear Regression",
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models")
async def get_models() -> Dict[str, Any]:
    """Retrieve full AI model comparison leaderboards and available architectures."""
    return {
        "available_models": predictor.available_models,
        "regression_leaderboard": predictor.metadata.get("regression_leaderboard", []),
        "classification_leaderboard": predictor.metadata.get("classification_leaderboard", []),
    }


@app.post("/api/explain")
async def explain_grade(payload: StudentPredictionRequest) -> Dict[str, Any]:
    """Explainable AI (XAI): Returns waterfall feature attribution points."""
    try:
        return predictor.explain_prediction(
            studytime=payload.studytime,
            failures=payload.failures,
            absences=payload.absences,
            g1=payload.g1,
            g2=payload.g2,
            model_name=payload.model_name or "Linear Regression",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/optimize-goal")
async def optimize_academic_goal(payload: GoalOptimizationRequest) -> Dict[str, Any]:
    """Counterfactual AI Goal Optimizer: Determines study/absence changes to hit target score."""
    try:
        return predictor.optimize_goal(
            target_g3=payload.target_g3,
            studytime=payload.studytime,
            failures=payload.failures,
            absences=payload.absences,
            g1=payload.g1,
            g2=payload.g2,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/clusters")
async def get_clusters() -> Dict[str, Any]:
    """Retrieve unsupervised K-Means student persona clusters and silhouette scores."""
    return predictor.metadata.get("unsupervised_clusters", {})


@app.get("/api/metrics")
async def get_metrics() -> Dict[str, Any]:
    """Retrieve model training metrics, benchmark scores, coefficients, and correlation matrix."""
    return predictor.metadata


@app.get("/api/sample-students")
async def get_sample_students() -> List[Dict[str, Any]]:
    """Provide realistic preset student profiles for rapid testing."""
    return [
        {
            "id": "STU-101",
            "name": "Maria Silva",
            "profile": "High Honor Student",
            "studytime": 3,
            "failures": 0,
            "absences": 2,
            "G1": 16,
            "G2": 17,
        },
        {
            "id": "STU-102",
            "name": "Joao Santos",
            "profile": "Average Consistent",
            "studytime": 2,
            "failures": 0,
            "absences": 6,
            "G1": 11,
            "G2": 12,
        },
        {
            "id": "STU-103",
            "name": "Lucas Ferreira",
            "profile": "At-Risk Borderline",
            "studytime": 1,
            "failures": 2,
            "absences": 22,
            "G1": 7,
            "G2": 8,
        },
        {
            "id": "STU-104",
            "name": "Beatriz Costa",
            "profile": "Rapid Improver",
            "studytime": 3,
            "failures": 0,
            "absences": 4,
            "G1": 9,
            "G2": 14,
        },
        {
            "id": "STU-105",
            "name": "Tiago Oliveira",
            "profile": "Chronic Absentee",
            "studytime": 1,
            "failures": 1,
            "absences": 38,
            "G1": 8,
            "G2": 7,
        },
    ]


@app.post("/api/predict-batch")
async def predict_batch(file: UploadFile = File(...), model_name: str = "Linear Regression"):
    """Upload CSV to evaluate an entire class cohort and identify at-risk students."""
    try:
        contents = await file.read()
        try:
            df = pd.read_csv(io.StringIO(contents.decode("utf-8")), sep=",")
            if not all(col in df.columns for col in CORE_FEATURES):
                df = pd.read_csv(io.StringIO(contents.decode("utf-8")), sep=";")
        except Exception:
            df = pd.read_csv(io.StringIO(contents.decode("utf-8")), sep=";")

        batch_results = predictor.predict_batch(df, model_name=model_name)
        records = batch_results.to_dict(orient="records")
        
        # Aggregate cohort statistics
        total = len(records)
        passing = sum(1 for r in records if r.get("predicted_status") == "PASS")
        at_risk = total - passing
        avg_predicted = round(float(batch_results["predicted_G3"].mean()), 2)

        return {
            "total_students": total,
            "pass_count": passing,
            "at_risk_count": at_risk,
            "pass_percentage": round((passing / total) * 100, 1) if total > 0 else 0,
            "average_predicted_grade": avg_predicted,
            "students": records[:50],  # Return up to 50 for visualization
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process CSV: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
