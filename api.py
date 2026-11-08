import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

model = joblib.load(MODELS_DIR / "best_model.joblib")
with open(MODELS_DIR / "best_metrics.json") as f:
    best_metrics = json.load(f)
with open(MODELS_DIR / "feature_importance.json") as f:
    feature_importance = json.load(f)

INPUT_COLUMNS = best_metrics["input_columns"]
CAT_COLUMNS = [
    "workclass",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

app = FastAPI(title="Adult Income Prediction API", version="1.0")


class Person(BaseModel):
    age: int = Field(..., ge=17, le=100)
    workclass: Optional[str] = None
    education_num: int = Field(..., ge=1, le=16)
    marital_status: str
    occupation: Optional[str] = None
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(0, ge=0)
    capital_loss: int = Field(0, ge=0)
    hours_per_week: int = Field(..., ge=1, le=100)
    native_country: Optional[str] = None


def to_frame(p: Person) -> pd.DataFrame:
    row = p.model_dump()
    # Training jaisi hi feature engineering
    row["net_capital"] = row["capital_gain"] - row["capital_loss"]
    row["has_capital_gain"] = int(row["capital_gain"] > 0)
    row["has_capital_loss"] = int(row["capital_loss"] > 0)
    row["is_full_time"] = int(row["hours_per_week"] >= 40)
    df = pd.DataFrame([row])[INPUT_COLUMNS]
    for c in CAT_COLUMNS:
        df[c] = df[c].astype(object).where(df[c].notna(), np.nan)
    return df


@app.get("/")
def root():
    return {"message": "Adult Income API is running. See /docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return {
        "model": best_metrics["model"],
        "test_metrics": {
            k: best_metrics[k]
            for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]
        },
        "top_features": feature_importance[:5],
    }


@app.post("/predict")
def predict(person: Person):
    df = to_frame(person)
    proba = float(model.predict_proba(df)[0, 1])
    label = int(proba >= 0.5)
    return {
        "prediction": ">50K" if label == 1 else "<=50K",
        "label": label,
        "probability_over_50k": round(proba, 4),
    }
