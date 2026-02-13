from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import numpy as np
import os

# =======================
# App Initialization
# =======================
app = FastAPI(
    title="Smart Doctor Recommendation API",
    description="ML-powered doctor recommendation system",
    version="1.2.0"
)

# =======================
# Paths
# =======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # app folder
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")      # ../model

# =======================
# Load Pickles
# =======================
try:
    model = pickle.load(open(os.path.join(MODEL_DIR, "doctor_model.pkl"), "rb"))
    scaler = pickle.load(open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb"))
    doctors_df = pickle.load(open(os.path.join(MODEL_DIR, "doctors_data.pkl"), "rb"))
except Exception as e:
    raise RuntimeError(f"Failed to load model or doctors data: {e}")

# =======================
# Input Schema
# =======================
class PatientInput(BaseModel):
    age: int = Field(..., example=35)
    gender: int = Field(..., example=1)
    location: str = Field(..., example="Lahore")
    chronic_conditions: str = Field(..., example="Diabetes")
    top_n: int = Field(default=3, example=3)

# =======================
# Utility Functions
# =======================
def compute_features(patient: PatientInput, doctor_row) -> np.ndarray:
    location_match = int(patient.location.lower() == doctor_row['location'].lower())
    has_chronic = int(patient.chronic_conditions.lower() != "none")
    condition_match = int(has_chronic and doctor_row['specialty'].lower() in patient.chronic_conditions.lower())

    years_exp = doctor_row['years_experience']
    if years_exp <= 5: experience_level = 0
    elif years_exp <= 10: experience_level = 1
    elif years_exp <= 20: experience_level = 2
    else: experience_level = 3

    features = np.array([[ 
        patient.age,
        patient.gender,
        location_match,
        has_chronic,
        condition_match,
        doctor_row['years_experience'],
        doctor_row['rating'],
        doctor_row['success_rate'],
        experience_level,
        doctor_row['specialty_code']
    ]])

    # Scale numeric features: age, years_exp, rating, success_rate
    features[:, [0,5,6,7]] = scaler.transform(features[:, [0,5,6,7]])

    return features

# =======================
# API Routes
# =======================
@app.get("/")
def root():
    return {"message": "Smart Doctor Recommendation API is live"}

@app.get("/health")
def health():
    return {"model_loaded": True, "doctors_loaded": True, "status": "healthy"}

@app.post("/recommend_top_n")
def recommend_top_n(patient: PatientInput):
    try:
        scores = []
        for _, doctor in doctors_df.iterrows():
            features = compute_features(patient, doctor)
            prob = model.predict_proba(features)[0][1]

            scores.append({
                "doctor_id": doctor['doctor_id'],
                "doctor_name": doctor['doctor_name'],
                "specialty": doctor['specialty'],
                "years_experience": doctor['years_experience'],
                "rating": doctor['rating'],
                "success_rate": doctor['success_rate'],
                "score": round(prob, 3)
            })

        # Sort by score descending and take top-N
        top_doctors = sorted(scores, key=lambda x: x['score'], reverse=True)[:patient.top_n]

        return {"top_doctors": top_doctors}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
