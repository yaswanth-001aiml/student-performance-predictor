from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import os

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "student_model.pkl")

model = joblib.load(MODEL_PATH)


class StudentData(BaseModel):
    Hours_Studied: float
    Attendance: float
    Sleep_Hours: float
    Previous_Scores: float
    Tutoring_Sessions: float


@app.get("/")
def home():
    return {"message": "Student Performance Predictor API is running!"}


@app.post("/predict")
def predict(data: StudentData):

    features = [[
        data.Hours_Studied,
        data.Attendance,
        data.Sleep_Hours,
        data.Previous_Scores,
        data.Tutoring_Sessions
    ]]

    prediction = model.predict(features)[0]

    score = round(float(prediction), 2)

    # Grade
    if score >= 90:
        grade = "A+"
    elif score >= 80:
        grade = "A"
    elif score >= 70:
        grade = "B"
    elif score >= 60:
        grade = "C"
    elif score >= 50:
        grade = "D"
    else:
        grade = "F"

    status = "PASS" if score >= 35 else "FAIL"

    return {
        "predicted_score": score,
        "grade": grade,
        "status": status
    }
