from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load trained model and scaler
model = joblib.load("employability_model.pkl")
scaler = joblib.load("scaler.pkl")


@app.route("/", methods=["GET"])
def home():
    return "SkillLens API is running"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    required_fields = [
        "cgpa", "projects", "internships", "certifications",
        "python", "ml", "sql", "communication"
    ]

    # Validate input
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

    # Prepare features
    features = np.array([[
        data["cgpa"],
        data["projects"],
        data["internships"],
        data["certifications"],
        data["python"],
        data["ml"],
        data["sql"],
        data["communication"]
    ]])

    features_scaled = scaler.transform(features)

    # Prediction + probability
    probability = model.predict_proba(features_scaled)[0][1]
    score = int(probability * 100)

    # Skill gap logic
    skill_gaps = []

    if data["ml"] == 0:
        skill_gaps.append("Machine Learning fundamentals")

    if data["internships"] == 0:
        skill_gaps.append("Internship experience")

    if data["projects"] < 2:
        skill_gaps.append("More hands-on projects")

    if data["communication"] == 0:
        skill_gaps.append("Communication skills")

    confidence = (
        "High" if score >= 75
        else "Medium" if score >= 50
        else "Low"
    )

    return jsonify({
        "employability_score": score,
        "confidence": confidence,
        "label": "Employable" if score >= 60 else "Needs Improvement",
        "skill_gaps": skill_gaps
    })


if __name__ == "__main__":
    app.run(debug=True)
