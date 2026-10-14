import pandas as pd
import joblib

# Load trained pipelines
cath_model = joblib.load("model_files/cath_pipeline.pkl")
lad_model = joblib.load("model_files/lad_pipeline.pkl")
lcx_model = joblib.load("model_files/lcx_pipeline.pkl")
rca_model = joblib.load("model_files/rca_pipeline.pkl")

# Prediction function
def predict_patient(patient_data):

    patient_df = pd.DataFrame([patient_data])

    models = {
        "Cath": cath_model,
        "LAD": lad_model,
        "LCX": lcx_model,
        "RCA": rca_model
    }

    results = {}

    for target, model in models.items():

        prediction = model.predict(patient_df)[0]

        probabilities = model.predict_proba(patient_df)[0]
        classes = model.classes_

        predicted_index = list(classes).index(prediction)
        probability = probabilities[predicted_index]

        results[target] = {
            "prediction": prediction,
            "probability": float(probability)
        }

    return results