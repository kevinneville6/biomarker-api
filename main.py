from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
import os
import io

app = FastAPI(title="Biomarker Analysis API")

# Load model and feature schema once at startup
MODEL_PATH = "biomarker_model.pkl"
FEATURES_PATH = "feature_columns.pkl"

biomarker_model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURES_PATH)

DATA_DIR = "data_outputs"
os.makedirs(DATA_DIR, exist_ok=True)


@app.post("/agent1/process")
async def process_data(file: UploadFile = File(None), source: str = Form("csv")):
    """
    Preprocess uploaded CSV or SQL-fetched data.
    Align columns to feature schema.
    """
    try:
        if source == "csv":
            contents = await file.read()
            df = pd.read_csv(io.BytesIO(contents))
        else:
            # Placeholder: connect to SQL Server if needed
            return JSONResponse({"status": "error", "message": "SQL fetch not implemented"}, status_code=400)

        # Align features
        missing_cols = [c for c in feature_columns if c not in df.columns]
        for c in missing_cols:
            df[c] = 0
        df = df[feature_columns]

        features_path = os.path.join(DATA_DIR, "features.csv")
        df.to_csv(features_path, index=False)

        return {"status": "success", "features_path": features_path}

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/agent2/predict")
async def predict(features_path: str = Form(...)):
    """
    Run model prediction on preprocessed features.
    """
    try:
        df = pd.read_csv(features_path)
        preds = biomarker_model.predict(df)
        conf = biomarker_model.predict_proba(df)[:, 1] if hasattr(biomarker_model, "predict_proba") else None

        result_df = pd.DataFrame({
            "Biomarker": df.index,
            "Predicted_Class": preds,
            "Confidence": conf
        })

        scored_path = os.path.join(DATA_DIR, "scored_predictions.csv")
        result_df.to_csv(scored_path, index=False)

        return {"status": "success", "scored_path": scored_path}

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/agent3/aggregate")
async def aggregate(scored_path: str = Form(...)):
    """
    Aggregate results into final biomarker scores.
    """
    try:
        df = pd.read_csv(scored_path)
        df["Final_Score"] = df["Confidence"] * 100  # Simple weighted example
        df_sorted = df.sort_values(by="Final_Score", ascending=False)

        result_path = os.path.join(DATA_DIR, "biomarker_results.csv")
        df_sorted.to_csv(result_path, index=False)

        return {"status": "success", "results_path": result_path}

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
