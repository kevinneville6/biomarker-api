from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import io
import glob
import datetime

app = FastAPI(title="Biomarker Analysis API")

MODEL_PATH = "biomarker_model.pkl"
FEATURES_PATH = "feature_columns.pkl"
DATA_DIR = "data_outputs"
os.makedirs(DATA_DIR, exist_ok=True)

# Load model + features
biomarker_model = None
feature_columns = None

@app.on_event("startup")
def load_model():
    global biomarker_model, feature_columns
    try:
        biomarker_model = joblib.load(MODEL_PATH)
        feature_columns = joblib.load(FEATURES_PATH)
        print("✅ Model and features loaded successfully.")
    except Exception as e:
        print(f"⚠️ Warning: Could not load model or features: {e}")


@app.get("/")
async def root():
    return {
        "message": "Biomarker API is running 🚀",
        "docs": "/docs",
        "endpoints": [
            "/agent1/process",
            "/agent1/process_json",
            "/agent2/predict",
            "/agent2/predict_json",
            "/agent3/aggregate",
            "/agent3/aggregate_json",
            "/download/features",
            "/download/scored",
            "/download/results"
        ]
    }


# --------------------------
# Agent 1 – Preprocessing
# --------------------------

@app.post("/agent1/process")
async def process_data(file: UploadFile = File(...)):
    """Upload raw CSV → align with features → save features.csv"""
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        return await save_features(df)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


class DataInput(BaseModel):
    data: list[dict]  # JSON list of records


@app.post("/agent1/process_json")
async def process_data_json(input: DataInput):
    """Upload JSON → align with features → save features.csv"""
    try:
        df = pd.DataFrame(input.data)
        return await save_features(df)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def save_features(df: pd.DataFrame):
    if feature_columns is None:
        return JSONResponse({"status": "error", "message": "Feature columns not loaded"}, status_code=500)

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    features_path = os.path.join(DATA_DIR, f"features_{timestamp}.csv")
    df.to_csv(features_path, index=False)

    return {"status": "success", "features_path": features_path}


# --------------------------
# Agent 2 – Prediction
# --------------------------

@app.post("/agent2/predict")
async def predict():
    """Run ML predictions on last saved CSV"""
    return await run_prediction()


@app.post("/agent2/predict_json")
async def predict_json():
    """Run ML predictions and return results in JSON"""
    return await run_prediction(json_output=True)


async def run_prediction(json_output=False):
    try:
        if biomarker_model is None:
            return JSONResponse({"status": "error", "message": "Model not loaded"}, status_code=500)

        files = sorted(glob.glob(os.path.join(DATA_DIR, "features_*.csv")))
        if not files:
            return JSONResponse({"status": "error", "message": "No features file found. Run Agent1 first."}, status_code=400)

        df = pd.read_csv(files[-1])
        preds = biomarker_model.predict(df)
        conf = biomarker_model.predict_proba(df)[:, 1] if hasattr(biomarker_model, "predict_proba") else [None] * len(preds)

        result_df = pd.DataFrame({
            "Biomarker": df.index,
            "Predicted_Class": preds,
            "Confidence": conf
        })

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        scored_path = os.path.join(DATA_DIR, f"scored_predictions_{timestamp}.csv")
        result_df.to_csv(scored_path, index=False)

        if json_output:
            return {"status": "success", "predictions": result_df.to_dict(orient="records")}
        return {"status": "success", "scored_path": scored_path}

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# --------------------------
# Agent 3 – Aggregation
# --------------------------

@app.post("/agent3/aggregate")
async def aggregate():
    """Aggregate results → save biomarker_results.csv"""
    return await run_aggregation()


@app.post("/agent3/aggregate_json")
async def aggregate_json():
    """Aggregate results and return JSON"""
    return await run_aggregation(json_output=True)


async def run_aggregation(json_output=False):
    try:
        files = sorted(glob.glob(os.path.join(DATA_DIR, "scored_predictions_*.csv")))
        if not files:
            return JSONResponse({"status": "error", "message": "No scored predictions found. Run Agent2 first."}, status_code=400)

        df = pd.read_csv(files[-1])
        df["Final_Score"] = df["Confidence"] * 100
        df_sorted = df.sort_values(by="Final_Score", ascending=False)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = os.path.join(DATA_DIR, f"biomarker_results_{timestamp}.csv")
        df_sorted.to_csv(result_path, index=False)

        if json_output:
            return {"status": "success", "results": df_sorted.to_dict(orient="records")}
        return {"status": "success", "results_path": result_path}

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# --------------------------
# Download Endpoints
# --------------------------

@app.get("/download/features")
async def download_features():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "features_*.csv")))
    if not files:
        return JSONResponse({"status": "error", "message": "No features file found"}, status_code=400)
    return FileResponse(files[-1], filename="features.csv", media_type="text/csv")


@app.get("/download/scored")
async def download_scored():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "scored_predictions_*.csv")))
    if not files:
        return JSONResponse({"status": "error", "message": "No scored predictions found"}, status_code=400)
    return FileResponse(files[-1], filename="scored_predictions.csv", media_type="text/csv")


@app.get("/download/results")
async def download_results():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "biomarker_results_*.csv")))
    if not files:
        return JSONResponse({"status": "error", "message": "No results file found"}, status_code=400)
    return FileResponse(files[-1], filename="biomarker_results.csv", media_type="text/csv")


# --------------------------
# Run locally / Render
# --------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=port)