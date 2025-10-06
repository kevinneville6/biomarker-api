from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
import os
import io
import datetime

app = FastAPI(title="Biomarker Analysis API")

MODEL_PATH = "biomarker_model.pkl"
FEATURES_PATH = "feature_columns.pkl"

# Load model + features globally
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
            "/agent2/predict",
            "/agent3/aggregate"
        ]
    }


@app.post("/agent1/process")
async def process_data(file: UploadFile = File(...)):
    """Upload raw CSV → align with features → return JSON"""
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        if feature_columns is None:
            return JSONResponse({"status": "error", "message": "Feature columns not loaded"}, status_code=500)

        # Align schema
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_columns]

        return {"status": "success", "features": df.to_dict(orient="records")}

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/agent2/predict")
async def predict(features: dict):
    """Run ML predictions on JSON features → return JSON"""
    try:
        if biomarker_model is None:
            return JSONResponse({"status": "error", "message": "Model not loaded"}, status_code=500)

        df = pd.DataFrame(features["features"])

        preds = biomarker_model.predict(df)
        conf = biomarker_model.predict_proba(df)[:, 1] if hasattr(biomarker_model, "predict_proba") else [None] * len(preds)

        result_df = pd.DataFrame({
            "Biomarker": df.index,
            "Predicted_Class": preds,
            "Confidence": conf
        })

        return {"status": "success", "predictions": result_df.to_dict(orient="records")}

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/agent3/aggregate")
async def aggregate(predictions: dict):
    """Aggregate predictions → return JSON"""
    try:
        df = pd.DataFrame(predictions["predictions"])

        df["Final_Score"] = df["Confidence"] * 100
        df_sorted = df.sort_values(by="Final_Score", ascending=False)

        return {
            "status": "success",
            "results": df_sorted.to_dict(orient="records")
        }

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# -------------------------------------------------
# Run locally or on Render with dynamic PORT
# -------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=port)