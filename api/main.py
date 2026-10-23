from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from joblib import load

app = FastAPI(
    title="Drug Review Benchmark API",
    description="Research-only prediction of the dataset condition label from review text.",
    version="0.1.0",
)


class PredictionRequest(BaseModel):
    text: str = Field(min_length=3, max_length=20000)


def resolve_latest_model():
    pointer = Path("artifacts/latest")
    if not pointer.exists():
        return None
    run_dir = Path(pointer.read_text(encoding="utf-8").strip())
    metadata_path = run_dir / "metrics.json"
    if not metadata_path.exists():
        return None
    import json
    info = json.loads(metadata_path.read_text(encoding="utf-8"))
    name = info.get("best_model_by_macro_f1")
    candidate = run_dir / f"{name}.joblib" if name else None
    return candidate if candidate and candidate.exists() else None


@app.get("/health")
def health():
    return {"status": "ok", "model_available": resolve_latest_model() is not None}


@app.post("/predict")
def predict(request: PredictionRequest):
    model_path = resolve_latest_model()
    if model_path is None:
        raise HTTPException(status_code=503, detail="No trained model found. Run `drugbench run` first.")
    model = load(model_path)
    prediction = str(model.predict([request.text])[0])
    response = {"predicted_condition": prediction, "model": model_path.stem}
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([request.text])[0]
        response["class_probabilities"] = {
            str(cls): float(prob) for cls, prob in zip(model.classes_, probabilities)
        }
    return response
