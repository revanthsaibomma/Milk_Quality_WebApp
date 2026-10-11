from pathlib import Path
import joblib
import numpy as np
from flask import Flask, render_template, request

from milk_logic import build_report

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "milk_model.joblib"

app = Flask(__name__)
app.config["SECRET_KEY"] = "change-this-before-production"

model_bundle = None
model_error = None
if MODEL_PATH.exists():
    try:
        model_bundle = joblib.load(MODEL_PATH)
    except Exception as exc:
        model_error = f"Could not load model file: {exc}"
else:
    model_error = "Model is not trained yet. Put your milk grading CSV in this folder and run: python train_model.py"


@app.get("/")
def index():
    return render_template("index.html", model_ready=model_bundle is not None,
                           model_error=model_error, result=None, form={})


@app.post("/predict")
def predict():
    if model_bundle is None:
        return render_template("index.html", model_ready=False,
                               model_error=model_error, result=None,
                               form=request.form), 503
    try:
        form = request.form
        raw = {
            "pH": float(form["pH"]),
            "Temprature": float(form["Temprature"]),
            "Taste": int(form["Taste"]),
            "Odor": int(form["Odor"]),
            "Fat": int(form["Fat"]),
            "Turbidity": int(form["Turbidity"]),
            "Colour": float(form["Colour"]),
        }
        storage_temp = float(form["storage_temp"])
        elapsed = float(form["elapsed_hrs"])
        milk_type = form["milk_type"]
        consumer_group = form["consumer_group"]

        if not (0 <= raw["pH"] <= 14):
            raise ValueError("pH must be between 0 and 14.")
        if not (-5 <= raw["Temprature"] <= 100 and -5 <= storage_temp <= 100):
            raise ValueError("Temperature values must be between -5°C and 100°C.")
        if not (0 <= raw["Colour"] <= 255):
            raise ValueError("Colour must be between 0 and 255.")
        if elapsed < 0:
            raise ValueError("Elapsed storage time cannot be negative.")

        feature_names = model_bundle["feature_names"]
        scaler = model_bundle["scaler"]
        model = model_bundle["model"]
        x = engineer_features(raw, feature_names)
        x_scaled = scaler.transform(x)
        grade_idx = int(model.predict(x_scaled)[0])
        class_map = model_bundle["class_map"]
        grade = class_map[grade_idx]
        probs = model.predict_proba(x_scaled)[0]
        probabilities = {class_map[int(c)]: float(probs[list(model.classes_).index(c)])
                         for c in model.classes_}
        result = build_report(
            grade=grade,
            confidence=float(max(probabilities.values())),
            probabilities=probabilities,
            storage_temp_c=storage_temp,
            elapsed_hrs=elapsed,
            milk_type=milk_type,
            consumer_group=consumer_group,
            raw=raw,
        )
        return render_template("index.html", model_ready=True, model_error=None,
                               result=result, form=form)
    except (ValueError, KeyError) as exc:
        return render_template("index.html", model_ready=True, model_error=None,
                               result=None, form=request.form,
                               validation_error=str(exc)), 400
    except Exception as exc:
        app.logger.exception("Prediction failed")
        return render_template("index.html", model_ready=True, model_error=None,
                               result=None, form=request.form,
                               validation_error=f"Prediction failed: {exc}"), 500


def engineer_features(raw, feature_names):
    ph = raw["pH"]
    temp = raw["Temprature"]
    turb = raw["Turbidity"]
    taste = raw["Taste"]
    odor = raw["Odor"]
    colour = raw["Colour"]
    row = dict(raw)
    row.update({
        "pH_deviation": abs(ph - 6.7),
        "temp_risk": int(temp > 40),
        "sensory_score": taste + odor + (1 - turb),
        "colour_norm": colour / 255.0,
        "pH_colour": abs(ph - 6.7) * (colour / 255.0),
        "turb_temp": turb * int(temp > 40),
    })
    return np.asarray([[row.get(name, 0) for name in feature_names]], dtype=np.float32)


if __name__ == "__main__":
    app.run(debug=True)
