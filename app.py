from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
import io
import os
import logging

app = Flask(__name__)
CORS(app)

# Production logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "tb_model_final_v3": {
        "file": "tb_model_final_v3.keras",
        "label": "Model 1",
        "input_size": (224, 224),
        "color_mode": "rgb",
        "preprocess": "efficientnet",
    },
    "best_model_epoch": {
        "file": "best_model_epoch.keras",
        "label": "Model 2",
        "input_size": (64, 64),
        "color_mode": "grayscale",
        "preprocess": "scale_01",
    },
    "MMADUs_Tuberculosis_Detection": {
        "file": "MMADUs_Tuberculosis_Detection.keras",
        "label": "Model 3",
        "input_size": (250, 250),
        "color_mode": "grayscale",
        "preprocess": "scale_01",
    },
}

DEFAULT_MODEL_ID = "tb_model_final_v3"
MODEL_CACHE = {}


def get_model(model_id: str):
    entry = MODEL_REGISTRY.get(model_id)
    if not entry:
        raise ValueError(f"Unknown model: {model_id}")

    if model_id not in MODEL_CACHE:
        model_path = entry["file"]

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info(f"Loading model: {model_id}")
        MODEL_CACHE[model_id] = tf.keras.models.load_model(model_path)

    return MODEL_CACHE[model_id]


def prepare_image(img_bytes: io.BytesIO, model_id: str):
    entry = MODEL_REGISTRY.get(model_id)
    if not entry:
        raise ValueError(f"Unknown model: {model_id}")

    img = load_img(
        img_bytes,
        target_size=entry["input_size"],
        color_mode=entry["color_mode"],
    )

    x = img_to_array(img)

    if entry["preprocess"] == "efficientnet":
        x = preprocess_input(x)
    elif entry["preprocess"] == "scale_01":
        x = x / 255.0
    else:
        raise ValueError(f"Unsupported preprocess type: {entry['preprocess']}")

    x = np.expand_dims(x, axis=0)
    return x


def classify_score(pred: float):
    if pred < 0.11:
        return {
            "label": "Normal",
            "decision": "Normal",
            "confidence": "High",
        }

    if pred < 0.26:
        return {
            "label": "Borderline - Likely Normal",
            "decision": "Borderline",
            "confidence": "Low",
            "warning": "Manual review recommended",
        }

    if pred < 0.35:
        return {
            "label": "Borderline - Likely TB",
            "decision": "Borderline",
            "confidence": "Low",
            "warning": "Manual review required",
        }

    return {
        "label": "Tuberculosis",
        "decision": "TB",
        "confidence": "High",
    }


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "service": "TB AI API",
        "models_available": len(MODEL_REGISTRY),
    })


@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({
        "status": "connected",
        "model_count": len(MODEL_REGISTRY),
    })


@app.route("/models", methods=["GET"])
def list_models():
    models = [
        {
            "id": model_id,
            "label": entry["label"],
        }
        for model_id, entry in MODEL_REGISTRY.items()
    ]

    return jsonify({
        "models": models,
        "default": DEFAULT_MODEL_ID,
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({
                "error": "No image file provided"
            }), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({
                "error": "Empty file selected"
            }), 400

        img_data = file.read()

        if not img_data:
            return jsonify({
                "error": "Uploaded file is empty"
            }), 400

        model_results = []

        for model_id in MODEL_REGISTRY.keys():
            x = prepare_image(io.BytesIO(img_data), model_id)

            model = get_model(model_id)

            pred = float(model.predict(x, verbose=0)[0][0])

            classification = classify_score(pred)

            model_results.append({
                "id": model_id,
                "label": MODEL_REGISTRY[model_id]["label"],
                "score": pred,
                "decision": classification["decision"],
                "detail": classification["label"],
                "confidence": classification["confidence"],
                "warning": classification.get("warning"),
            })

        decision_counts = {
            "Normal": 0,
            "TB": 0,
            "Borderline": 0,
        }

        for result in model_results:
            decision_counts[result["decision"]] += 1

        max_count = max(decision_counts.values())

        majority_decisions = [
            decision
            for decision, count in decision_counts.items()
            if count == max_count
        ]

        if max_count == len(model_results):
            final_decision = majority_decisions[0]

            verdict = {
                "label": (
                    "CONFIRMED: Normal"
                    if final_decision == "Normal"
                    else "CONFIRMED: Tuberculosis"
                ),
                "confidence": "High",
                "status": final_decision,
            }

        elif len(majority_decisions) == 1 and majority_decisions[0] in {"Normal", "TB"}:
            final_decision = majority_decisions[0]

            verdict = {
                "label": (
                    "LIKELY: Normal"
                    if final_decision == "Normal"
                    else "LIKELY: Tuberculosis"
                ),
                "confidence": "Medium",
                "status": final_decision,
                "warning": "Majority agreement, but not unanimous. Manual review recommended.",
            }

        else:
            verdict = {
                "label": "UNCERTAIN: Mixed Signals",
                "confidence": "Low",
                "status": "Uncertain",
                "warning": "Model outputs disagree or are borderline. Manual review required.",
            }

        return jsonify({
            "label": verdict["label"],
            "confidence": verdict["confidence"],
            "status": verdict["status"],
            "warning": verdict.get("warning"),
            "models": model_results,
        })

    except Exception as e:
        logger.exception("Prediction error")

        return jsonify({
            "error": "Internal server error",
            "detail": str(e),
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    logger.info(f"🚀 Backend running on port {port}")

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )