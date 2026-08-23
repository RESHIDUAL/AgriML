import os
import sys
import time
import pickle
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_file

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.disease.disease_predictor import DiseasePredictor
from src.disease.disease_info import get_disease_info

app = Flask(__name__, template_folder="templates")

CROP_MODEL_PATH = os.path.join(BASE_DIR, "weights", "crop_model_final.pkl")
if not os.path.exists(CROP_MODEL_PATH):
    CROP_MODEL_PATH = os.path.join(BASE_DIR, "src", "crop", "crop_model_final.pkl")

crop_model = None
crop_scaler = None
crop_encoder = None
crop_features = []

if os.path.exists(CROP_MODEL_PATH):
    with open(CROP_MODEL_PATH, "rb") as f:
        crop_data = pickle.load(f)
        crop_model = crop_data.get("model")
        crop_scaler = crop_data.get("scaler")
        crop_encoder = crop_data.get("label_encoder")
        crop_features = crop_data.get("feature_cols", ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])

DISEASE_WEIGHTS_ORDER = [
    os.path.join(BASE_DIR, "weights", "leaf_disease_model_final.pth"),
    os.path.join(BASE_DIR, "weights", "plantdoc_fold1.pth"),
    os.path.join(BASE_DIR, "weights", "plantvillage_pretrained.pth"),
]

disease_weights = next((p for p in DISEASE_WEIGHTS_ORDER if os.path.exists(p)), None)
disease_predictor = None

if disease_weights:
    try:
        disease_predictor = DiseasePredictor(model_path=disease_weights, backbone="mobilenet_v3_large")
    except Exception as e:
        print(f"Notice: Disease model initialization deferred: {e}")

latest_sensor_reading = {"status": "waiting", "message": "No sensor readings yet."}
last_esp_timestamp = None
ESP_TIMEOUT_SEC = 15


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/api/recommend_crop", methods=["POST"])
def api_recommend_crop():
    global latest_sensor_reading, last_esp_timestamp
    data = request.get_json(force=True)

    n = data.get("N", data.get("n"))
    p = data.get("P", data.get("p"))
    k = data.get("K", data.get("k"))
    ph = data.get("pH", data.get("ph"))

    if None in (n, p, k, ph):
        return jsonify({"status": "error", "error": "Missing required fields: N, P, K, pH"}), 400

    temp = float(data.get("temperature", 25.0))
    hum = float(data.get("humidity", 70.0))
    rain = float(data.get("rainfall", 150.0))

    if crop_model is None or crop_scaler is None or crop_encoder is None:
        return jsonify({"status": "error", "error": "Crop model is not loaded."}), 500

    X_in = crop_scaler.transform([[n, p, k, temp, hum, ph, rain]])
    pred_idx = crop_model.predict(X_in)[0]
    probs = crop_model.predict_proba(X_in)[0]

    top3_indices = np.argsort(probs)[::-1][:3]
    top3 = [
        {
            "crop": crop_encoder.inverse_transform([i])[0],
            "confidence": round(float(probs[i]) * 100, 1),
        }
        for i in top3_indices
    ]

    res = {
        "status": "ok",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "recommended_crop": crop_encoder.inverse_transform([pred_idx])[0],
        "top_3": top3,
        "inputs": {
            "N": n, "P": p, "K": k, "pH": ph,
            "temperature": temp, "humidity": hum, "rainfall": rain
        }
    }

    if not data.get("manual_override", False):
        last_esp_timestamp = time.time()
        latest_sensor_reading.update(res)

    return jsonify(res)


@app.route("/recommend", methods=["POST"])
def esp_recommend():
    return api_recommend_crop()


@app.route("/api/predict_disease", methods=["POST"])
def api_predict_disease():
    global disease_predictor
    if "image" not in request.files:
        return jsonify({"status": "error", "error": "No image file provided."}), 400

    img_file = request.files["image"]
    if img_file.filename == "":
        return jsonify({"status": "error", "error": "Empty filename."}), 400

    if disease_predictor is None:
        active_weights = next((p for p in DISEASE_WEIGHTS_ORDER if os.path.exists(p)), None)
        if active_weights:
            try:
                disease_predictor = DiseasePredictor(model_path=active_weights)
            except Exception as e:
                return jsonify({"status": "error", "error": f"Failed to load disease model: {e}"}), 500
        else:
            return jsonify({"status": "error", "error": "Disease model weights not found. Please train Stage 1 and Stage 2 first."}), 500

    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        img_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = disease_predictor.predict(tmp_path, top_k=3)
        result["status"] = "ok"
    except Exception as e:
        result = {"status": "error", "error": str(e)}
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return jsonify(result)


@app.route("/api/sample_image", methods=["GET"])
def api_sample_image():
    cls_name = request.args.get("class", "Tomato leaf late blight")
    sample_dir = os.path.join(BASE_DIR, "PlantDoc-Dataset", "test", cls_name)

    if not os.path.exists(sample_dir):
        sample_dir = os.path.join(BASE_DIR, "PlantDoc-Dataset", "train", cls_name)

    if os.path.exists(sample_dir):
        imgs = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        if imgs:
            return send_file(imgs[0], mimetype="image/jpeg")

    return jsonify({"error": "Sample not found"}), 404


@app.route("/latest", methods=["GET"])
def latest():
    return jsonify(latest_sensor_reading)


@app.route("/status", methods=["GET"])
def server_status():
    connected = False
    secs = None
    if last_esp_timestamp is not None:
        secs = time.time() - last_esp_timestamp
        connected = secs < ESP_TIMEOUT_SEC

    return jsonify({
        "status": "online",
        "esp_connected": connected,
        "seconds_since_last_reading": round(secs, 1) if secs else None,
        "disease_model_loaded": disease_predictor is not None,
        "crop_model_loaded": crop_model is not None,
    })


if __name__ == "__main__":
    print("Starting AgriML Live Agricultural Platform on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
