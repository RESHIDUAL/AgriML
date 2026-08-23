from flask import Flask, request, jsonify, render_template_string
from datetime import datetime
import tkinter as tk
import threading
import requests
import pandas as pd
import numpy as np
import pickle
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# ---------------------------------------------------------------------------
# CONFIG -- edit these
# ---------------------------------------------------------------------------
OPENWEATHER_API_KEY = os.environ.get('OPENWEATHER_API_KEY', '')
DATASET_URL = "https://raw.githubusercontent.com/AbhishekKandoi/Crop-Yield-Prediction-based-on-Indian-Agriculture/main/Crop%20Recommendation%20dataset.csv"
import sys
import os

# --------------------------------------------------------------
# PyInstaller Path Fix: Look in the hidden temp folder if frozen
# --------------------------------------------------------------
if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

candidate_paths = [
    os.path.join(base_path, 'crop_model_final.pkl'),
    os.path.abspath(os.path.join(base_path, '..', '..', 'weights', 'crop_model_final.pkl')),
    os.path.abspath(os.path.join(base_path, '..', '..', 'crop_model_final.pkl')),
]
MODEL_PKL = next((p for p in candidate_paths if os.path.exists(p)), candidate_paths[0])

# How long (seconds) after the last ESP32 POST before we call it "idle".
# ESP32 sends every SEND_INTERVAL_MS (5000ms in the sketch) -> give it 3x margin.
ESP_TIMEOUT_SEC = 15

# In-memory store of the most recent reading, shown on the dashboard.
latest_reading = {"status": "waiting", "message": "No reading received yet."}

# Timestamp (epoch seconds) of the last time the ESP32 actually POSTed
# (only set inside /recommend when manual_override is NOT used, so manual
# GUI testing never fakes the "connected" indicator).
last_esp_seen_ts = None

# Global model accuracy and class count (set during training/loading)
model_accuracy = 0.0
n_classes = 0

DASHBOARD_HTML_ORIGINAL = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Smart Crop Rod - Live Dashboard</title>
<meta name="description" content="Real-time soil sensor monitoring and ML-powered crop recommendation system">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:          #070d0b;
    --surface:     rgba(255,255,255,0.035);
    --surface-2:   rgba(255,255,255,0.025);
    --border:      rgba(255,255,255,0.08);
    --border-acc:  rgba(0,196,154,0.28);
    --teal:        #00c49a;
    --teal-lo:     rgba(0,196,154,0.12);
    --teal-mid:    rgba(0,196,154,0.25);
    --amber:       #f0a832;
    --amber-lo:    rgba(240,168,50,0.12);
    --amber-brd:   rgba(240,168,50,0.28);
    --red:         #ff5f5f;
    --text:        #ddeee9;
    --text-sec:    #7a9590;
    --text-dim:    #405550;
    --mono:        'JetBrains Mono', monospace;
    --sans:        'Inter', sans-serif;
    --r-sm:        8px;
    --r-md:        12px;
    --r-lg:        16px;
    --r-xl:        22px;
    --ease:        cubic-bezier(0.4,0,0.2,1);
  }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: var(--sans);
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    background-image:
      radial-gradient(ellipse 90% 55% at 50% -15%, rgba(0,196,154,0.07) 0%, transparent 65%),
      radial-gradient(ellipse 55% 40% at 85% 85%,  rgba(0,130,100,0.04) 0%, transparent 55%);
  }

  .wrap { max-width: 1080px; margin: 0 auto; padding: 36px 22px 80px; }

  .hdr {
    display: flex; align-items: flex-start; justify-content: space-between;
    flex-wrap: wrap; gap: 16px;
    padding-bottom: 24px; margin-bottom: 32px;
    border-bottom: 1px solid var(--border);
  }
  .hdr-title { font-size: 17px; font-weight: 600; letter-spacing: 0.01em; }
  .hdr-sub   { font-family: var(--mono); font-size: 11px; color: var(--text-sec); margin-top: 4px; }

  .hdr-right { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }

  .status-badge {
    display: flex; align-items: center; gap: 7px;
    font-family: var(--mono); font-size: 11px; color: var(--text-sec);
  }
  .dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--teal); flex-shrink: 0;
  }
  .dot.live  { animation: pulse-dot 2s ease infinite; }
  .dot.error { background: var(--red); animation: none; }
  .dot.idle  { background: var(--text-dim); animation: none; }
  @keyframes pulse-dot {
    0%,100% { box-shadow: 0 0 0 0 rgba(0,196,154,0.5); }
    50%     { box-shadow: 0 0 0 7px rgba(0,196,154,0); }
  }

  #btn-refresh {
    display: flex; align-items: center; gap: 8px;
    padding: 9px 18px;
    background: var(--teal-lo);
    border: 1px solid var(--border-acc);
    border-radius: var(--r-sm);
    color: var(--teal);
    font-family: var(--sans); font-size: 13px; font-weight: 500;
    cursor: pointer; letter-spacing: 0.01em;
    transition: background 0.2s var(--ease), box-shadow 0.2s var(--ease), transform 0.15s var(--ease);
  }
  #btn-refresh:hover:not(:disabled) {
    background: var(--teal-mid);
    box-shadow: 0 0 18px rgba(0,196,154,0.22);
    transform: translateY(-1px);
  }
  #btn-refresh:active { transform: translateY(0); }
  #btn-refresh:disabled { opacity: 0.55; cursor: not-allowed; }
  #btn-refresh.loading .ico-refresh { animation: spin 0.75s linear infinite; }
  .ico-refresh { width: 14px; height: 14px; flex-shrink: 0; }
  @keyframes spin { to { transform: rotate(360deg); } }

  .lbl {
    font-family: var(--mono); font-size: 10px; letter-spacing: 0.11em;
    text-transform: uppercase; color: var(--teal); margin-bottom: 12px;
  }

  .hero {
    position: relative; overflow: hidden;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--r-xl);
    padding: 30px; margin-bottom: 20px;
    backdrop-filter: blur(20px);
    box-shadow: 0 6px 32px rgba(0,0,0,0.45), 0 0 40px rgba(0,196,154,0.07);
    animation: fadeUp 0.45s var(--ease) both;
  }
  .hero::before {
    content: ''; position: absolute; inset: 0 0 auto 0; height: 1px;
    background: linear-gradient(90deg, transparent 10%, var(--teal) 50%, transparent 90%);
    opacity: 0.45;
  }

  .hero-body {
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 20px; align-items: start;
  }

  .crop-eyebrow { font-family: var(--mono); font-size: 10px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--teal); margin-bottom: 8px; }

  .crop-name {
    font-size: 54px; font-weight: 700; line-height: 1; text-transform: capitalize;
    background: linear-gradient(130deg, var(--text) 30%, var(--teal) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    margin-bottom: 14px;
  }

  .conf-pill {
    display: inline-flex; align-items: center; gap: 7px;
    padding: 6px 15px;
    background: var(--amber-lo); border: 1px solid var(--amber-brd);
    border-radius: 20px;
    font-family: var(--mono); font-size: 13px; color: var(--amber);
  }

  .top3 { margin-top: 26px; display: flex; flex-direction: column; gap: 11px; }
  .t3-row {
    display: grid;
    grid-template-columns: 18px 1fr auto;
    gap: 13px; align-items: center;
  }
  .t3-rank { font-family: var(--mono); font-size: 11px; color: var(--text-dim); }
  .t3-name { font-size: 13px; font-weight: 500; text-transform: capitalize; margin-bottom: 5px; }
  .bar-track { height: 5px; background: rgba(255,255,255,0.05); border-radius: 3px; overflow: hidden; }
  .bar-fill  { height: 100%; border-radius: 3px; transition: width 0.55s var(--ease); }
  .bf-1 { background: linear-gradient(90deg, #007a61, var(--teal)); }
  .bf-2 { background: rgba(0,196,154,0.38); }
  .bf-3 { background: rgba(0,196,154,0.18); }
  .t3-pct { font-family: var(--mono); font-size: 12px; color: var(--text-sec); min-width: 42px; text-align: right; }

  .grid-2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 18px; margin-bottom: 18px;
  }

  .panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--r-lg);
    padding: 22px;
    backdrop-filter: blur(20px);
    box-shadow: 0 4px 22px rgba(0,0,0,0.38);
    animation: fadeUp 0.45s var(--ease) both;
  }
  .panel:nth-child(2) { animation-delay: 0.07s; }

  .s-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(108px, 1fr));
    gap: 12px;
  }
  .s-card {
    padding: 14px 14px 16px;
    background: var(--surface-2);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: var(--r-md);
    transition: border-color 0.2s, background 0.2s;
    cursor: default;
  }
  .s-card:hover {
    border-color: var(--border-acc);
    background: var(--teal-lo);
  }
  .s-label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-sec); margin-bottom: 8px; }
  .s-val   { font-family: var(--mono); font-size: 24px; font-weight: 500; color: var(--text); line-height: 1; }
  .s-unit  { font-size: 11px; color: var(--text-dim); margin-left: 2px; }
  .s-na    { color: var(--text-dim); }

  .model-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--r-lg);
    padding: 22px 28px;
    backdrop-filter: blur(20px);
    box-shadow: 0 4px 22px rgba(0,0,0,0.38);
    animation: fadeUp 0.45s var(--ease) 0.14s both;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 20px 24px;
    align-items: center;
  }
  .m-stat .m-lbl  { font-size: 10px; text-transform: uppercase; letter-spacing: 0.07em; color: var(--text-sec); margin-bottom: 5px; }
  .m-stat .m-val  { font-family: var(--mono); font-size: 19px; font-weight: 500; color: var(--teal); line-height: 1.1; }
  .m-stat .m-sub  { font-size: 11px; color: var(--text-dim); margin-top: 3px; }

  .src-badge {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 3px 10px; border-radius: 10px;
    font-family: var(--mono); font-size: 10px; letter-spacing: 0.04em;
  }
  .src-badge.sensor { background: var(--teal-lo);  color: var(--teal);  border: 1px solid rgba(0,196,154,0.22); }
  .src-badge.api    { background: var(--amber-lo); color: var(--amber); border: 1px solid var(--amber-brd); }
  .src-badge.manual { background: rgba(120,120,255,0.10); color: #9aa0ff; border: 1px solid rgba(120,120,255,0.25); }

  .ts-strip {
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 8px;
    margin-top: 14px; padding-top: 14px;
    border-top: 1px solid var(--border);
  }
  .ts-strip .ts-txt { font-family: var(--mono); font-size: 11px; color: var(--text-dim); }

  .empty {
    text-align: center; padding: 88px 20px;
    animation: fadeUp 0.45s var(--ease) both;
  }
  .empty-title { font-size: 19px; font-weight: 600; margin-bottom: 10px; }
  .empty-desc  { font-size: 13px; color: var(--text-sec); line-height: 1.65; max-width: 380px; margin: 0 auto; }
  .empty-code  {
    display: inline-block; margin-top: 22px;
    font-family: var(--mono); font-size: 12px; color: var(--teal);
    background: var(--teal-lo); border: 1px solid var(--border-acc);
    border-radius: var(--r-sm); padding: 11px 22px;
  }

  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  @media (max-width: 700px) {
    .grid-2 { grid-template-columns: 1fr; }
    .hero-body { grid-template-columns: 1fr; }
    .crop-name { font-size: 40px; }
  }
  @media (max-width: 480px) {
    .crop-name { font-size: 32px; }
    .model-panel { grid-template-columns: 1fr 1fr; }
  }
</style>
</head>
<body>
<div class="wrap">

  <header class="hdr">
    <div>
      <div class="hdr-title">Smart Crop Rod</div>
      <div class="hdr-sub">ML-Powered Real-Time Crop Recommendation</div>
    </div>
    <div class="hdr-right">
      <div class="status-badge">
        <span class="dot idle" id="dot"></span>
        <span id="status-txt">connecting</span>
      </div>
      <button id="btn-refresh" onclick="manualRefresh()">
        <svg class="ico-refresh" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
          <path d="M13.5 8A5.5 5.5 0 1 1 8 2.5c1.8 0 3.4.87 4.4 2.2"/>
          <polyline points="13.5 2.5 13.5 5 11 5"/>
        </svg>
        Get New Reading
      </button>
    </div>
  </header>

  <main id="content">
    <div class="empty">
      <div class="empty-title">Waiting for sensor data</div>
      <div class="empty-desc">Ensure the ESP32 is powered, connected to WiFi, and pointed at this server's /recommend endpoint.</div>
      <div class="empty-code">ESP32 &rarr; POST /recommend &rarr; Dashboard</div>
    </div>
  </main>

</div>

<script>
var cachedKey = '';

function fmt(val, dp) {
  if (val === null || val === undefined) return null;
  return parseFloat(val).toFixed(dp === undefined ? 1 : dp);
}

function sCard(label, val, unit, dp) {
  var v = fmt(val, dp);
  if (v === null) return '';
  return '<div class="s-card"><div class="s-label">' + label + '</div>'
       + '<div class="s-val">' + v + '<span class="s-unit">' + unit + '</span></div></div>';
}

function render(data, info) {
  var s = data.sensor_input;
  var w = data.weather_used;
  var top3 = data.top_3;

  var barsHtml = top3.map(function(c, i) {
    return '<div class="t3-row">'
         + '<div class="t3-rank">' + (i+1) + '</div>'
         + '<div>'
         +   '<div class="t3-name">' + c.crop + '</div>'
         +   '<div class="bar-track"><div class="bar-fill bf-' + (i+1) + '" style="width:' + c.confidence + '%"></div></div>'
         + '</div>'
         + '<div class="t3-pct">' + c.confidence + '%</div>'
         + '</div>';
  }).join('');

  var srcBadge = w.source === 'sensor'
    ? '<span class="src-badge sensor">DHT11 Sensor</span>'
    : (w.source === 'manual'
      ? '<span class="src-badge manual">Manual Entry</span>'
      : '<span class="src-badge api">Weather API Fallback</span>');

  var acc = info && info.accuracy ? (info.accuracy * 100).toFixed(1) + '%' : '-';
  var cls = info && info.n_classes ? info.n_classes : '-';

  document.getElementById('content').innerHTML =
    '<div class="lbl">Crop Recommendation</div>'
  + '<div class="hero">'
  +   '<div class="hero-body">'
  +     '<div>'
  +       '<div class="crop-eyebrow">Recommended Crop</div>'
  +       '<div class="crop-name">' + data.recommended_crop + '</div>'
  +       '<div class="conf-pill">' + top3[0].confidence + '% Confidence</div>'
  +     '</div>'
  +   '</div>'
  +   '<div class="top3">' + barsHtml + '</div>'
  +   '<div class="ts-strip">'
  +     '<span class="ts-txt">Last reading &nbsp;/&nbsp; ' + data.timestamp + '</span>'
  +     srcBadge
  +   '</div>'
  + '</div>'

  + '<div class="grid-2">'
  +   '<div class="panel">'
  +     '<div class="lbl">Soil Readings</div>'
  +     '<div class="s-grid">'
  +       sCard('Nitrogen', s.N, ' mg/kg', 0)
  +       sCard('Phosphorus', s.P, ' mg/kg', 0)
  +       sCard('Potassium', s.K, ' mg/kg', 0)
  +       sCard('pH Level', s.pH, '', 1)
  +       sCard('Moisture', s.soil_moisture_pct, '%', 1)
  +       sCard('Soil Temp', s.soil_temp_c, '°C', 1)
  +     '</div>'
  +   '</div>'
  +   '<div class="panel">'
  +     '<div class="lbl">Environment</div>'
  +     '<div class="s-grid">'
  +       sCard('Temperature', w.temperature, '°C', 1)
  +       sCard('Humidity', w.humidity, '%', 0)
  +       sCard('Rainfall', w.rainfall_forecast_mm, ' mm', 1)
  +     '</div>'
  +   '</div>'
  + '</div>'

  + '<div class="lbl">Model Information</div>'
  + '<div class="model-panel">'
  +   '<div class="m-stat"><div class="m-lbl">Algorithm</div><div class="m-val" style="font-size:15px;">Random Forest</div><div class="m-sub">Ensemble classifier</div></div>'
  +   '<div class="m-stat"><div class="m-lbl">Estimators</div><div class="m-val">300</div><div class="m-sub">Decision trees</div></div>'
  +   '<div class="m-stat"><div class="m-lbl">Test Accuracy</div><div class="m-val">' + acc + '</div><div class="m-sub">Held-out test set</div></div>'
  +   '<div class="m-stat"><div class="m-lbl">Input Features</div><div class="m-val">7</div><div class="m-sub">N · P · K · Temp · Humidity · pH · Rainfall</div></div>'
  +   '<div class="m-stat"><div class="m-lbl">Crop Classes</div><div class="m-val">' + cls + '</div><div class="m-sub">Distinct crops</div></div>'
  + '</div>';
}

async function fetchAll() {
  try {
    var [latRes, infoRes] = await Promise.all([
      fetch('/latest'),
      fetch('/model_info')
    ]);
    var data = await latRes.json();
    var info = infoRes.ok ? await infoRes.json() : null;

    var dot  = document.getElementById('dot');
    var stxt = document.getElementById('status-txt');

    if (data.status === 'ok') {
      dot.className  = 'dot live';
      stxt.textContent = 'live';
      var key = data.timestamp;
      if (key !== cachedKey) {
        render(data, info);
        cachedKey = key;
      }
    } else {
      dot.className  = 'dot idle';
      stxt.textContent = 'waiting for rod';
    }
  } catch(e) {
    document.getElementById('dot').className  = 'dot error';
    document.getElementById('status-txt').textContent = 'connection error';
  }
}

async function manualRefresh() {
  var btn = document.getElementById('btn-refresh');
  btn.classList.add('loading');
  btn.disabled = true;
  cachedKey = '';
  await fetchAll();
  setTimeout(function() {
    btn.classList.remove('loading');
    btn.disabled = false;
  }, 700);
}

fetchAll();
setInterval(fetchAll, 5000);
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# CORS -- allow dashboard JS / desktop GUI to reach the API regardless of origin
# ---------------------------------------------------------------------------
@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

# ---------------------------------------------------------------------------
# TRAIN / LOAD MODEL ONCE AT STARTUP
# ---------------------------------------------------------------------------
feature_cols = ["n", "p", "k", "temperature", "humidity", "ph", "rainfall"]

def train_model():
    global model_accuracy, n_classes, model, scaler, label_encoder
    print("Downloading dataset and training model...")
    df = pd.read_csv(DATASET_URL)
    df.columns = [c.strip().lower() for c in df.columns]

    X = df[feature_cols].values
    y_raw = df["label"].values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    n_classes = len(label_encoder.classes_)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    model_accuracy = model.score(X_test, y_test)
    print(f"Model ready. Test accuracy: {model_accuracy:.2%} | Classes: {n_classes}")

    try:
        bundle = {
            "model": model,
            "scaler": scaler,
            "label_encoder": label_encoder,
            "accuracy": model_accuracy,
        }
        with open(MODEL_PKL, "wb") as f:
            pickle.dump(bundle, f)
        print(f"Model saved to {MODEL_PKL}")
    except Exception as e:
        print(f"Could not save model pickle: {e}")


def load_from_pkl():
    global model_accuracy, n_classes, model, scaler, label_encoder
    print(f"Loading pre-trained model from {MODEL_PKL} ...")
    with open(MODEL_PKL, "rb") as f:
        bundle = pickle.load(f)
    if not isinstance(bundle, dict):
        raise ValueError("PKL format not recognized -- need a dict bundle.")
    model = bundle["model"]
    scaler = bundle["scaler"]
    label_encoder = bundle["label_encoder"]
    model_accuracy = bundle.get("accuracy", 0.0)
    n_classes = len(label_encoder.classes_)
    print(f"Model loaded from pickle. Test accuracy: {model_accuracy:.2%} | Classes: {n_classes}")


try:
    load_from_pkl()
except Exception as pkl_err:
    print(f"Pickle load failed ({pkl_err}), training from dataset...")
    train_model()


# ---------------------------------------------------------------------------
# WEATHER HELPERS
# ---------------------------------------------------------------------------
def get_weather_and_rainfall(lat, lon):
    try:
        w = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY, "units": "metric"},
            timeout=10,
        ).json()
        temperature = w["main"]["temp"]
        humidity = w["main"]["humidity"]

        f = requests.get(
            "https://api.openweathermap.org/data/2.5/forecast",
            params={"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY, "units": "metric"},
            timeout=10,
        ).json()
        rainfall = sum(e.get("rain", {}).get("3h", 0.0) for e in f.get("list", [])[:40])

        return temperature, humidity, rainfall
    except Exception as e:
        print(f"Weather fetch failed ({e}), using fallback values.")
        return 26.0, 75.0, 180.0


def get_rainfall_only(lat, lon):
    """Used when the rod supplies temp/humidity via DHT11."""
    try:
        f = requests.get(
            "https://api.openweathermap.org/data/2.5/forecast",
            params={"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY, "units": "metric"},
            timeout=10,
        ).json()
        return sum(e.get("rain", {}).get("3h", 0.0) for e in f.get("list", [])[:40])
    except Exception as e:
        print(f"Rainfall forecast fetch failed ({e}), using fallback value.")
        return 180.0


# ---------------------------------------------------------------------------
# MAIN ENDPOINT -- what the ESP32 (and the GUI) call
# ---------------------------------------------------------------------------
@app.route("/recommend", methods=["POST"])
def recommend():
    global last_esp_seen_ts

    data = request.get_json(force=True)

    manual_override = bool(data.get("manual_override", False))

    n   = data.get("N")
    p   = data.get("P")
    k   = data.get("K")
    ph  = data.get("pH")

    soil_moisture_pct = data.get("soil_moisture_pct")
    soil_temp_c       = data.get("soil_temp_c")

    lat = data.get("lat", 12.9716)
    lon = data.get("lon", 77.5946)

    if None in (n, p, k, ph):
        return jsonify({"error": "Missing required fields: N, P, K, pH"}), 400

    ambient_temp_c       = data.get("ambient_temp_c")
    ambient_humidity_pct = data.get("ambient_humidity_pct")

    if manual_override:
        # Full manual control from the desktop GUI -- use exactly what was typed.
        temperature    = data.get("temperature", 26.0)
        humidity       = data.get("humidity", 70.0)
        rainfall       = data.get("rainfall", 180.0)
        weather_source = "manual"
    elif ambient_temp_c is not None and ambient_humidity_pct is not None:
        # Real ESP32 reading with DHT11 attached.
        temperature    = ambient_temp_c
        humidity       = ambient_humidity_pct
        rainfall       = get_rainfall_only(lat, lon)
        weather_source = "sensor"
        last_esp_seen_ts = time.time()
    else:
        weather_source = "api_fallback"
        temperature, humidity, rainfall = get_weather_and_rainfall(lat, lon)
        if not manual_override:
            last_esp_seen_ts = time.time()

    X_input  = scaler.transform([[n, p, k, temperature, humidity, ph, rainfall]])
    pred_idx = model.predict(X_input)[0]
    probs    = model.predict_proba(X_input)[0]

    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [
        {"crop": label_encoder.inverse_transform([i])[0], "confidence": round(float(probs[i]) * 100, 1)}
        for i in top3_idx
    ]

    response = {
        "status":           "ok",
        "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "recommended_crop": label_encoder.inverse_transform([pred_idx])[0],
        "top_3":            top3,
        "weather_used": {
            "temperature":          round(float(temperature), 1),
            "humidity":             round(float(humidity), 1),
            "rainfall_forecast_mm": round(float(rainfall), 1),
            "source":               weather_source,
        },
        "sensor_input": {
            "N": n, "P": p, "K": k, "pH": ph,
            "soil_moisture_pct": soil_moisture_pct,
            "soil_temp_c":       soil_temp_c,
        },
    }

    # Only real ESP32 traffic (not manual GUI tests) updates the dashboard's
    # "latest reading" and the connection heartbeat.
    if not manual_override:
        latest_reading.update(response)

    return jsonify(response)


@app.route("/latest", methods=["GET"])
def latest():
    return jsonify(latest_reading)


@app.route("/status", methods=["GET"])
def status():
    """Lightweight heartbeat the GUI polls to show ESP32 connected/idle."""
    connected = False
    seconds_since = None
    last_seen_str = None
    if last_esp_seen_ts is not None:
        seconds_since = time.time() - last_esp_seen_ts
        connected = seconds_since < ESP_TIMEOUT_SEC
        last_seen_str = datetime.fromtimestamp(last_esp_seen_ts).strftime("%Y-%m-%d %H:%M:%S")
    return jsonify({
        "esp_connected": connected,
        "seconds_since_last_reading": round(seconds_since, 1) if seconds_since is not None else None,
        "last_seen": last_seen_str,
    })


@app.route("/model_info", methods=["GET"])
def model_info():
    return jsonify({
        "type":          "Random Forest Classifier",
        "n_estimators":  300,
        "accuracy":      round(model_accuracy, 4),
        "n_features":    len(feature_cols),
        "feature_names": feature_cols,
        "n_classes":     n_classes,
        "classes":       list(label_encoder.classes_),
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "running", "message": "POST sensor data to /recommend"})


# ---------------------------------------------------------------------------
# DASHBOARD (unchanged web view -- kept for browser access)
# ---------------------------------------------------------------------------
DASHBOARD_HTML = """
<!DOCTYPE html>
<html><body style="background:#070d0b;color:#ddeee9;font-family:sans-serif;padding:40px;">
<h2>Smart Crop Rod server is running.</h2>
<p>Use the Tkinter desktop app (smart_crop_gui.py) for the full control panel,
or hit <code>/latest</code>, <code>/status</code>, <code>/model_info</code> directly.</p>
</body></html>
"""


DASHBOARD_HTML = DASHBOARD_HTML_ORIGINAL


@app.route("/", methods=["GET"])
def dashboard():
    return render_template_string(DASHBOARD_HTML)
# ---------------------------------------------------------------------------
# THEME
# ---------------------------------------------------------------------------
BG       = "#070d0b"
PANEL    = "#0d1613"
PANEL2   = "#0a1210"
BORDER   = "#1c2b26"
TEAL     = "#00c49a"
TEAL_LO  = "#0d2b24"
AMBER    = "#f0a832"
RED      = "#ff5f5f"
TEXT     = "#ddeee9"
TEXT_SEC = "#7a9590"
TEXT_DIM = "#3d5450"

FONT_SANS = ("Segoe UI", 10)
FONT_SANS_B = ("Segoe UI", 11, "bold")
FONT_MONO = ("Consolas", 10)
FONT_MONO_LG = ("Consolas", 22, "bold")
FONT_TITLE = ("Segoe UI", 15, "bold")

DEFAULT_SERVER = "http://127.0.0.1:5000"

# Matches the FALLBACK_N/P/K and placeholder values in the ESP32 sketch.
DEFAULTS = {
    "N": "50", "P": "35", "K": "50", "pH": "6.5",
    "temperature": "26.0", "humidity": "72.0", "rainfall": "180.0",
    "soil_moisture_pct": "55.0", "soil_temp_c": "24.0",
}

FIELD_ORDER = [
    ("N", "Nitrogen (N)", "mg/kg"),
    ("P", "Phosphorus (P)", "mg/kg"),
    ("K", "Potassium (K)", "mg/kg"),
    ("pH", "pH Level", "0-14"),
    ("temperature", "Temperature", "\u00b0C"),
    ("humidity", "Humidity", "%"),
    ("rainfall", "Rainfall (forecast)", "mm"),
    ("soil_moisture_pct", "Soil Moisture", "%"),
    ("soil_temp_c", "Soil Temp", "\u00b0C"),
]


class SmartCropGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Smart Crop Rod - Control Panel")
        self.geometry("1040x680")
        self.configure(bg=BG)
        self.minsize(900, 600)

        self.entries = {}
        self.esp_connected = False
        self.polling = True

        self._build_header()
        self._build_body()
        self._build_footer()

        self.reset_defaults()
        self._poll_status_loop()

    # ------------------------------------------------------------------
    # UI CONSTRUCTION
    # ------------------------------------------------------------------
    def _panel(self, parent, **kw):
        f = tk.Frame(parent, bg=PANEL, highlightbackground=BORDER,
                     highlightthickness=1, **kw)
        return f

    def _build_header(self):
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill="x", padx=20, pady=(16, 10))

        left = tk.Frame(hdr, bg=BG)
        left.pack(side="left")
        tk.Label(left, text="Smart Crop Rod", font=FONT_TITLE, bg=BG, fg=TEXT).pack(anchor="w")
        tk.Label(left, text="Manual + Live Sensor Control Panel", font=FONT_MONO,
                 bg=BG, fg=TEXT_SEC).pack(anchor="w")

        right = tk.Frame(hdr, bg=BG)
        right.pack(side="right")

        status_row = tk.Frame(right, bg=BG)
        status_row.pack(side="top", anchor="e")
        self.status_dot = tk.Canvas(status_row, width=10, height=10, bg=BG,
                                     highlightthickness=0)
        self.status_dot.pack(side="left", padx=(0, 6))
        self._draw_dot(TEXT_DIM)
        self.status_label = tk.Label(status_row, text="checking server...",
                                      font=FONT_MONO, bg=BG, fg=TEXT_SEC)
        self.status_label.pack(side="left")

        server_row = tk.Frame(right, bg=BG)
        server_row.pack(side="top", anchor="e", pady=(6, 0))
        tk.Label(server_row, text="Server:", font=FONT_SANS, bg=BG, fg=TEXT_SEC).pack(side="left")
        self.server_var = tk.StringVar(value=DEFAULT_SERVER)
        server_entry = tk.Entry(server_row, textvariable=self.server_var, width=26,
                                 font=FONT_MONO, bg=PANEL2, fg=TEXT,
                                 insertbackground=TEXT, relief="flat")
        server_entry.pack(side="left", padx=6)

    def _draw_dot(self, color):
        self.status_dot.delete("all")
        self.status_dot.create_oval(1, 1, 9, 9, fill=color, outline="")

    def _build_body(self):
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=20, pady=10)

        # ---- LEFT: input panel ----
        left = self._panel(body)
        left.pack(side="left", fill="y", padx=(0, 12))
        left.configure(width=280)

        tk.Label(left, text="SENSOR / MANUAL INPUT", font=("Segoe UI", 9, "bold"),
                 bg=PANEL, fg=TEAL).pack(anchor="w", padx=16, pady=(16, 12))

        for key, label, unit in FIELD_ORDER:
            row = tk.Frame(left, bg=PANEL)
            row.pack(fill="x", padx=16, pady=5)
            tk.Label(row, text=label, font=FONT_SANS, bg=PANEL, fg=TEXT,
                     width=16, anchor="w").pack(side="left")
            var = tk.StringVar()
            ent = tk.Entry(row, textvariable=var, width=9, font=FONT_MONO,
                           bg=PANEL2, fg=TEXT, insertbackground=TEXT, relief="flat")
            ent.pack(side="left", padx=(4, 4))
            tk.Label(row, text=unit, font=("Segoe UI", 8), bg=PANEL,
                     fg=TEXT_DIM, width=6, anchor="w").pack(side="left")
            self.entries[key] = var

        # source-of-values hint, updated after a fetch
        self.source_hint = tk.Label(left, text="Fields are editable. Type anything.",
                                     font=("Segoe UI", 8), bg=PANEL, fg=TEXT_DIM,
                                     wraplength=250, justify="left")
        self.source_hint.pack(anchor="w", padx=16, pady=(10, 16))

        # ---- RIGHT: results panel ----
        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        self.hero = self._panel(right)
        self.hero.pack(fill="x", pady=(0, 12))
        self._render_empty_hero()

        grid2 = tk.Frame(right, bg=BG)
        grid2.pack(fill="both", expand=True)

        self.soil_panel = self._panel(grid2)
        self.soil_panel.pack(side="left", fill="both", expand=True, padx=(0, 6))
        tk.Label(self.soil_panel, text="SOIL SNAPSHOT USED", font=("Segoe UI", 9, "bold"),
                 bg=PANEL, fg=TEAL).pack(anchor="w", padx=14, pady=(14, 8))
        self.soil_body = tk.Frame(self.soil_panel, bg=PANEL)
        self.soil_body.pack(fill="both", expand=True, padx=14, pady=(0, 14))

        self.env_panel = self._panel(grid2)
        self.env_panel.pack(side="left", fill="both", expand=True, padx=(6, 0))
        tk.Label(self.env_panel, text="ENVIRONMENT USED", font=("Segoe UI", 9, "bold"),
                 bg=PANEL, fg=TEAL).pack(anchor="w", padx=14, pady=(14, 8))
        self.env_body = tk.Frame(self.env_panel, bg=PANEL)
        self.env_body.pack(fill="both", expand=True, padx=14, pady=(0, 14))

    def _build_footer(self):
        foot = tk.Frame(self, bg=BG)
        foot.pack(fill="x", padx=20, pady=(0, 18))

        self.btn_fetch = tk.Button(foot, text="Fetch Sensor Data", font=FONT_SANS_B,
                                    bg=TEAL_LO, fg=TEAL, activebackground=TEAL,
                                    activeforeground=BG, relief="flat", bd=0,
                                    padx=16, pady=8, cursor="hand2",
                                    command=self.fetch_sensor_data)
        self.btn_fetch.pack(side="left", padx=(0, 10))

        self.btn_analyze = tk.Button(foot, text="Get Recommendation", font=FONT_SANS_B,
                                      bg=TEAL, fg=BG, activebackground="#00e0b0",
                                      activeforeground=BG, relief="flat", bd=0,
                                      padx=16, pady=8, cursor="hand2",
                                      command=self.get_recommendation)
        self.btn_analyze.pack(side="left", padx=(0, 10))

        self.btn_reset = tk.Button(foot, text="Reset Defaults", font=FONT_SANS,
                                    bg=PANEL2, fg=TEXT_SEC, activebackground=BORDER,
                                    activeforeground=TEXT, relief="flat", bd=0,
                                    padx=14, pady=8, cursor="hand2",
                                    command=self.reset_defaults)
        self.btn_reset.pack(side="left")

        self.footer_msg = tk.Label(foot, text="", font=FONT_MONO, bg=BG, fg=TEXT_DIM)
        self.footer_msg.pack(side="right")

    # ------------------------------------------------------------------
    # RESULT RENDERING
    # ------------------------------------------------------------------
    def _clear(self, frame):
        for w in frame.winfo_children():
            w.destroy()

    def _render_empty_hero(self):
        self._clear(self.hero)
        tk.Label(self.hero, text="No recommendation yet", font=FONT_TITLE,
                 bg=PANEL, fg=TEXT_SEC).pack(anchor="w", padx=20, pady=(18, 4))
        tk.Label(self.hero, text="Click \"Fetch Sensor Data\" to pull the ESP32's last "
                                  "reading, or edit the fields and click \"Get Recommendation\".",
                 font=FONT_SANS, bg=PANEL, fg=TEXT_DIM, wraplength=600,
                 justify="left").pack(anchor="w", padx=20, pady=(0, 18))

    def _render_hero(self, crop, confidence, top3, timestamp, source):
        self._clear(self.hero)
        top = tk.Frame(self.hero, bg=PANEL)
        top.pack(fill="x", padx=20, pady=(18, 8))

        left = tk.Frame(top, bg=PANEL)
        left.pack(side="left", fill="both", expand=True)
        tk.Label(left, text="RECOMMENDED CROP", font=("Segoe UI", 9, "bold"),
                 bg=PANEL, fg=TEAL).pack(anchor="w")
        tk.Label(left, text=crop.title(), font=FONT_MONO_LG, bg=PANEL, fg=TEXT).pack(anchor="w", pady=(4, 6))
        pill = tk.Label(left, text=f"{confidence:.1f}% confidence", font=FONT_MONO,
                         bg=TEAL_LO, fg=AMBER, padx=12, pady=4)
        pill.pack(anchor="w")

        right = tk.Frame(top, bg=PANEL)
        right.pack(side="right", fill="both", padx=(30, 0))
        tk.Label(right, text="TOP 3 ALTERNATIVES", font=("Segoe UI", 9, "bold"),
                 bg=PANEL, fg=TEAL).pack(anchor="w", pady=(0, 8))
        for i, c in enumerate(top3):
            row = tk.Frame(right, bg=PANEL)
            row.pack(fill="x", pady=3)
            tk.Label(row, text=f"#{i+1}", font=FONT_MONO, bg=PANEL, fg=TEXT_DIM,
                     width=3).pack(side="left")
            tk.Label(row, text=c["crop"].title(), font=FONT_SANS, bg=PANEL, fg=TEXT,
                     width=14, anchor="w").pack(side="left")
            bar_track = tk.Frame(row, bg=BORDER, width=120, height=6)
            bar_track.pack(side="left", padx=6)
            bar_track.pack_propagate(False)
            fill_w = max(2, int(120 * c["confidence"] / 100))
            tk.Frame(bar_track, bg=TEAL if i == 0 else TEXT_DIM,
                     width=fill_w, height=6).place(x=0, y=0)
            tk.Label(row, text=f"{c['confidence']:.1f}%", font=FONT_MONO, bg=PANEL,
                     fg=TEXT_SEC, width=6).pack(side="left")

        src_colors = {"sensor": TEAL, "manual": "#9aa0ff", "api_fallback": AMBER}
        src_labels = {"sensor": "DHT11 Sensor", "manual": "Manual Entry", "api_fallback": "Weather API Fallback"}
        bottom = tk.Frame(self.hero, bg=PANEL)
        bottom.pack(fill="x", padx=20, pady=(6, 16))
        tk.Frame(bottom, bg=BORDER, height=1).pack(fill="x", pady=(0, 8))
        row = tk.Frame(bottom, bg=PANEL)
        row.pack(fill="x")
        tk.Label(row, text=f"Timestamp: {timestamp}", font=FONT_MONO, bg=PANEL,
                 fg=TEXT_DIM).pack(side="left")
        tk.Label(row, text=src_labels.get(source, source), font=FONT_MONO,
                 bg=PANEL, fg=src_colors.get(source, TEXT_SEC)).pack(side="right")

    def _kv_row(self, parent, label, value, unit=""):
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill="x", pady=3)
        tk.Label(row, text=label, font=FONT_SANS, bg=PANEL, fg=TEXT_SEC,
                 width=14, anchor="w").pack(side="left")
        text = "-" if value is None else f"{value}{unit}"
        tk.Label(row, text=text, font=FONT_MONO, bg=PANEL, fg=TEXT).pack(side="left")

    def _render_soil(self, sensor_input):
        self._clear(self.soil_body)
        self._kv_row(self.soil_body, "Nitrogen", sensor_input.get("N"), " mg/kg")
        self._kv_row(self.soil_body, "Phosphorus", sensor_input.get("P"), " mg/kg")
        self._kv_row(self.soil_body, "Potassium", sensor_input.get("K"), " mg/kg")
        self._kv_row(self.soil_body, "pH Level", sensor_input.get("pH"))
        self._kv_row(self.soil_body, "Moisture", sensor_input.get("soil_moisture_pct"), " %")
        self._kv_row(self.soil_body, "Soil Temp", sensor_input.get("soil_temp_c"), " \u00b0C")

    def _render_env(self, weather_used):
        self._clear(self.env_body)
        self._kv_row(self.env_body, "Temperature", weather_used.get("temperature"), " \u00b0C")
        self._kv_row(self.env_body, "Humidity", weather_used.get("humidity"), " %")
        self._kv_row(self.env_body, "Rainfall", weather_used.get("rainfall_forecast_mm"), " mm")
        self._kv_row(self.env_body, "Source", weather_used.get("source"))

    # ------------------------------------------------------------------
    # ACTIONS
    # ------------------------------------------------------------------
    def reset_defaults(self):
        for key, var in self.entries.items():
            var.set(DEFAULTS.get(key, ""))
        self.footer_msg.config(text="Fields reset to sketch fallback defaults.")

    def _server_url(self):
        return self.server_var.get().strip().rstrip("/")

    def _get_float(self, key, default=0.0):
        try:
            return float(self.entries[key].get())
        except (ValueError, KeyError):
            return default

    def fetch_sensor_data(self):
        self.btn_fetch.config(state="disabled", text="Fetching...")
        threading.Thread(target=self._fetch_sensor_data_thread, daemon=True).start()

    def _fetch_sensor_data_thread(self):
        url = self._server_url() + "/latest"
        try:
            r = requests.get(url, timeout=6)
            data = r.json()
        except Exception as e:
            self.after(0, lambda: self._fetch_failed(str(e)))
            return
        self.after(0, lambda: self._fetch_done(data))

    def _fetch_failed(self, err):
        self.btn_fetch.config(state="normal", text="Fetch Sensor Data")
        self.footer_msg.config(text=f"Fetch failed: {err}", fg=RED)

    def _fetch_done(self, data):
        self.btn_fetch.config(state="normal", text="Fetch Sensor Data")
        if data.get("status") != "ok":
            self.footer_msg.config(text="No ESP32 reading available yet.", fg=AMBER)
            self.source_hint.config(text="No sensor reading on the server yet. "
                                          "Fields left as-is -- edit and Get Recommendation to test manually.")
            return

        s = data["sensor_input"]
        w = data["weather_used"]
        for key in ("N", "P", "K", "pH", "soil_moisture_pct", "soil_temp_c"):
            if s.get(key) is not None:
                self.entries[key].set(str(s[key]))
        self.entries["temperature"].set(str(w.get("temperature", "")))
        self.entries["humidity"].set(str(w.get("humidity", "")))
        self.entries["rainfall"].set(str(w.get("rainfall_forecast_mm", "")))

        top3 = data["top_3"]
        self._render_hero(data["recommended_crop"], top3[0]["confidence"], top3,
                           data["timestamp"], w.get("source", "sensor"))
        self._render_soil(s)
        self._render_env(w)

        self.source_hint.config(text=f"Loaded from ESP32's last reading ({data['timestamp']}). "
                                      f"Edit any field, then Get Recommendation to re-test.")
        self.footer_msg.config(text="Fetched latest ESP32 reading.", fg=TEXT_DIM)

    def get_recommendation(self):
        self.btn_analyze.config(state="disabled", text="Analyzing...")
        payload = {
            "manual_override": True,
            "N": self._get_float("N"),
            "P": self._get_float("P"),
            "K": self._get_float("K"),
            "pH": self._get_float("pH"),
            "temperature": self._get_float("temperature"),
            "humidity": self._get_float("humidity"),
            "rainfall": self._get_float("rainfall"),
            "soil_moisture_pct": self._get_float("soil_moisture_pct"),
            "soil_temp_c": self._get_float("soil_temp_c"),
        }
        threading.Thread(target=self._get_recommendation_thread, args=(payload,), daemon=True).start()

    def _get_recommendation_thread(self, payload):
        url = self._server_url() + "/recommend"
        try:
            r = requests.post(url, json=payload, timeout=10)
            data = r.json()
        except Exception as e:
            self.after(0, lambda: self._analyze_failed(str(e)))
            return
        if r.status_code != 200:
            self.after(0, lambda: self._analyze_failed(data.get("error", f"HTTP {r.status_code}")))
            return
        self.after(0, lambda: self._analyze_done(data))

    def _analyze_failed(self, err):
        self.btn_analyze.config(state="normal", text="Get Recommendation")
        self.footer_msg.config(text=f"Recommendation failed: {err}", fg=RED)

    def _analyze_done(self, data):
        self.btn_analyze.config(state="normal", text="Get Recommendation")
        top3 = data["top_3"]
        self._render_hero(data["recommended_crop"], top3[0]["confidence"], top3,
                           data["timestamp"], data["weather_used"].get("source", "manual"))
        self._render_soil(data["sensor_input"])
        self._render_env(data["weather_used"])
        self.footer_msg.config(text="Recommendation computed from your manual values.", fg=TEXT_DIM)

    # ------------------------------------------------------------------
    # CONNECTION STATUS (lightweight heartbeat only -- never touches
    # the input fields or the results panel)
    # ------------------------------------------------------------------
    def _poll_status_loop(self):
        threading.Thread(target=self._poll_status_once, daemon=True).start()
        self.after(5000, self._poll_status_loop)

    def _poll_status_once(self):
        url = self._server_url() + "/status"
        try:
            r = requests.get(url, timeout=4)
            data = r.json()
            connected = bool(data.get("esp_connected"))
            last_seen = data.get("last_seen")
            self.after(0, lambda: self._set_status(True, connected, last_seen))
        except Exception:
            self.after(0, lambda: self._set_status(False, False, None))

    def _set_status(self, server_reachable, esp_connected, last_seen):
        if not server_reachable:
            self._draw_dot(RED)
            self.status_label.config(text="server unreachable", fg=RED)
        elif esp_connected:
            self._draw_dot(TEAL)
            self.status_label.config(text="ESP32 connected", fg=TEAL)
        else:
            self._draw_dot(TEXT_DIM)
            suffix = f" (last seen {last_seen})" if last_seen else " (no readings yet)"
            self.status_label.config(text="ESP32 idle" + suffix, fg=TEXT_SEC)


# ---------------------------------------------------------------------------
# STARTUP -- run the Flask server in a background thread, GUI in the main thread
# ---------------------------------------------------------------------------
def run_server():
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # give Flask a moment to bind the port before the GUI starts polling /status
    time.sleep(1.0)

    app_gui = SmartCropGUI()
    app_gui.mainloop()
