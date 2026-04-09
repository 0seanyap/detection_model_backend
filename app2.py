import os
import io
import requests
import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO
import gdown

app = Flask(__name__)

# ---------- CONFIG ----------
MODEL_PATH = "lynne_best.pt"
# FILE_ID = os.environ.get("MODEL_FILE_ID")  # Set this in Render env vars
MODEL_URL = os.environ.get("MODEL_URL")

# ---------- DOWNLOAD MODEL ----------
import os
import urllib.request

MODEL_PATH = "lynne_best.pt"
MODEL_URL = os.environ.get("MODEL_URL")

def download_model():
    if os.path.exists(MODEL_PATH):
        print("Model already exists.")
        return

    print("Downloading model from Dropbox...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete!")

download_model()

# ---------- LOAD MODEL ----------
model = YOLO(MODEL_PATH)

# ---------- GLOBAL VARIABLES ----------
last_uploaded_image = None
last_detections = []

# ---------- HELPER FUNCTIONS ----------
def compute_severity(pothole_crop, area_norm):
    h_box, w_box = pothole_crop.shape[:2]
    center_h, center_w = h_box // 2, w_box // 2
    patch_h = max(1, h_box // 10)
    patch_w = max(1, w_box // 10)
    shift_y = h_box // 5
    shifted_center_h = min(center_h + shift_y, h_box - 1)

    center_patch = pothole_crop[
        shifted_center_h - patch_h//2 : shifted_center_h + patch_h//2 + 1,
        center_w - patch_w//2 : center_w + patch_w//2 + 1
    ]
    center_gray = cv2.cvtColor(center_patch, cv2.COLOR_BGR2GRAY)
    center_avg = center_gray.mean() / 255.0
    darkness_bonus = 1.0 - center_avg
    water_bonus = center_avg if center_avg > 0.75 else 0
    severity_score = 0.60 * area_norm + 0.3 * darkness_bonus + 0.10 * water_bonus

    if severity_score < 0.2:
        severity = "Low"
    elif severity_score < 0.3:
        severity = "Medium"
    else:
        severity = "High"

    return severity, round(severity_score, 3), round(center_avg, 3)

def draw_boxes(frame, detections):
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        color = {"Low": (0,255,0), "Medium": (0,255,255), "High": (0,0,255)}.get(d["severity"], (255,255,255))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"Pothole {d['id']} | {d['severity']} ({d['severity_score']})",
                    (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw center patch
        h_box = y2 - y1
        w_box = x2 - x1
        center_h = (y1 + y2) // 2
        center_w = (x1 + x2) // 2
        patch_h = max(1, h_box // 10)
        patch_w = max(1, w_box // 10)
        shift_y = h_box // 5
        shifted_center_h = min(center_h + shift_y, y2 - 1)

        patch_y1 = shifted_center_h - patch_h // 2
        patch_y2 = shifted_center_h + patch_h // 2
        patch_x1 = center_w - patch_w // 2
        patch_x2 = center_w + patch_w // 2

        cv2.rectangle(frame, (patch_x1, patch_y1), (patch_x2, patch_y2), (255,0,0), 2)
        cv2.putText(frame, f"Brightness: {d['center_avg']}",
                    (patch_x1, patch_y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    return frame

# ---------- ROUTES ----------
@app.route("/")
def home():
    return """
    <h2>Pothole Detection - Live Feed</h2>
    <p id="status">Waiting for upload...</p>
    <img id="preview" src="" style="max-width:800px; display:none; margin-top:12px;">
    <div id="table-container"></div>

    <script>
    setInterval(async () => {
        const res = await fetch("/image", { method: "HEAD" });
        if (res.ok) {
            document.getElementById("preview").src = "/image?t=" + Date.now();
            document.getElementById("preview").style.display = "block";
            document.getElementById("status").innerText = "Last uploaded image:";
        }

        const tableRes = await fetch("/results");
        const html = await tableRes.text();
        document.getElementById("table-container").innerHTML = html;
    }, 1000);
    </script>
    """

@app.route("/results")
def results():
    if not last_detections:
        return "<p>No detections yet.</p>"
    rows = ""
    for d in last_detections:
        color = {"Low":"green","Medium":"orange","High":"red"}.get(d["severity"],"black")
        rows += f"""
        <tr>
            <td>{d['id']}</td>
            <td>{d['confidence']:.2f}</td>
            <td style="color:{color}; font-weight:bold">{d['severity']}</td>
            <td>{d['severity_score']}</td>
            <td>{d['center_avg']}</td>
            <td>{d['bbox']}</td>
        </tr>
        """
    return f"""
    <table border="1" cellpadding="8" cellspacing="0" style="border-collapse:collapse; margin-top:16px;">
        <tr style="background:#eee">
            <th>#</th><th>Confidence</th><th>Severity</th>
            <th>Score</th><th>Center Brightness</th><th>BBox</th>
        </tr>
        {rows}
    </table>
    """

@app.route("/detect", methods=["POST"])
def detect():
    global last_uploaded_image, last_detections

    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    img_bytes = file.read()
    img = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
    frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_area = frame.shape[0]*frame.shape[1]

    results = model(frame)
    last_detections = []

    for i, det in enumerate(results[0].boxes):
        x1, y1, x2, y2 = map(int, det.xyxy[0])
        conf = float(det.conf[0])
        if conf < 0.5:
            continue

        pothole_crop = frame[y1:y2, x1:x2]
        area_norm = ((x2 - x1)*(y2 - y1)) / img_area
        severity, score, center_avg = compute_severity(pothole_crop, area_norm)

        last_detections.append({
            "id": i,
            "class": model.names[int(det.cls[0])],
            "confidence": conf,
            "bbox": [x1, y1, x2, y2],
            "severity": severity,
            "severity_score": score,
            "center_avg": center_avg
        })

    annotated_frame = draw_boxes(frame, last_detections)
    _, buffer = cv2.imencode(".jpg", annotated_frame)
    last_uploaded_image = buffer.tobytes()

    return jsonify(last_detections)

@app.route("/image")
def get_image():
    if last_uploaded_image is None:
        return "No image uploaded yet", 404
    return send_file(io.BytesIO(last_uploaded_image), mimetype="image/jpeg")

# ---------- MAIN ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)