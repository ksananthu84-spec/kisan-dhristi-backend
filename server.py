from datetime import datetime
import io
import json
import os
from typing import Any, Dict, List, Optional
import urllib.request

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from PIL import Image
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

app = FastAPI(title="Kisan Rover AI Core & Telemetry Server")

# ---------------------------------------------------------
# 1. CORS CONFIGURATION
# ---------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# 2. AGRICULTURAL TREATMENT GUIDE
# ---------------------------------------------------------
TREATMENT_GUIDE = {
    "Healthy": (
        "Foliage is vigorous and healthy. Maintain standard drip irrigation and"
        " balanced NPK nutrition."
    ),
    "Late_Blight": (
        "Phytophthora infestans detected. Apply Metalaxyl or Mancozeb fungicide"
        " immediately; prune infected lower leaves."
    ),
    "Septoria_Leaf_Spot": (
        "Septoria lycopersici detected. Spray Copper Oxychloride or"
        " Chlorothalonil; avoid overhead wetting."
    ),
    "Yellow_Leaf_Curl_Virus": (
        "TYLCV viral infection detected. Spray Neem oil (0.5%) to control"
        " whitefly vectors."
    ),
}

# ---------------------------------------------------------
# 3. AI MODEL LOADING & SETUP
# ---------------------------------------------------------
with open("classes.json", "r") as f:
    classes = json.load(f)

DEVICE = torch.device("cpu")
model = models.efficientnet_b0(weights=None)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, len(classes))

# Load whichever model file is present in the repository
MODEL_FILE = "tomato_model.pth" if os.path.exists("tomato_model.pth") else "ginger_model.pth"
model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

os.makedirs("received_photos", exist_ok=True)
app.mount("/photos", StaticFiles(directory="received_photos"), name="photos")

# ---------------------------------------------------------
# 4. ROVER STATE & 2D PLANT MATRIX MAP (4x5 GRID)
# ---------------------------------------------------------
GRID_ROWS = 4
GRID_COLS = 5

field_matrix: List[List[Optional[Dict[str, Any]]]] = [
    [None for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)
]

rover_state = {
    "direction": "STOP",
    "speed": 180,
    "current_row": 0,
    "current_col": 0,
    "auto_mode": True,
}

alert_counter = 0
latest_farmer_alert = {
    "id": 0,
    "disease": "System Ready",
    "display_title": "Ready for Mission",
    "confidence": "0.0%",
    "treatment": "Rover online. Waiting for scan...",
    "is_healthy": True,
    "timestamp": "--:--",
    "photo_url": None,
}
alert_history = []

NTFY_TOPIC = "kisan_rover_tomato"


class RoverCommand(BaseModel):
    action: str  # "FORWARD", "BACKWARD", "LEFT", "RIGHT", "STOP", "AUTO_TOGGLE"
    row: Optional[int] = None
    col: Optional[int] = None


def send_system_push_alert(disease: str, confidence: str, treatment: str):
    """Sends native Android notification with high priority, sound & vibration"""
    try:
        url = f"https://ntfy.sh/{NTFY_TOPIC}"
        message = f"Confidence: {confidence}\nRemedy: {treatment}"
        req = urllib.request.Request(
            url,
            data=message.encode("utf-8"),
            headers={
                "Title": f"DISEASE ALERT: {disease}",
                "Priority": "high",
                "Tags": "warning,herb",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            pass
        print(f"[PUSH] Push alert delivered to topic: {NTFY_TOPIC}")
    except Exception as e:
        print(f"[PUSH ERROR] Could not deliver push alert: {e}")


def standardize_exposure(cv_img_bgr):
    """Standardizes low exposure to match training benchmark"""
    gray = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2GRAY)
    current_luma = float(np.mean(gray))

    if current_luma < 95.0:
        gain = min(135.0 / max(current_luma, 1.0), 3.2)
        cv_img_bgr = cv2.convertScaleAbs(cv_img_bgr, alpha=gain, beta=8)

    return cv_img_bgr


# ---------------------------------------------------------
# 5. ROVER TELEOPERATION & MAPPING ENDPOINTS
# ---------------------------------------------------------
@app.post("/api/rover/control")
def control_rover(cmd: RoverCommand):
    """Receives manual motor commands or coordinate shifts from mobile app"""
    global rover_state

    if cmd.action == "AUTO_TOGGLE":
        rover_state["auto_mode"] = not rover_state["auto_mode"]
    else:
        rover_state["direction"] = cmd.action

    if cmd.row is not None and 0 <= cmd.row < GRID_ROWS:
        rover_state["current_row"] = cmd.row
    if cmd.col is not None and 0 <= cmd.col < GRID_COLS:
        rover_state["current_col"] = cmd.col

    print(
        f"[ROVER MOTOR]: {rover_state['direction']} | Pos:"
        f" ({rover_state['current_row']}, {rover_state['current_col']}) | Auto:"
        f" {rover_state['auto_mode']}"
    )
    return {"status": "ok", "rover": rover_state}


@app.get("/api/rover/status")
def get_rover_status():
    """Polled by the rover motor controller to execute movement"""
    return rover_state


@app.get("/api/field_matrix")
def get_field_matrix():
    """Returns the full 2D plant health matrix and current rover position"""
    return JSONResponse(
        content={
            "rows": GRID_ROWS,
            "cols": GRID_COLS,
            "matrix": field_matrix,
            "rover_position": {
                "row": rover_state["current_row"],
                "col": rover_state["current_col"],
            },
            "rover_state": rover_state,
        },
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/api/latest_alert")
def get_latest_alert():
    """Returns the most recent single diagnosis (no-cache)"""
    return JSONResponse(
        content=latest_farmer_alert,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/api/history")
def get_alert_history():
    """Returns inspection logs"""
    return JSONResponse(
        content={"total": len(alert_history), "history": alert_history[-20:]},
        headers={"Cache-Control": "no-cache"},
    )


# ---------------------------------------------------------
# 6. INFERENCE PIPELINE (ACCEPTS BOTH 'file' & 'image')
# ---------------------------------------------------------
def crop_center_square(cv_img_bgr):
    """Crops the center square matching the phone app's targeting reticle."""
    h, w, _ = cv_img_bgr.shape
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    return cv_img_bgr[start_y : start_y + min_dim, start_x : start_x + min_dim]


@app.post("/predict")
async def predict_crop_health(
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
):
    global alert_counter, latest_farmer_alert, alert_history, field_matrix

    upload = file or image
    if not upload:
        raise HTTPException(
            status_code=400, detail="No image file provided in request."
        )

    contents = await upload.read()
    nparr = np.frombuffer(contents, np.uint8)
    raw_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if raw_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    # 1. Crop to the exact center square the camera targeting reticle aims at
    square_bgr = crop_center_square(raw_bgr)

    # 2. Tensor Transformation
    pil_image = Image.fromarray(cv2.cvtColor(square_bgr, cv2.COLOR_BGR2RGB))
    tensor = transform(pil_image).unsqueeze(0).to(DEVICE)

    # 3. Forward Pass
    with torch.no_grad():
        logits = model(tensor)
        probabilities = F.softmax(logits, dim=1)[0].numpy()

    top_idx = int(np.argmax(probabilities))
    predicted_label = classes[top_idx]
    confidence_score = round(float(probabilities[top_idx]) * 100, 2)

    # Confidence Gate: If disease confidence is too low (< 50%), treat as Healthy/Inconclusive
    if predicted_label != "Healthy" and confidence_score < 50.0:
        healthy_idx = classes.index("Healthy") if "Healthy" in classes else -1
        if healthy_idx != -1 and probabilities[healthy_idx] > 0.25:
            predicted_label = "Healthy"
            confidence_score = round(float(probabilities[healthy_idx]) * 100, 2)

    treatment = TREATMENT_GUIDE.get(
        predicted_label, "Maintain balanced crop care."
    )
    timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp_display = datetime.now().strftime("%I:%M %p, %d %b")
    filename = f"{timestamp_file}_{predicted_label}.jpg"
    cv2.imwrite(f"received_photos/{filename}", square_bgr)

    is_healthy = predicted_label.upper() == "HEALTHY"
    alert_counter += 1
    clean_title = predicted_label.replace("_", " ").title()

    latest_farmer_alert = {
        "id": alert_counter,
        "disease": predicted_label,
        "display_title": clean_title,
        "confidence": f"{confidence_score}%",
        "treatment": treatment,
        "is_healthy": is_healthy,
        "timestamp": timestamp_display,
        "photo_url": f"/photos/{filename}",
    }
    alert_history.append(latest_farmer_alert)

    cur_r = rover_state["current_row"]
    cur_c = rover_state["current_col"]
    field_matrix[cur_r][cur_c] = {
        "disease": predicted_label,
        "display_title": clean_title,
        "confidence": f"{confidence_score}%",
        "treatment": treatment,
        "is_healthy": is_healthy,
        "timestamp": timestamp_display,
        "photo_url": f"/photos/{filename}",
    }

    if not is_healthy:
        send_system_push_alert(clean_title, f"{confidence_score}%", treatment)

    return {
        "status": "success",
        "class": predicted_label,
        "disease": predicted_label,
        "confidence": f"{confidence_score}%",
        "plant_coordinate": {"row": cur_r, "col": cur_c},
        "recommended_action": treatment,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, access_log=False)