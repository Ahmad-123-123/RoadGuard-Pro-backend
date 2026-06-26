from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
from datetime import datetime
import io, base64, cv2, numpy as np
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model — use ONNX if available (3-5x faster on CPU) ──────────────
import os
print("Loading model...")
if os.path.exists("best.onnx"):
    print("  Using ONNX model (fast CPU mode)")
    model = YOLO("best.onnx", task="detect")
else:
    print("  Using PT model — export to ONNX for faster inference:")
    print("  python -c \"from ultralytics import YOLO; YOLO('best.pt').export(format='onnx', imgsz=320)\"")
    model = YOLO("best.pt")

# Warm up
dummy = np.zeros((320, 320, 3), dtype=np.uint8)
model.predict(source=dummy, conf=0.25, imgsz=320, save=False, verbose=False)
print("Model ready.")

potholes = []


# ── Pydantic model for POST /potholes ─────────────────────────────────────
class PotholeIn(BaseModel):
    lat:       float
    lon:       float
    severity:  str
    timestamp: str = ""


# ── Severity calculator ────────────────────────────────────────────────────
def compute_severity(boxes) -> str:
    count = len(boxes)
    if count == 0:
        return "None"

    areas    = []
    max_conf = 0.0
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        areas.append((x2 - x1) * (y2 - y1))
        conf = float(box.conf[0])
        if conf > max_conf:
            max_conf = conf

    max_area = max(areas)

    # Frame resized to 320px wide (~76800px total)
    # High:   box > 6% of frame OR 3+ potholes OR large+confident
    # Medium: box > 2% of frame OR 2 potholes
    # Low:    small single detection
    if count >= 3 or max_area > 4500 or (max_area > 2500 and max_conf >= 0.70):
        return "High"
    elif count == 2 or max_area > 1500 or (max_area > 800 and max_conf >= 0.60):
        return "Medium"
    return "Low"


# ── Bounding-box drawer ────────────────────────────────────────────────────
def draw_boxes(img_array: np.ndarray, boxes, severity: str) -> np.ndarray:
    annotated = img_array.copy()
    color_map = {
        "High":   (220, 38,  38),
        "Medium": (245, 158, 11),
        "Low":    (22,  163, 74),
    }
    color = color_map.get(severity, (100, 100, 100))

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        cl = 10
        for cx, cy, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(annotated, (cx, cy), (cx + dx*cl, cy), color, 2)
            cv2.line(annotated, (cx, cy), (cx, cy + dy*cl), color, 2)

        label = f"Pothole {conf:.0%} [{severity}]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 3, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

    if len(boxes) > 0:
        h, w = annotated.shape[:2]
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, 32), color, -1)
        cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
        cv2.putText(annotated, f"  POTHOLE x{len(boxes)}  |  {severity}", (4, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

    return annotated


# ── Resize + encode to base64 JPEG ────────────────────────────────────────
def encode_frame(img_array: np.ndarray, width: int = 480, quality: int = 55) -> str:
    """Resize to target width and encode as JPEG at given quality (lower = faster)."""
    h, w = img_array.shape[:2]
    if w > width:
        scale  = width / w
        new_h  = int(h * scale)
        img_array = cv2.resize(img_array, (width, new_h), interpolation=cv2.INTER_LINEAR)
    _, buf = cv2.imencode('.jpg', cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR),
                          [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode('utf-8')


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"message": "RoadGuard Pro API — Live Detection Mode"}


@app.get("/potholes")
async def get_potholes():
    return potholes


@app.post("/potholes")
async def add_pothole(data: PotholeIn):
    pothole = {
        "lat":       data.lat,
        "lon":       data.lon,
        "severity":  data.severity,
        "timestamp": data.timestamp or datetime.now().isoformat(),
    }
    potholes.append(pothole)
    print(f"[SAVED] {pothole['severity']} @ {pothole['lat']:.5f},{pothole['lon']:.5f}")
    return {"status": "saved", "total": len(potholes)}


@app.websocket("/ws/live")
async def live_stream(websocket: WebSocket):
    await websocket.accept()
    frame_id = 0
    print("[WS] Client connected")

    try:
        while True:
            data = await websocket.receive_text()

            # ── Decode incoming JPEG frame ──────────────────────────────
            img_data  = base64.b64decode(data.split(',')[1] if ',' in data else data)
            img_array = np.frombuffer(img_data, dtype=np.uint8)
            # Decode directly via OpenCV (faster than PIL)
            frame_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                continue
            # Resize to 320px wide FIRST (reduces YOLO workload significantly)
            h, w = frame_bgr.shape[:2]
            if w > 320:
                new_w = 320
                new_h = int(h * 320 / w)
                frame_bgr = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            frame_id += 1

            # ── YOLOv8 inference ───────────────────────────────────────
            results  = model.predict(
                source  = frame_rgb,
                conf    = 0.29,
                imgsz   = 320,        # force 320px model input (fastest)
                save    = False,
                verbose = False,      # silence per-frame console spam
            )
            boxes    = results[0].boxes
            detected = len(boxes) > 0
            count    = len(boxes)
            severity = compute_severity(boxes)

            # Log detections so you can tune confidence threshold
            if count > 0:
                confs = [f"{float(b.conf[0]):.2f}" for b in boxes]
                areas = [f"{int((b.xyxy[0][2]-b.xyxy[0][0])*(b.xyxy[0][3]-b.xyxy[0][1]))}" for b in boxes]
                print(f"[DETECT] #{frame_id} — {count} box(es) conf={confs} area={areas} -> {severity}")
            elif frame_id % 30 == 0:
                print(f"[CLEAR]  #{frame_id} — no detections")

            # ── Draw boxes & encode ────────────────────────────────────
            annotated = draw_boxes(frame_rgb, boxes, severity)
            # Encode at low quality / small width for fast transfer
            frame_b64 = encode_frame(annotated, width=480, quality=50)

            # ── Send result ────────────────────────────────────────────
            await websocket.send_json({
                "detected": detected,
                "count":    count,
                "severity": severity,
                "frame_id": frame_id,
                "frame":    f"data:image/jpeg;base64,{frame_b64}",
            })

    except WebSocketDisconnect:
        print(f"[WS] Disconnected after {frame_id} frames")
    except Exception as e:
        print(f"[WS] Error: {e}")