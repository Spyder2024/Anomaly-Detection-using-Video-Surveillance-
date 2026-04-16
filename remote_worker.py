import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import cv2
import base64
import numpy as np
import random
from typing import List, Dict, Any

app = FastAPI(title="Agentic Anomaly Detection: Remote CPU Worker")

class FramePayload(BaseModel):
    frame_id: str
    base64_frame: str # JPEG-Base64 encoded
    threshold: float = 0.6

def decode_frame(base64_string: str) -> np.ndarray:
    """Decodes a Base64 JPEG string into an OpenCV frame."""
    try:
        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        raise ValueError(f"Frame decoding failed: {e}")

@app.post("/process_frame")
async def process_frame(payload: FramePayload):
    """
    Offloads CPU-intensive vision analysis (OpenCV) from the orchestrator.
    This runs on the remote machine's CPU.
    """
    print(f"[*] Processing Frame: {payload.frame_id}")
    
    # 1. Decode the frame
    try:
        frame = decode_frame(payload.base64_frame)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # 2. RUN CPU-INTENSIVE OPENCV LOGIC
    # Simulation: Gaussian Blur and Canny Edge Detection (Common CPU tasks)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # 3. Simulate Anomaly Detection based on vision features
    # (In a real scenario, this would be your ML model or specific OpenCV logic)
    edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
    mock_score = round(random.uniform(0.1, 0.95), 2)
    mock_bbox = [random.randint(0, 100), random.randint(0, 100), 50, 50]
    
    print(f" -> Vision Node (CPU) Score: {mock_score}. Density: {edge_density:.4f}")
    
    return {
        "status": "success",
        "raw_anomaly_score": mock_score,
        "bounding_box_data": mock_bbox,
        "vision_metadata": {
            "edge_density": float(edge_density),
            "resolution": list(frame.shape[:2])
        }
    }

if __name__ == "__main__":
    # Start the worker on all interfaces (0.0.0.0) to be reachable via Tailscale
    print("[*] Starting Remote CPU Worker on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)