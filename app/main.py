import asyncio
import os
import time
import threading
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from effects import EffectEngine
from patterns import PatternEngine
from roast import RoastEngine


# ---------------------------------------------------------------------------
# Frame buffer (thread-safe)
# ---------------------------------------------------------------------------
class FrameBuffer:
    def __init__(self):
        self._input = None
        self._output_jpeg = None
        self._lock = threading.Lock()
        self._new_frame = threading.Event()
        self.last_input_time = 0

    def set_input(self, frame: np.ndarray):
        with self._lock:
            self._input = frame
            self.last_input_time = time.time()

    def get_input(self) -> np.ndarray | None:
        with self._lock:
            return self._input.copy() if self._input is not None else None

    def has_recent_input(self, max_age: float = 3.0) -> bool:
        return (time.time() - self.last_input_time) < max_age

    def set_output(self, frame: np.ndarray):
        with self._lock:
            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            self._output_jpeg = jpeg.tobytes()
            self._new_frame.set()

    def get_output_jpeg(self) -> bytes | None:
        with self._lock:
            return self._output_jpeg

    def wait_for_frame(self, timeout: float = 1.0) -> bool:
        result = self._new_frame.wait(timeout)
        self._new_frame.clear()
        return result


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------
CAMERA_EFFECTS = [
    "none", "roast", "neon_edges", "kaleidoscope", "color_cycle",
    "feedback", "wave_warp", "thermal", "rgb_glitch", "vhs",
    "deep_fry", "swirl", "mirror_quad", "dreamscape", "comic", "pixelate",
]

PATTERN_TYPES = [
    "plasma", "hypnotic", "fire", "lava_lamp", "tunnel",
    "fractal_spin", "rainbow_wave", "aurora", "wormhole",
]


class AppState:
    def __init__(self):
        self.mode = "pattern"          # "camera" | "pattern"
        self.camera_effect = "none"
        self.pattern_type = "plasma"
        self.intensity = 0.7
        self.speed = 1.0
        self.auto_cycle = False
        self.auto_cycle_interval = 15  # seconds
        self.blackout = False
        self.width = int(os.environ.get("STREAM_WIDTH", 1280))
        self.height = int(os.environ.get("STREAM_HEIGHT", 720))

    def to_dict(self):
        return {
            "mode": self.mode,
            "camera_effect": self.camera_effect,
            "pattern_type": self.pattern_type,
            "intensity": self.intensity,
            "speed": self.speed,
            "auto_cycle": self.auto_cycle,
            "auto_cycle_interval": self.auto_cycle_interval,
            "blackout": self.blackout,
        }


# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
buf = FrameBuffer()
state = AppState()
effect_engine: EffectEngine | None = None
pattern_engine: PatternEngine | None = None
roast_engine: RoastEngine | None = None


def _text_frame(text: str, sub: str = "") -> np.ndarray:
    frame = np.zeros((state.height, state.width, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    sz = cv2.getTextSize(text, font, 1.5, 2)[0]
    x = (state.width - sz[0]) // 2
    y = (state.height + sz[1]) // 2
    cv2.putText(frame, text, (x, y), font, 1.5, (0, 255, 255), 2, cv2.LINE_AA)
    if sub:
        ssz = cv2.getTextSize(sub, font, 0.7, 1)[0]
        cv2.putText(frame, sub, ((state.width - ssz[0]) // 2, y + 50),
                    font, 0.7, (100, 100, 100), 1, cv2.LINE_AA)
    return frame


# ---------------------------------------------------------------------------
# Processing loop (runs in background thread)
# ---------------------------------------------------------------------------
def _processing_loop():
    target_fps = int(os.environ.get("TARGET_FPS", 25))
    frame_time = 1.0 / target_fps
    t = 0.0
    last_cycle = time.time()
    cycle_idx = 0

    while True:
        loop_start = time.time()
        try:
            if state.blackout:
                buf.set_output(np.zeros((state.height, state.width, 3), dtype=np.uint8))
            elif state.mode == "camera":
                frame = buf.get_input()
                if frame is None or not buf.has_recent_input():
                    frame = _text_frame("WAITING FOR CAMERA", "Run sender script on your PC")
                else:
                    frame = cv2.resize(frame, (state.width, state.height))
                    if state.camera_effect == "roast":
                        frame = roast_engine.process(frame, t)
                    elif state.camera_effect != "none":
                        frame = effect_engine.apply(
                            frame, state.camera_effect, t,
                            state.intensity, state.speed,
                        )
                buf.set_output(frame)
            elif state.mode == "pattern":
                frame = pattern_engine.generate(
                    state.pattern_type, t, state.speed, state.intensity,
                )
                buf.set_output(frame)
            else:
                buf.set_output(_text_frame("PARTY CAM"))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            buf.set_output(_text_frame("ERROR", str(exc)[:60]))

        t += frame_time * state.speed

        # Auto-cycle
        if state.auto_cycle:
            now = time.time()
            if now - last_cycle > state.auto_cycle_interval:
                if state.mode == "camera":
                    cycle_idx = (cycle_idx + 1) % len(CAMERA_EFFECTS)
                    state.camera_effect = CAMERA_EFFECTS[cycle_idx]
                else:
                    cycle_idx = (cycle_idx + 1) % len(PATTERN_TYPES)
                    state.pattern_type = PATTERN_TYPES[cycle_idx]
                last_cycle = now

        elapsed = time.time() - loop_start
        if elapsed < frame_time:
            time.sleep(frame_time - elapsed)


# ---------------------------------------------------------------------------
# FastAPI lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_app: FastAPI):
    global effect_engine, pattern_engine, roast_engine

    effect_engine = EffectEngine(state.width, state.height)
    pattern_engine = PatternEngine(state.width, state.height)

    api_key = os.environ.get("GEMINI_API_KEY", "")
    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    roast_engine = RoastEngine(api_key if api_key else None, model_name)
    print(f"[startup] Gemini {'configured' if api_key else 'not set — roast uses fallback text'}")

    thread = threading.Thread(target=_processing_loop, daemon=True)
    thread.start()
    print("[startup] Processing loop running")

    yield
    print("[shutdown] done")


app = FastAPI(title="Party Camera", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/api/frame")
async def receive_frame(frame: UploadFile = File(...)):
    data = await frame.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is not None:
        buf.set_input(img)
        return {"status": "ok"}
    return JSONResponse({"status": "error", "msg": "bad frame"}, 400)


@app.get("/api/state")
async def get_state():
    return state.to_dict()


@app.post("/api/mode")
async def set_mode(request: Request):
    data = await request.json()
    for key in ("mode", "camera_effect", "pattern_type"):
        if key in data:
            setattr(state, key, data[key])
    if "intensity" in data:
        state.intensity = max(0.0, min(1.0, float(data["intensity"])))
    if "speed" in data:
        state.speed = max(0.1, min(3.0, float(data["speed"])))
    if "auto_cycle" in data:
        state.auto_cycle = bool(data["auto_cycle"])
    if "blackout" in data:
        state.blackout = bool(data["blackout"])
    return state.to_dict()


@app.get("/api/effects")
async def list_effects():
    return {"camera_effects": CAMERA_EFFECTS, "pattern_types": PATTERN_TYPES}


@app.get("/api/snapshot")
async def snapshot():
    jpeg = buf.get_output_jpeg()
    if jpeg:
        return Response(content=jpeg, media_type="image/jpeg")
    return Response(status_code=204)


@app.get("/stream/mjpeg")
async def mjpeg_stream():
    async def generate():
        while True:
            if buf.wait_for_frame(timeout=1.0):
                jpeg = buf.get_output_jpeg()
                if jpeg:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                    )
            await asyncio.sleep(0.01)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/projector", response_class=HTMLResponse)
async def projector_page():
    return open("static/projector.html").read()


@app.get("/control", response_class=HTMLResponse)
async def control_page():
    return open("static/control.html").read()


@app.get("/", response_class=HTMLResponse)
async def root():
    return """<!DOCTYPE html>
<html><body style="background:#111;color:#fff;font-family:sans-serif;text-align:center;padding:60px">
<h1 style="font-size:3em">&#127881; Party Camera</h1>
<p style="margin:30px"><a href="/control" style="color:#0ff;font-size:1.5em">&#128241; Control Panel</a></p>
<p style="margin:30px"><a href="/projector" style="color:#f0f;font-size:1.5em">&#128253; Projector View</a></p>
</body></html>"""
