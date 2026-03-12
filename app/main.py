import asyncio
import os
import subprocess
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
from clips import ClipPlayer


# ---------------------------------------------------------------------------
# Frame buffer (thread-safe, zero-copy where possible)
# ---------------------------------------------------------------------------
class FrameBuffer:
    def __init__(self):
        self._input = None
        self._output_jpeg: bytes | None = None
        self._output_raw: np.ndarray | None = None
        self._lock = threading.Lock()
        self._new_frame = threading.Event()
        self.last_input_time: float = 0
        self._jpeg_quality = [cv2.IMWRITE_JPEG_QUALITY, 80]

    def set_input(self, frame: np.ndarray):
        with self._lock:
            self._input = frame
            self.last_input_time = time.time()

    def get_input(self) -> np.ndarray | None:
        with self._lock:
            return self._input  # no copy — processing loop is sole consumer

    def has_recent_input(self, max_age: float = 3.0) -> bool:
        return (time.time() - self.last_input_time) < max_age

    def set_output(self, frame: np.ndarray):
        """Store raw frame; JPEG is encoded lazily on demand."""
        with self._lock:
            self._output_raw = frame
            self._output_jpeg = None  # invalidate cache
            self._new_frame.set()

    def get_output_jpeg(self) -> bytes | None:
        with self._lock:
            if self._output_raw is None:
                return None
            if self._output_jpeg is None:
                _, enc = cv2.imencode(".jpg", self._output_raw, self._jpeg_quality)
                self._output_jpeg = enc.tobytes()
            return self._output_jpeg

    def get_output_raw(self) -> np.ndarray | None:
        with self._lock:
            return self._output_raw  # no copy — RTSP pipe just reads bytes

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
    "gun_barrel", "golden_eye", "casino_hud", "silhouette", "martini_vision",
    "tuxedo", "blood_drip",
]

PATTERN_TYPES = [
    "plasma", "hypnotic", "fire", "lava_lamp", "tunnel",
    "fractal_spin", "rainbow_wave", "aurora", "wormhole",
    "gun_barrel_pattern", "golden_rings", "casino_felt",
]


class AppState:
    def __init__(self):
        self.mode = "pattern"
        self.camera_effect = "none"
        self.pattern_type = "plasma"
        self.intensity = 0.7
        self.speed = 1.0
        self.auto_cycle = True
        self.auto_cycle_interval = 30
        self.shadow_boost = True
        self.blackout = False
        self.width = int(os.environ.get("STREAM_WIDTH", 1280))
        self.height = int(os.environ.get("STREAM_HEIGHT", 720))
        self.camera_source = os.environ.get("CAMERA_SOURCE", "http")  # "http" | "rtsp"
        self.camera_rtsp_url = os.environ.get("CAMERA_RTSP_URL", "")

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
            "shadow_boost": self.shadow_boost,
            "camera_source": self.camera_source,
            "camera_rtsp_url": self.camera_rtsp_url,
        }


# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
buf = FrameBuffer()
state = AppState()
effect_engine: EffectEngine | None = None
pattern_engine: PatternEngine | None = None
roast_engine: RoastEngine | None = None
clip_player: ClipPlayer | None = None
rtsp_input_thread: threading.Thread | None = None


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
# RTSP camera input (DJI Osmo Action 4 / any RTSP source)
# ---------------------------------------------------------------------------
def _rtsp_input_loop(url: str):
    """Pull frames from an RTSP source (e.g. DJI Osmo) and push to FrameBuffer."""
    print(f"[rtsp-in] Connecting to {url}")
    while True:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            print(f"[rtsp-in] Cannot connect to {url}, retrying in 3s...")
            time.sleep(3)
            continue
        print(f"[rtsp-in] Connected to {url}")
        consecutive_fails = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                consecutive_fails += 1
                if consecutive_fails > 30:
                    print("[rtsp-in] Stream lost, reconnecting...")
                    break
                time.sleep(0.01)
                continue
            consecutive_fails = 0
            buf.set_input(frame)
        cap.release()
        time.sleep(2)


# ---------------------------------------------------------------------------
# Stream output via FFmpeg → RTMP to MediaMTX (re-served as RTSP/HLS)
# ---------------------------------------------------------------------------
class RTSPStreamer:
    def __init__(self, url: str, width: int, height: int, fps: int):
        # Convert rtsp:// URL to rtmp:// for publishing
        self._url = url.replace("rtsp://", "rtmp://").replace(":8554", ":1935")
        self._w = width
        self._h = height
        self._fps = fps
        self._proc: subprocess.Popen | None = None
        self._frame_size = width * height * 3

    def start(self):
        cmd = [
            "ffmpeg", "-y",
            "-loglevel", "warning",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self._w}x{self._h}",
            "-r", str(self._fps),
            "-i", "-",
            "-pix_fmt", "yuv420p",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-profile:v", "baseline",
            "-b:v", "3000k",
            "-maxrate", "3500k",
            "-bufsize", "500k",
            "-g", str(self._fps),
            "-bf", "0",
            "-f", "flv",
            self._url,
        ]
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        print(f"[stream-out] FFmpeg publishing to {self._url}", flush=True)

    def send_frame(self, frame: np.ndarray):
        if self._proc and self._proc.stdin:
            try:
                self._proc.stdin.write(frame.tobytes())
            except BrokenPipeError:
                stderr = ""
                if self._proc.stderr:
                    try:
                        stderr = self._proc.stderr.read1(2048).decode(errors="ignore")
                    except Exception:
                        pass
                print(f"[stream-out] Pipe broken, restarting... {stderr}", flush=True)
                self.stop()
                time.sleep(2)
                self.start()

    def stop(self):
        if self._proc:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            self._proc.terminate()
            self._proc = None


# ---------------------------------------------------------------------------
# Processing loop
# ---------------------------------------------------------------------------
_black_frame: np.ndarray | None = None


def _boost_shadows(frame: np.ndarray) -> np.ndarray:
    """Lift shadows and increase visibility in dark rooms."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    # Extra gamma lift for shadows (gamma=0.7 brightens darks)
    lf = l.astype(np.float32) / 255.0
    l = (np.power(lf, 0.7) * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


# Camera auto-cycle order — roast appears repeatedly
_CAMERA_CYCLE = []
for _fx in CAMERA_EFFECTS:
    _CAMERA_CYCLE.append(_fx)
    # Insert roast after every 3rd effect
    if len(_CAMERA_CYCLE) % 4 == 3:
        _CAMERA_CYCLE.append("roast")


def _processing_loop(rtsp: RTSPStreamer | None):
    global _black_frame
    target_fps = int(os.environ.get("TARGET_FPS", 60))
    frame_time = 1.0 / target_fps
    t = 0.0
    last_cycle = time.time()
    cam_cycle_idx = 0
    pat_cycle_idx = 0
    _black_frame = np.zeros((state.height, state.width, 3), dtype=np.uint8)

    # FPS logging
    fps_counter = 0
    fps_timer = time.time()
    rtsp_write_ms = 0.0

    # Pre-allocate resize buffer
    target_size = (state.width, state.height)

    while True:
        loop_start = time.time()
        try:
            if state.blackout:
                buf.set_output(_black_frame)
            elif state.mode == "clips":
                frame = clip_player.get_frame(state.width, state.height)
                if frame is None:
                    frame = _text_frame("NO CLIPS FOUND", "Add .mp4 files to clips/ folder")
                buf.set_output(frame)
            elif state.mode == "camera":
                frame = buf.get_input()
                if frame is None or not buf.has_recent_input():
                    frame = pattern_engine.generate(
                        state.pattern_type, t, state.speed, state.intensity,
                    )
                else:
                    if frame.shape[1] != state.width or frame.shape[0] != state.height:
                        frame = cv2.resize(frame, target_size)
                    if state.camera_effect == "roast":
                        frame = roast_engine.process(frame, t)
                    elif state.camera_effect != "none":
                        frame = effect_engine.apply(
                            frame, state.camera_effect, t,
                            state.intensity, state.speed,
                        )
                # Shadow boost for dark rooms
                if state.shadow_boost and frame is not None:
                    frame = _boost_shadows(frame)
                buf.set_output(frame)
            elif state.mode == "pattern":
                frame = pattern_engine.generate(
                    state.pattern_type, t, state.speed, state.intensity,
                )
                buf.set_output(frame)
            else:
                buf.set_output(_text_frame("PARTY CAM"))
        except Exception:
            import traceback
            traceback.print_exc()
            buf.set_output(_text_frame("ERROR"))

        # Push to RTSP output
        if rtsp:
            raw = buf.get_output_raw()
            if raw is not None:
                t0 = time.time()
                rtsp.send_frame(raw)
                rtsp_write_ms = (time.time() - t0) * 1000

        t += frame_time * state.speed

        # FPS logging
        fps_counter += 1
        now_fps = time.time()
        if now_fps - fps_timer >= 5.0:
            actual_fps = fps_counter / (now_fps - fps_timer)
            print(f"[perf] FPS: {actual_fps:.1f}/{target_fps} | "
                  f"loop: {elapsed*1000:.1f}ms | "
                  f"rtsp_write: {rtsp_write_ms:.1f}ms | "
                  f"mode: {state.mode}/{state.camera_effect if state.mode=='camera' else state.pattern_type}",
                  flush=True)
            fps_counter = 0
            fps_timer = now_fps

        # Auto-cycle (roast revisits built in)
        if state.auto_cycle:
            now = time.time()
            if now - last_cycle > state.auto_cycle_interval:
                if state.mode == "camera" and buf.has_recent_input():
                    cam_cycle_idx = (cam_cycle_idx + 1) % len(_CAMERA_CYCLE)
                    state.camera_effect = _CAMERA_CYCLE[cam_cycle_idx]
                else:
                    pat_cycle_idx = (pat_cycle_idx + 1) % len(PATTERN_TYPES)
                    state.pattern_type = PATTERN_TYPES[pat_cycle_idx]
                last_cycle = now

        elapsed = time.time() - loop_start
        if elapsed < frame_time:
            time.sleep(frame_time - elapsed)


# ---------------------------------------------------------------------------
# FastAPI lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_app: FastAPI):
    global effect_engine, pattern_engine, roast_engine, clip_player, rtsp_input_thread

    effect_engine = EffectEngine(state.width, state.height)
    pattern_engine = PatternEngine(state.width, state.height)

    api_key = os.environ.get("GEMINI_API_KEY", "")
    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    roast_engine = RoastEngine(api_key if api_key else None, model_name)
    print(f"[startup] Gemini {'configured' if api_key else 'not set — roast uses fallback text'}")

    clip_player = ClipPlayer("/app/clips")
    print(f"[startup] Clip player: {clip_player.clip_count()} clips found")

    # Start RTSP camera input if configured
    if state.camera_rtsp_url:
        rtsp_input_thread = threading.Thread(
            target=_rtsp_input_loop, args=(state.camera_rtsp_url,), daemon=True,
        )
        rtsp_input_thread.start()
        print(f"[startup] RTSP camera input: {state.camera_rtsp_url}")

    # RTSP output streamer
    rtsp_url = os.environ.get("RTSP_URL", "")
    rtsp = None
    if rtsp_url:
        rtsp = RTSPStreamer(
            rtsp_url, state.width, state.height,
            int(os.environ.get("TARGET_FPS", 60)),
        )
        rtsp.start()

    thread = threading.Thread(target=_processing_loop, args=(rtsp,), daemon=True)
    thread.start()
    print("[startup] Processing loop running")

    yield

    if rtsp:
        rtsp.stop()
    print("[shutdown] done")


app = FastAPI(title="Party Camera", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/api/frame")
async def receive_frame(frame: UploadFile = File(...)):
    if state.camera_source == "rtsp" and state.camera_rtsp_url:
        return {"status": "ignored", "msg": "RTSP source active"}
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
    global rtsp_input_thread
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
    if "shadow_boost" in data:
        state.shadow_boost = bool(data["shadow_boost"])
    # Allow changing RTSP camera URL at runtime
    if "camera_rtsp_url" in data:
        new_url = data["camera_rtsp_url"].strip()
        if new_url != state.camera_rtsp_url:
            state.camera_rtsp_url = new_url
            if new_url and (rtsp_input_thread is None or not rtsp_input_thread.is_alive()):
                rtsp_input_thread = threading.Thread(
                    target=_rtsp_input_loop, args=(new_url,), daemon=True,
                )
                rtsp_input_thread.start()
            state.camera_source = "rtsp" if new_url else "http"
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
            await asyncio.sleep(0.005)

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
<p style="margin:30px"><a href="/projector" style="color:#f0f;font-size:1.5em">&#128253; Projector View (MJPEG)</a></p>
<p style="margin:15px;color:#888">RTSP: rtsp://&lt;server-ip&gt;:8554/party</p>
<p style="margin:15px;color:#888">HLS: http://&lt;server-ip&gt;:8889/party</p>
</body></html>"""
