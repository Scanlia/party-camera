# Party Camera 🎉

Real-time video effects, AI roasts, and psychedelic pattern generator — built for projecting at parties. Control everything from your phone.

## Architecture

```
Windows PC (webcam) ──HTTP POST──▶ Docker Server ──MJPEG──▶ Android Projector
                                        ▲
                                Phone Control Panel
```

## Quick Start

### 1. Server (Docker)

```bash
cd docker/compose/party-camera
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
docker compose up -d --build
```

The server runs on port **8090**.

### 2. Android Projector

Open Chrome on the projector and navigate to:

```
http://<server-ip>:8090/projector
```

Set Chrome to fullscreen (tap the three dots → "Add to Home screen" for a clean kiosk-like experience, or just press F11 / use immersive mode).

### 3. Phone Controller

Open on your phone:

```
http://<server-ip>:8090/control
```

### 4. Camera Sender (Windows)

```bash
cd sender
pip install -r requirements.txt
python send_camera.py http://<server-ip>:8090
```

Pass a second argument for camera index (default `0`):
```bash
python send_camera.py http://192.168.1.50:8090 1
```

## Modes

### 📹 Camera Effects (requires sender)
| Effect | Description |
|--------|-------------|
| No Effect | Clean passthrough |
| AI Roast 🔥 | Face detection + Gemini generates a roast |
| Neon Edges | Canny edges with flowing rainbow colours |
| Kaleidoscope | Rotating 4-fold mirror symmetry |
| Color Cycle | Continuous hue rotation |
| Ghost Trails | Frame feedback with zoom + hue drift |
| Wave Warp | Sinusoidal displacement |
| Thermal | False-colour thermal vision |
| RGB Glitch | Channel offset + block displacement |
| VHS | Retro tape effect with tracking noise |
| Deep Fry | Extreme saturation + sharpening + JPEG artifacts |
| Swirl | Vortex distortion from centre |
| Mirror | 4-way mirror |
| Dreamscape | Feedback + edge glow + colour shift combo |
| Comic | Adaptive threshold outlines + posterisation |
| Pixelate | Chunky nearest-neighbour downscale |

### 🌀 Generative Patterns (no camera needed)
| Pattern | Description |
|---------|-------------|
| Plasma | Classic demo-scene plasma |
| Hypnotic | Pulsing concentric interference |
| Fire | Bottom-up heat propagation |
| Lava Lamp | Floating metaball blobs |
| Tunnel | Infinite depth tunnel |
| Fractal | Rotating Julia-set-like pattern |
| Rainbow | Flowing rainbow sine waves |
| Aurora | Northern lights curtains |
| Wormhole | Spiralling depth vortex |

## Controls

- **Intensity** — how strong the effect is (0–100%)
- **Speed** — animation speed multiplier (0.1–3.0x)
- **Auto Cycle** — automatically switch effects every 15 seconds
- **Blackout** — instant black screen (for transitions)

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | — | Google Gemini API key for AI Roast mode |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model name |
| `STREAM_WIDTH` | `1280` | Output stream width |
| `STREAM_HEIGHT` | `720` | Output stream height |
| `TARGET_FPS` | `25` | Processing frame rate |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Landing page with links |
| `GET` | `/control` | Mobile control panel |
| `GET` | `/projector` | Fullscreen projector view |
| `GET` | `/stream/mjpeg` | Raw MJPEG stream |
| `GET` | `/api/snapshot` | Single JPEG frame |
| `GET` | `/api/state` | Current state JSON |
| `POST` | `/api/mode` | Update state (JSON body) |
| `POST` | `/api/frame` | Receive camera frame |
| `GET` | `/api/effects` | List all effects |

## Tech Stack

- **FastAPI** + **Uvicorn** — async web server
- **OpenCV** — video processing, face detection
- **NumPy** — real-time effect computation
- **Google Gemini** — AI roast generation
- **Pillow** — image conversion for Gemini API
- MJPEG streaming — universal browser compatibility
