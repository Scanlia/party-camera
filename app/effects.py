"""Camera video effects — all NumPy / OpenCV, no extra dependencies."""

import cv2
import numpy as np


class EffectEngine:
    def __init__(self, width: int = 1280, height: int = 720):
        self.w = width
        self.h = height
        # Pre-compute coordinate grids (float32 for remap)
        ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
        self._map_x = xs
        self._map_y = ys
        self._cx = width / 2.0
        self._cy = height / 2.0
        # Normalised coords  -1..1
        self._nx = (xs - self._cx) / self._cx
        self._ny = (ys - self._cy) / self._cy
        self._radius = np.sqrt(self._nx ** 2 + self._ny ** 2)
        self._angle = np.arctan2(self._ny, self._nx)
        # Persistent state
        self._feedback_buf = None
        self._dream_buf = None

    # ------------------------------------------------------------------
    def apply(self, frame: np.ndarray, name: str, t: float,
              intensity: float = 0.7, speed: float = 1.0) -> np.ndarray:
        fn = getattr(self, f"_fx_{name}", None)
        if fn is None:
            return frame
        return fn(frame, t * speed, intensity)

    # === EFFECTS ======================================================

    def _fx_neon_edges(self, frame, t, intensity):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        # Multi-hue edges
        hue_offset = int(t * 40) % 180
        hue_map = ((self._map_x / self.w * 180 + hue_offset) % 180).astype(np.uint8)
        hsv = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        hsv[..., 0] = hue_map
        hsv[..., 1] = 255
        hsv[..., 2] = edges
        colored = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        glow = cv2.GaussianBlur(colored, (21, 21), 0)
        dim = (frame * max(0.1, 1 - intensity * 0.8)).astype(np.uint8)
        return cv2.add(dim, cv2.addWeighted(colored, 1.0, glow, intensity, 0))

    def _fx_kaleidoscope(self, frame, t, intensity):
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        M = cv2.getRotationMatrix2D((cx, cy), t * 15, 1.0)
        rot = cv2.warpAffine(frame, M, (w, h))
        q = rot[:cy, :cx]
        top = np.hstack([q, cv2.flip(q, 1)])
        full = np.vstack([top, cv2.flip(top, 0)])
        result = cv2.resize(full, (w, h))
        # Hue shift
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        hsv[..., 0] = (hsv[..., 0].astype(np.int16) + int(t * 25)) % 180
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _fx_color_cycle(self, frame, t, intensity):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + t * 50 * intensity) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * (1 + intensity * 0.5), 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _fx_feedback(self, frame, t, intensity):
        ff = frame.astype(np.float32)
        if self._feedback_buf is None:
            self._feedback_buf = ff.copy()
        alpha = 0.15 + (1 - intensity) * 0.25
        self._feedback_buf = alpha * ff + (1 - alpha) * self._feedback_buf
        # Slight zoom for trailing effect
        M = cv2.getRotationMatrix2D((self._cx, self._cy), 0.4, 1.004)
        self._feedback_buf = cv2.warpAffine(
            self._feedback_buf, M, (self.w, self.h),
        )
        result = np.clip(self._feedback_buf, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        hsv[..., 0] = (hsv[..., 0].astype(np.int16) + 1) % 180
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _fx_wave_warp(self, frame, t, intensity):
        amp = 20 * intensity
        dx = (amp * np.sin(self._map_y * 0.02 + t * 3)).astype(np.float32)
        dy = (amp * np.cos(self._map_x * 0.02 + t * 2)).astype(np.float32)
        mx = np.clip(self._map_x + dx, 0, self.w - 1)
        my = np.clip(self._map_y + dy, 0, self.h - 1)
        return cv2.remap(frame, mx, my, cv2.INTER_LINEAR)

    def _fx_thermal(self, frame, t, intensity):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        colored = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        return cv2.addWeighted(colored, intensity, frame, 1 - intensity, 0)

    def _fx_rgb_glitch(self, frame, t, intensity):
        result = frame.copy()
        shift = max(1, int(12 * intensity * (1 + np.sin(t * 3)) * 0.5))
        # Offset R and B channels
        result[:, shift:, 2] = frame[:, :-shift, 2]
        result[:, :-shift, 0] = frame[:, shift:, 0]
        # Random block displacement
        if np.random.random() < 0.35 * intensity:
            y = np.random.randint(0, self.h - 60)
            bh = np.random.randint(10, 60)
            off = np.random.randint(-40, 40)
            block = result[y:y + bh].copy()
            result[y:y + bh] = np.roll(block, off, axis=1)
        # Scanlines
        result[::3] = (result[::3].astype(np.float32) * 0.8).astype(np.uint8)
        return result

    def _fx_vhs(self, frame, t, intensity):
        result = frame.copy()
        # Chromatic aberration
        s = 3
        result[s:, :, 2] = frame[:-s, :, 2]
        # Scanlines
        sl = np.ones(self.h, dtype=np.float32)
        sl[::2] = 0.85
        result = (result * sl[:, None, None]).astype(np.uint8)
        # Tracking noise band
        ny = int((t * 60) % self.h)
        nh = int(5 + 12 * intensity)
        y1, y2 = max(0, ny), min(self.h, ny + nh)
        noise = np.random.randint(0, 255, (y2 - y1, self.w, 3), dtype=np.uint8)
        result[y1:y2] = cv2.addWeighted(result[y1:y2], 0.3, noise, 0.7, 0)
        # Wobble
        for y in range(0, self.h, 4):
            off = int(2 * np.sin(y * 0.01 + t * 5) * intensity)
            result[y] = np.roll(result[y], off, axis=0)
        # Desaturate
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        hsv[..., 1] = (hsv[..., 1].astype(np.float32) * 0.75).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _fx_deep_fry(self, frame, t, intensity):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * (2 + intensity), 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * (1.3 + intensity * 0.5), 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        kernel = np.array([[-1, -1, -1], [-1, 9 + intensity * 4, -1], [-1, -1, -1]])
        result = cv2.filter2D(result, -1, kernel)
        noise = np.random.randint(0, int(30 * intensity) + 1, result.shape, dtype=np.uint8)
        result = cv2.add(result, noise)
        q = max(5, int(20 - intensity * 15))
        _, jpeg = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, q])
        return cv2.imdecode(jpeg, cv2.IMREAD_COLOR)

    def _fx_swirl(self, frame, t, intensity):
        strength = intensity * 3 + np.sin(t) * 0.5
        new_angle = self._angle + strength * np.exp(-self._radius * 2)
        mx = (self._cx + self._radius * self._cx * np.cos(new_angle + t * 0.5)).astype(np.float32)
        my = (self._cy + self._radius * self._cy * np.sin(new_angle + t * 0.5)).astype(np.float32)
        mx = np.clip(mx, 0, self.w - 1)
        my = np.clip(my, 0, self.h - 1)
        return cv2.remap(frame, mx, my, cv2.INTER_LINEAR)

    def _fx_mirror_quad(self, frame, t, intensity):
        h, w = frame.shape[:2]
        q = frame[:h // 2, :w // 2]
        top = np.hstack([q, cv2.flip(q, 1)])
        full = np.vstack([top, cv2.flip(top, 0)])
        return cv2.resize(full, (w, h))

    def _fx_dreamscape(self, frame, t, intensity):
        ff = frame.astype(np.float32)
        if self._dream_buf is None:
            self._dream_buf = ff.copy()
        self._dream_buf = 0.2 * ff + 0.8 * self._dream_buf
        M = cv2.getRotationMatrix2D((self._cx, self._cy), 0.3, 1.003)
        self._dream_buf = cv2.warpAffine(self._dream_buf, M, (self.w, self.h))
        result = np.clip(self._dream_buf, 0, 255).astype(np.uint8)
        # Hue rotate
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        hsv[..., 0] = (hsv[..., 0].astype(np.int16) + int(t * 15)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1].astype(np.float32) * 1.3, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # Edge glow
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        glow = cv2.GaussianBlur(edges, (15, 15), 0)
        hue = int(t * 40) % 180
        glow_hsv = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        glow_hsv[..., 0] = hue
        glow_hsv[..., 1] = 255
        glow_hsv[..., 2] = glow
        glow_bgr = cv2.cvtColor(glow_hsv, cv2.COLOR_HSV2BGR)
        return cv2.add(result, (glow_bgr.astype(np.float32) * intensity).astype(np.uint8))

    def _fx_comic(self, frame, t, intensity):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.adaptiveThreshold(
            cv2.medianBlur(gray, 7), 255,
            cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2,
        )
        div = max(1, int(64 / (1 + intensity)))
        quantized = (frame // div * div + div // 2).astype(np.uint8)
        edges3 = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return cv2.bitwise_and(quantized, edges3)

    def _fx_pixelate(self, frame, t, intensity):
        ps = max(2, int(24 * intensity))
        small = cv2.resize(frame, (self.w // ps, self.h // ps), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
