"""Camera video effects — all NumPy / OpenCV, no extra dependencies."""

import cv2
import numpy as np


class EffectEngine:
    def __init__(self, width: int = 1280, height: int = 720):
        self.w = width
        self.h = height
        ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
        self._map_x = xs
        self._map_y = ys
        self._cx = width / 2.0
        self._cy = height / 2.0
        self._nx = (xs - self._cx) / self._cx
        self._ny = (ys - self._cy) / self._cy
        self._radius = np.sqrt(self._nx ** 2 + self._ny ** 2)
        self._angle = np.arctan2(self._ny, self._nx)
        self._feedback_buf = None
        self._dream_buf = None
        self._blood_y = 0

    def apply(self, frame: np.ndarray, name: str, t: float,
              intensity: float = 0.7, speed: float = 1.0) -> np.ndarray:
        fn = getattr(self, f"_fx_{name}", None)
        if fn is None:
            return frame
        return fn(frame, t * speed, intensity)

    # === ORIGINAL EFFECTS =============================================

    def _fx_neon_edges(self, frame, t, intensity):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
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
        M = cv2.getRotationMatrix2D((self._cx, self._cy), 0.4, 1.004)
        self._feedback_buf = cv2.warpAffine(self._feedback_buf, M, (self.w, self.h))
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
        result[:, shift:, 2] = frame[:, :-shift, 2]
        result[:, :-shift, 0] = frame[:, shift:, 0]
        if np.random.random() < 0.35 * intensity:
            y = np.random.randint(0, self.h - 60)
            bh = np.random.randint(10, 60)
            off = np.random.randint(-40, 40)
            block = result[y:y + bh].copy()
            result[y:y + bh] = np.roll(block, off, axis=1)
        result[::3] = (result[::3].astype(np.float32) * 0.8).astype(np.uint8)
        return result

    def _fx_vhs(self, frame, t, intensity):
        result = frame.copy()
        s = 3
        result[s:, :, 2] = frame[:-s, :, 2]
        sl = np.ones(self.h, dtype=np.float32)
        sl[::2] = 0.85
        result = (result * sl[:, None, None]).astype(np.uint8)
        ny = int((t * 60) % self.h)
        nh = int(5 + 12 * intensity)
        y1, y2 = max(0, ny), min(self.h, ny + nh)
        noise = np.random.randint(0, 255, (y2 - y1, self.w, 3), dtype=np.uint8)
        result[y1:y2] = cv2.addWeighted(result[y1:y2], 0.3, noise, 0.7, 0)
        for y in range(0, self.h, 4):
            off = int(2 * np.sin(y * 0.01 + t * 5) * intensity)
            result[y] = np.roll(result[y], off, axis=0)
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
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        hsv[..., 0] = (hsv[..., 0].astype(np.int16) + int(t * 15)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1].astype(np.float32) * 1.3, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
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

    # === BOND / CASINO ROYALE EFFECTS =================================

    def _fx_gun_barrel(self, frame, t, intensity):
        """Classic James Bond gun barrel view — circular aperture following face/centre."""
        result = np.zeros_like(frame)
        mask = np.zeros((self.h, self.w), dtype=np.uint8)
        # Pulsing barrel radius
        base_r = int(min(self.w, self.h) * 0.3)
        pulse = int(np.sin(t * 2) * 15 * intensity)
        radius = base_r + pulse
        cx, cy = self.w // 2, self.h // 2
        cv2.circle(mask, (cx, cy), radius, 255, -1)
        # Rifling lines
        n_lines = 8
        for i in range(n_lines):
            angle = (i / n_lines) * 2 * np.pi + t * 0.5
            x1 = int(cx + radius * np.cos(angle))
            y1 = int(cy + radius * np.sin(angle))
            x2 = int(cx + (radius + 800) * np.cos(angle))
            y2 = int(cy + (radius + 800) * np.sin(angle))
            cv2.line(result, (x1, y1), (x2, y2), (40, 40, 40), 3)
        # Apply barrel mask
        result[mask > 0] = frame[mask > 0]
        # White border ring
        cv2.circle(result, (cx, cy), radius, (200, 200, 200), 3)
        return result

    def _fx_golden_eye(self, frame, t, intensity):
        """Gold tint + lens flare + vignette — Goldeneye look."""
        # Gold colour grade
        gold = frame.copy().astype(np.float32)
        gold[..., 0] *= 0.3  # reduce blue
        gold[..., 1] *= 0.8  # slightly reduce green
        gold[..., 2] *= 1.3  # boost red
        gold = np.clip(gold, 0, 255).astype(np.uint8)
        # Sepia-gold overlay
        hsv = cv2.cvtColor(gold, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = 20  # gold hue
        hsv[..., 1] = np.clip(hsv[..., 1] * (0.5 + intensity), 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        # Lens flare
        flare_x = int(self.w * (0.3 + 0.4 * np.sin(t * 0.3)))
        flare_y = int(self.h * 0.3)
        for r, alpha in [(80, 0.15), (40, 0.3), (15, 0.5)]:
            overlay = result.copy()
            cv2.circle(overlay, (flare_x, flare_y), r, (0, 200, 255), -1)
            cv2.addWeighted(overlay, alpha * intensity, result, 1 - alpha * intensity, 0, result)
        # Vignette
        result = self._vignette(result, intensity * 0.6)
        return result

    def _fx_casino_hud(self, frame, t, intensity):
        """Casino Royale HUD — targeting reticle + data overlays."""
        result = frame.copy()
        cx, cy = self.w // 2, self.h // 2
        # Targeting circles
        for r in [80, 130, 200]:
            angle_off = t * 30
            cv2.ellipse(result, (cx, cy), (r, r), angle_off, 0, 360, (0, 255, 0), 1)
        # Crosshairs
        cv2.line(result, (cx - 220, cy), (cx - 50, cy), (0, 255, 0), 1)
        cv2.line(result, (cx + 50, cy), (cx + 220, cy), (0, 255, 0), 1)
        cv2.line(result, (cx, cy - 150), (cx, cy - 50), (0, 255, 0), 1)
        cv2.line(result, (cx, cy + 50), (cx, cy + 150), (0, 255, 0), 1)
        # Corner brackets
        s = 30
        for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
            bx = cx + dx * 200
            by = cy + dy * 120
            cv2.line(result, (bx, by), (bx + dx * s, by), (0, 255, 0), 2)
            cv2.line(result, (bx, by), (bx, by + dy * s), (0, 255, 0), 2)
        # Fake data readouts
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result, "MI6 SURVEILLANCE", (20, 30), font, 0.5, (0, 255, 0), 1)
        cv2.putText(result, f"TRACKING: ACTIVE", (20, 55), font, 0.4, (0, 255, 0), 1)
        cv2.putText(result, f"THREAT: {['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'][int(t) % 4]}",
                    (20, 80), font, 0.4, (0, 200, 255), 1)
        cv2.putText(result, f"007", (self.w - 80, 40), font, 1.0, (0, 200, 255), 2)
        # Scrolling data bar at bottom
        bar_y = self.h - 25
        scroll = int(t * 200) % 2000
        data_text = "// CLASSIFIED // CASINO ROYALE // FIELD REPORT // EYES ONLY // " * 3
        cv2.putText(result, data_text, (-scroll, bar_y), font, 0.35, (0, 180, 0), 1)
        return result

    def _fx_silhouette(self, frame, t, intensity):
        """Bond title sequence silhouette — high-contrast with colour washes."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, int(100 + 30 * np.sin(t)), 255, cv2.THRESH_BINARY)
        # Coloured background wash
        hue = int(t * 15) % 180
        bg = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        bg[..., 0] = hue
        bg[..., 1] = 200
        bg[..., 2] = int(180 * intensity)
        bg_bgr = cv2.cvtColor(bg, cv2.COLOR_HSV2BGR)
        # Silhouette is black, background is coloured
        result = bg_bgr.copy()
        result[thresh > 0] = 0
        return result

    def _fx_martini_vision(self, frame, t, intensity):
        """Blurry, doubled, tilting 'drunk' effect — shaken not stirred."""
        # Tilt
        angle = np.sin(t * 1.5) * 5 * intensity
        M = cv2.getRotationMatrix2D((self._cx, self._cy), angle, 1.0)
        tilted = cv2.warpAffine(frame, M, (self.w, self.h))
        # Double vision
        shift = max(1, int(15 * intensity * (0.5 + 0.5 * np.sin(t * 2))))
        layer2 = np.roll(tilted, shift, axis=1)
        result = cv2.addWeighted(tilted, 0.6, layer2, 0.4, 0)
        # Soft blur
        k = max(1, int(7 * intensity)) | 1
        result = cv2.GaussianBlur(result, (k, k), 0)
        # Warm tint
        result = result.astype(np.float32)
        result[..., 2] = np.clip(result[..., 2] * 1.1, 0, 255)
        result[..., 0] = result[..., 0] * 0.9
        return result.astype(np.uint8)

    def _fx_tuxedo(self, frame, t, intensity):
        """High-contrast B&W with gold accent — black tie elegance."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # High contrast
        clahe = cv2.createCLAHE(clipLimit=3.0 + intensity * 3, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        result = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        # Gold accent on bright areas
        bright_mask = enhanced > int(180 - 30 * intensity)
        gold = np.zeros_like(result)
        gold[bright_mask] = [0, 180, 255]  # BGR gold
        result = cv2.addWeighted(result, 0.85, gold, 0.15 * intensity, 0)
        # Vignette
        result = self._vignette(result, 0.5)
        return result

    def _fx_blood_drip(self, frame, t, intensity):
        """Blood drip down the screen — Casino Royale opening."""
        result = frame.copy()
        # Red overlay creeping from top
        drip_progress = (np.sin(t * 0.3) + 1) * 0.5 * intensity
        drip_y = int(self.h * drip_progress)
        if drip_y > 5:
            overlay = result[:drip_y].copy()
            red_tint = np.zeros_like(overlay)
            red_tint[..., 2] = 180  # Red
            result[:drip_y] = cv2.addWeighted(overlay, 0.5, red_tint, 0.5, 0)
            # Drip edge — wavy
            for x in range(0, self.w, 8):
                dy = int(np.sin(x * 0.03 + t * 2) * 15 + np.sin(x * 0.07 + t * 3) * 8)
                y_end = min(self.h - 1, drip_y + dy)
                if y_end > drip_y:
                    cv2.line(result, (x, drip_y), (x, y_end), (0, 0, 200), 3)
            # Drip drops
            for i in range(5):
                dx = int((i * 251 + t * 100) % self.w)
                drop_y = min(self.h - 1, drip_y + int(np.sin(i + t * 2) * 40 + 30))
                cv2.circle(result, (dx, drop_y), 4, (0, 0, 180), -1)
        return result

    # === Helpers =====================================================

    def _vignette(self, frame: np.ndarray, strength: float) -> np.ndarray:
        mask = 1.0 - self._radius * strength
        mask = np.clip(mask, 0, 1).astype(np.float32)
        return (frame.astype(np.float32) * mask[..., None]).astype(np.uint8)
