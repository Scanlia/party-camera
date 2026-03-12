"""Generative psychedelic pattern effects — pure NumPy, no camera needed."""

import cv2
import numpy as np


class PatternEngine:
    def __init__(self, width: int = 1280, height: int = 720):
        self.w = width
        self.h = height
        x = np.linspace(-1, 1, width, dtype=np.float32)
        y = np.linspace(-1, 1, height, dtype=np.float32)
        self.X, self.Y = np.meshgrid(x, y)
        self.R = np.sqrt(self.X ** 2 + self.Y ** 2)
        self.THETA = np.arctan2(self.Y, self.X)
        self._fire_buf = None

    def generate(self, name: str, t: float, speed: float = 1.0,
                 intensity: float = 0.7) -> np.ndarray:
        fn = getattr(self, f"_pat_{name}", None)
        if fn is None:
            return self._pat_plasma(t * speed, intensity)
        return fn(t * speed, intensity)

    @staticmethod
    def _to_bgr(r, g, b) -> np.ndarray:
        return np.stack([
            np.clip(b, 0, 255).astype(np.uint8),
            np.clip(g, 0, 255).astype(np.uint8),
            np.clip(r, 0, 255).astype(np.uint8),
        ], axis=-1)

    # === ORIGINAL PATTERNS ============================================

    def _pat_plasma(self, t, intensity):
        X, Y = self.X, self.Y
        v1 = np.sin(X * 10 + t)
        v2 = np.sin(10 * (X * np.sin(t / 2) + Y * np.cos(t / 3)) + t)
        v3 = np.sin(np.sqrt(100 * ((X + np.sin(t / 3)) ** 2 +
                                     (Y + np.cos(t / 2)) ** 2) + 1) + t)
        v4 = np.sin(X * 5 * np.cos(t / 4) + Y * 5 * np.sin(t / 3))
        v = (v1 + v2 + v3 + v4) * 0.25 * intensity
        r = np.sin(v * np.pi) * 128 + 127
        g = np.sin(v * np.pi + 2 * np.pi / 3) * 128 + 127
        b = np.sin(v * np.pi + 4 * np.pi / 3) * 128 + 127
        return self._to_bgr(r, g, b)

    def _pat_hypnotic(self, t, intensity):
        R, TH = self.R, self.THETA
        v = np.sin(R * 20 - t * 3) * np.cos(TH * 3 + t)
        v2 = np.sin(R * 12 + t * 2) * np.sin(TH * 5 - t * 0.7)
        v = (v + v2) * 0.5 * intensity
        r = np.sin(v * np.pi + t) * 128 + 127
        g = np.sin(v * np.pi + t + 2) * 128 + 127
        b = np.sin(v * np.pi + t + 4) * 128 + 127
        return self._to_bgr(r, g, b)

    def _pat_fire(self, t, intensity):
        if self._fire_buf is None:
            self._fire_buf = np.zeros((self.h, self.w), dtype=np.float32)
        self._fire_buf[-2:, :] = np.random.random((2, self.w)).astype(np.float32) * intensity
        below = np.roll(self._fire_buf, -1, axis=0)
        bl = np.roll(below, -1, axis=1)
        br = np.roll(below, 1, axis=1)
        b2 = np.roll(self._fire_buf, -2, axis=0)
        self._fire_buf = (below + bl + br + b2) / 4.02
        self._fire_buf[-2:, :] = np.random.random((2, self.w)).astype(np.float32) * intensity
        fire = np.clip(self._fire_buf * 255, 0, 255).astype(np.uint8)
        return cv2.applyColorMap(fire, cv2.COLORMAP_HOT)

    def _pat_lava_lamp(self, t, intensity):
        n_blobs = 6
        field = np.zeros((self.h, self.w), dtype=np.float32)
        for i in range(n_blobs):
            bx = np.sin(t * (0.3 + i * 0.1) + i * 2.1) * 0.4
            by = np.cos(t * (0.2 + i * 0.15) + i * 1.7) * 0.45
            dist = np.sqrt((self.X - bx) ** 2 + (self.Y - by) ** 2) + 0.01
            field += (0.25 * intensity) / dist
        v = np.clip(field, 0, 1)
        r = np.sin(v * np.pi * 2 + t) * 180 + 75
        g = np.sin(v * np.pi * 2 + t + 2) * 80 + 30
        b = np.sin(v * np.pi * 2 + t + 4) * 180 + 75
        return self._to_bgr(r, g, b)

    def _pat_tunnel(self, t, intensity):
        R = self.R + 0.001
        TH = self.THETA
        u = 1.0 / R + t * 0.5
        v = TH / np.pi
        r = np.sin(u * 10 + t) * 127 + 128
        g = np.sin(v * 10 + t * 0.7) * 127 + 128
        b = np.sin((u + v) * 5 + t * 1.3) * 127 + 128
        brightness = np.clip(1.0 / (R * 3 + 0.1), 0, 1) * intensity
        return self._to_bgr(r * brightness, g * brightness, b * brightness)

    def _pat_fractal_spin(self, t, intensity):
        X, Y = self.X, self.Y
        ct, st = np.cos(t * 0.3), np.sin(t * 0.3)
        rx = X * ct - Y * st
        ry = X * st + Y * ct
        zr = rx * 2
        zi = ry * 2
        for _ in range(8):
            zr2 = zr * zr - zi * zi + rx * np.cos(t * 0.5)
            zi2 = 2 * zr * zi + ry * np.sin(t * 0.5)
            zr, zi = zr2, zi2
            zr = np.clip(zr, -4, 4)
            zi = np.clip(zi, -4, 4)
        v = np.sqrt(zr ** 2 + zi ** 2)
        v = np.clip(v / 4.0, 0, 1) * intensity
        r = np.sin(v * np.pi * 3 + t) * 128 + 127
        g = np.sin(v * np.pi * 3 + t + 2) * 128 + 127
        b = np.sin(v * np.pi * 3 + t + 4) * 128 + 127
        return self._to_bgr(r, g, b)

    def _pat_rainbow_wave(self, t, intensity):
        X, Y = self.X, self.Y
        v = np.sin(X * 5 + t * 2) + np.sin(Y * 5 + t * 1.5)
        v += np.sin((X + Y) * 3 + t) + np.sin(np.sqrt(X ** 2 + Y ** 2) * 5 - t * 2)
        v = v * 0.25 * intensity
        hue = ((v + 1) * 90 + t * 30) % 180
        hsv = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        hsv[..., 0] = hue.astype(np.uint8)
        hsv[..., 1] = 255
        hsv[..., 2] = np.clip((np.abs(v) + 0.3) * 255, 50, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _pat_aurora(self, t, intensity):
        X, Y = self.X, self.Y
        v1 = np.sin(X * 3 + t * 0.7) * np.cos(Y * 2 - t * 0.3)
        v2 = np.sin(X * 5 - t * 0.5 + Y * 3) * 0.5
        v3 = np.cos(Y * 8 + np.sin(X * 2 + t) * 2)
        v = (v1 + v2 + v3) * 0.33 * intensity
        r = np.clip(np.sin(v * np.pi + 4) * 100 + 50, 0, 255)
        g = np.clip(np.sin(v * np.pi) * 150 + 100, 0, 255)
        b = np.clip(np.sin(v * np.pi + 2) * 130 + 80, 0, 255)
        fade = np.clip((self.Y + 1) * 0.8, 0.1, 1.0)
        return self._to_bgr(r * fade, g * fade, b * fade)

    def _pat_wormhole(self, t, intensity):
        R = self.R + 0.001
        TH = self.THETA
        u = 1.0 / R + t
        spiral = TH + t * 0.5 + np.log(R + 0.1) * 3
        v1 = np.sin(u * 8) * np.sin(spiral * 4)
        v2 = np.cos(u * 5 + t) * np.cos(spiral * 3 - t * 0.5)
        v = (v1 + v2) * 0.5 * intensity
        brightness = np.clip(1.0 / (R * 2 + 0.05), 0, 1)
        r = np.clip(np.sin(v * np.pi + t) * 200 + 55, 0, 255) * brightness
        g = np.clip(np.sin(v * np.pi + t + 2) * 150 + 55, 0, 255) * brightness
        b = np.clip(np.sin(v * np.pi + t + 4) * 250 + 55, 0, 255) * brightness
        return self._to_bgr(r, g, b)

    # === BOND-THEMED PATTERNS ========================================

    def _pat_gun_barrel_pattern(self, t, intensity):
        """Animated Bond gun barrel spiral."""
        R, TH = self.R, self.THETA
        # Spiralling rifling
        spiral = TH + R * 10 - t * 2
        rifling = np.sin(spiral * 8) * 0.5 + 0.5
        # Pulsing circles
        circles = np.sin(R * 30 - t * 4) * 0.5 + 0.5
        v = (rifling * 0.6 + circles * 0.4) * intensity
        # Classic Bond white-on-black
        brightness = np.clip(v * 255, 0, 255)
        # Centre bright spot
        spot = np.exp(-R ** 2 * 8) * 255 * (0.5 + 0.5 * np.sin(t * 3))
        brightness = np.clip(brightness + spot, 0, 255)
        gray = brightness.astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def _pat_golden_rings(self, t, intensity):
        """Expanding golden rings — Goldeneye/Spectre vibe."""
        R = self.R
        # Multiple ring sets at different speeds
        v1 = np.sin(R * 15 - t * 3) ** 2
        v2 = np.sin(R * 10 + t * 2) ** 2
        v3 = np.sin(R * 20 - t * 5) ** 2
        v = (v1 + v2 * 0.5 + v3 * 0.3) / 1.8 * intensity
        # Gold palette
        r = np.clip(v * 255, 0, 255)
        g = np.clip(v * 200, 0, 255)
        b = np.clip(v * 50, 0, 255)
        # Centre glow
        glow = np.exp(-R ** 2 * 4) * intensity
        r = np.clip(r + glow * 200, 0, 255)
        g = np.clip(g + glow * 160, 0, 255)
        b = np.clip(b + glow * 40, 0, 255)
        return self._to_bgr(r, g, b)

    def _pat_casino_felt(self, t, intensity):
        """Casino card table patterns — suits, chips, roulette."""
        X, Y = self.X, self.Y
        # Green felt base
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        frame[..., 1] = int(60 * intensity)  # dark green
        # Diamond/grid pattern
        grid = np.sin(X * 20 + t) * np.sin(Y * 20 + t * 0.7)
        grid = (grid * 0.5 + 0.5) * intensity
        frame[..., 1] = np.clip(frame[..., 1].astype(np.float32) + grid * 100, 0, 255).astype(np.uint8)
        # Roulette wheel in centre
        R, TH = self.R, self.THETA
        wheel_mask = R < 0.5
        sector = ((TH + t) / (2 * np.pi) * 37).astype(int) % 37
        is_red = np.isin(sector, [1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36])
        # Red sectors
        red_mask = wheel_mask & is_red
        frame[red_mask] = [0, 0, int(200 * intensity)]
        # Black sectors
        black_mask = wheel_mask & ~is_red & (sector > 0)
        frame[black_mask] = [int(30 * intensity), int(30 * intensity), int(30 * intensity)]
        # Gold trim ring
        ring_mask = (R > 0.48) & (R < 0.52)
        frame[ring_mask] = [0, int(180 * intensity), int(220 * intensity)]
        # Gold ball
        ball_angle = t * 3
        ball_r = 0.45
        bx = int(self.w / 2 + ball_r * self.w / 2 * np.cos(ball_angle))
        by = int(self.h / 2 + ball_r * self.h / 2 * np.sin(ball_angle))
        cv2.circle(frame, (bx, by), 6, (200, 255, 255), -1)
        return frame
