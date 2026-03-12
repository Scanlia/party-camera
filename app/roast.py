"""Face detection + Gemini AI roast overlay."""

import io
import os
import random
import threading
import time

import cv2
import numpy as np
from PIL import Image

ROAST_PROMPT = (
    "You are a comedy roast AI at a fun house party. Look at this person and "
    "give a short, funny, lighthearted roast (1-2 sentences max). Be playful "
    "and witty, not mean-spirited or offensive. Focus on their expression, "
    "vibe, or energy. Keep it PG-13 and party-appropriate. Be creative and "
    "specific to what you see. Do NOT use markdown formatting."
)

FALLBACK_ROASTS = [
    "You look like you googled 'how to look cool' and hit 'I'm Feeling Lucky'.",
    "Your vibe says 'just got here' but your face says 'been through it'.",
    "Main character energy... of a background extra.",
    "You've got the confidence of someone who hasn't checked a mirror today.",
    "Even autocorrect wouldn't know what to do with that expression.",
    "You look like you peaked in a dream once.",
    "That's the face of someone who microwaves water for tea.",
    "You look like your spirit animal is a confused golden retriever.",
    "Giving strong 'reply-all to the whole company' energy right now.",
    "You look like you'd lose a staring contest with a potato.",
    "Your face is giving 'terms and conditions I didn't read'.",
    "You look like the human equivalent of a participation trophy.",
]


class RoastEngine:
    def __init__(self, api_key: str | None, model_name: str = "gemini-2.5-flash"):
        self._client = None
        self._model = model_name
        if api_key:
            try:
                from google import genai
                self._client = genai.Client(api_key=api_key)
            except Exception as exc:
                print(f"[roast] Failed to init Gemini client: {exc}")

        self._face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # State
        self._current_roast = ""
        self._roast_until = 0.0
        self._last_roast = 0.0
        self._analysing = False
        self._analyse_start = 0.0
        self._cooldown = 8.0
        self._display_dur = 7.0

    # ------------------------------------------------------------------
    def process(self, frame: np.ndarray, t: float) -> np.ndarray:
        result = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80),
        )
        now = time.time()

        for (x, y, w, h) in faces:
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 255), 2)
            if self._analysing:
                scan_y = y + int((now - self._analyse_start) * 120) % h
                cv2.line(result, (x, scan_y), (x + w, scan_y), (0, 255, 0), 2)

        # Trigger new roast?
        if (
            len(faces) > 0
            and not self._analysing
            and now > self._roast_until
            and now - self._last_roast > self._cooldown
        ):
            largest = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest
            pad = int(0.3 * max(w, h))
            y1 = max(0, y - pad)
            y2 = min(frame.shape[0], y + h + pad)
            x1 = max(0, x - pad)
            x2 = min(frame.shape[1], x + w + pad)
            crop = frame[y1:y2, x1:x2]
            self._analysing = True
            self._analyse_start = now
            threading.Thread(target=self._fetch_roast, args=(crop,), daemon=True).start()

        # Draw overlay
        if now < self._roast_until and self._current_roast:
            self._draw_roast(result, self._current_roast)
        elif self._analysing:
            self._draw_scanning(result)

        return result

    # ------------------------------------------------------------------
    def _fetch_roast(self, face_crop: np.ndarray):
        try:
            if self._client:
                img = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=[ROAST_PROMPT, img],
                )
                self._current_roast = response.text.strip()
            else:
                time.sleep(1.5)  # fake "thinking" delay
                self._current_roast = random.choice(FALLBACK_ROASTS)
        except Exception as exc:
            print(f"[roast] API error: {exc}")
            self._current_roast = random.choice(FALLBACK_ROASTS)

        self._analysing = False
        self._last_roast = time.time()
        self._roast_until = time.time() + self._display_dur

    # ------------------------------------------------------------------
    @staticmethod
    def _draw_roast(frame: np.ndarray, text: str):
        h, w = frame.shape[:2]
        # Semi-transparent bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 130), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        # Word-wrap
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.75
        max_w = w - 40
        words = text.split()
        lines: list[str] = []
        cur = ""
        for word in words:
            test = f"{cur} {word}".strip()
            if cv2.getTextSize(test, font, scale, 2)[0][0] < max_w:
                cur = test
            else:
                if cur:
                    lines.append(cur)
                cur = word
        if cur:
            lines.append(cur)
        # Draw last 3 lines
        for i, line in enumerate(lines[-3:]):
            y = h - 90 + i * 35
            cv2.putText(frame, line, (22, y), font, scale, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, line, (20, y), font, scale, (0, 255, 255), 2, cv2.LINE_AA)

    @staticmethod
    def _draw_scanning(frame: np.ndarray):
        h, w = frame.shape[:2]
        txt = "SCANNING..."
        font = cv2.FONT_HERSHEY_SIMPLEX
        sz = cv2.getTextSize(txt, font, 1.2, 2)[0]
        x = (w - sz[0]) // 2
        cv2.putText(frame, txt, (x + 2, h - 38), font, 1.2, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, txt, (x, h - 40), font, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
