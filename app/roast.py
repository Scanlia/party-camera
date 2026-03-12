"""Face detection + Gemini AI roast overlay — Casino Royale Bond villain vibe."""

import random
import threading
import time

import cv2
import numpy as np
from PIL import Image

ROAST_PROMPT = (
    "You are an AI with the suave, cutting wit of a James Bond villain at a Casino Royale "
    "party. Look at this person and deliver a short, devastatingly funny roast (1-2 sentences max). "
    "Channel the elegant menace of Le Chiffre, the theatrical flair of Goldfinger, or the cold "
    "charm of Silva. Be witty and sharp like Bond himself would be. Reference casino, spy, or "
    "007 themes when it fits naturally. Keep it PG-13 and party-appropriate — playful not cruel. "
    "Be specific to what you see in the image. Do NOT use markdown formatting."
)

FALLBACK_ROASTS = [
    "I'd tell you the odds of looking that confused, but like Le Chiffre, I never reveal my cards.",
    "You have the confidence of a Bond villain... right before the third act.",
    "Shaken, not stirred — which is also how I'd describe your outfit choices tonight.",
    "M would classify that face as a threat to national morale.",
    "You look like you'd lose at poker even with Q feeding you the answers.",
    "The name's Bland. James Bland.",
    "MI6 called — they want their most average-looking agent back.",
    "You've got the vibe of a henchman who gets taken out in the pre-title sequence.",
    "Even Jaws had a better smile than that.",
    "You look like you'd order a martini and then ask for a straw.",
    "Double-O-Seven out of ten. And that's being generous, darling.",
    "I've seen better poker faces on a Tamagotchi.",
    "Your entrance had all the subtlety of an Aston Martin through a brick wall.",
    "You look like the kind of person who'd accidentally activate the ejector seat.",
    "Moneypenny would swipe left.",
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

        self._current_roast = ""
        self._roast_until = 0.0
        self._last_roast = 0.0
        self._analysing = False
        self._analyse_start = 0.0
        self._cooldown = 8.0
        self._display_dur = 7.0

    def process(self, frame: np.ndarray, t: float) -> np.ndarray:
        result = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80),
        )
        now = time.time()

        for (x, y, w, h) in faces:
            # Bond-style gold targeting rectangle
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 200, 255), 2)
            # Corner accents
            s = 15
            for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
                cx = x if dx < 0 else x + w
                cy = y if dy < 0 else y + h
                cv2.line(result, (cx, cy), (cx + dx * s, cy), (0, 200, 255), 2)
                cv2.line(result, (cx, cy), (cx, cy + dy * s), (0, 200, 255), 2)
            if self._analysing:
                scan_y = y + int((now - self._analyse_start) * 120) % h
                cv2.line(result, (x, scan_y), (x + w, scan_y), (0, 255, 0), 2)

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

        if now < self._roast_until and self._current_roast:
            self._draw_roast(result, self._current_roast)
        elif self._analysing:
            self._draw_scanning(result)

        return result

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
                time.sleep(1.5)
                self._current_roast = random.choice(FALLBACK_ROASTS)
        except Exception as exc:
            print(f"[roast] API error: {exc}")
            self._current_roast = random.choice(FALLBACK_ROASTS)

        self._analysing = False
        self._last_roast = time.time()
        self._roast_until = time.time() + self._display_dur

    @staticmethod
    def _draw_roast(frame: np.ndarray, text: str):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        # Dark bar with gold top line
        cv2.rectangle(overlay, (0, h - 130), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.line(frame, (0, h - 130), (w, h - 130), (0, 180, 220), 2)
        # 007 badge
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "007", (w - 70, h - 105), font, 0.6, (0, 180, 220), 2, cv2.LINE_AA)
        # Word-wrap
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
        for i, line in enumerate(lines[-3:]):
            y = h - 90 + i * 35
            cv2.putText(frame, line, (22, y), font, scale, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, line, (20, y), font, scale, (0, 220, 255), 2, cv2.LINE_AA)

    @staticmethod
    def _draw_scanning(frame: np.ndarray):
        h, w = frame.shape[:2]
        txt = "IDENTIFYING TARGET..."
        font = cv2.FONT_HERSHEY_SIMPLEX
        sz = cv2.getTextSize(txt, font, 1.0, 2)[0]
        x = (w - sz[0]) // 2
        cv2.putText(frame, txt, (x + 2, h - 38), font, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, txt, (x, h - 40), font, 1.0, (0, 200, 255), 2, cv2.LINE_AA)
