"""Video clip player — loops through .mp4 files in a directory."""

import os
import threading

import cv2
import numpy as np


class ClipPlayer:
    def __init__(self, clip_dir: str):
        self._dir = clip_dir
        self._clips: list[str] = []
        self._current_idx = 0
        self._cap: cv2.VideoCapture | None = None
        self._lock = threading.Lock()
        self._scan()

    def _scan(self):
        if not os.path.isdir(self._dir):
            return
        self._clips = sorted([
            os.path.join(self._dir, f)
            for f in os.listdir(self._dir)
            if f.lower().endswith((".mp4", ".mkv", ".avi", ".webm", ".mov"))
        ])
        if self._clips:
            self._open(0)

    def _open(self, idx: int):
        if self._cap:
            self._cap.release()
        self._current_idx = idx % max(1, len(self._clips))
        self._cap = cv2.VideoCapture(self._clips[self._current_idx])
        print(f"[clips] Playing: {self._clips[self._current_idx]}")

    def clip_count(self) -> int:
        return len(self._clips)

    def get_frame(self, width: int, height: int) -> np.ndarray | None:
        with self._lock:
            if not self._clips or not self._cap:
                return None
            ret, frame = self._cap.read()
            if not ret:
                # Next clip (or loop)
                self._open(self._current_idx + 1)
                ret, frame = self._cap.read()
                if not ret:
                    return None
            return cv2.resize(frame, (width, height))
