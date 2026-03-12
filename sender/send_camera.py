"""
Party Camera — Video Sender
============================
Run this on your Windows PC to send your webcam feed to the Party Camera server.

Usage:
    python send_camera.py <server_url> [camera_index]

Examples:
    python send_camera.py http://192.168.1.50:8090
    python send_camera.py http://192.168.1.50:8090 1
"""

import sys
import time

import cv2
import requests

SERVER = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8090"
CAMERA = int(sys.argv[2]) if len(sys.argv) > 2 else 0
FPS = 25
QUALITY = 80


def main():
    url = f"{SERVER.rstrip('/')}/api/frame"
    cap = cv2.VideoCapture(CAMERA)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {CAMERA}")
        sys.exit(1)

    print(f"Sending camera {CAMERA} → {url}")
    print("Press 'q' in the preview window to quit")

    frame_time = 1.0 / FPS
    errors = 0

    while True:
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            continue

        # Local preview
        cv2.imshow("Party Camera Sender [q to quit]", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Encode & send
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
        try:
            requests.post(
                url,
                files={"frame": ("f.jpg", jpeg.tobytes(), "image/jpeg")},
                timeout=2,
            )
            if errors:
                print("Reconnected ✓")
                errors = 0
        except requests.exceptions.RequestException:
            errors += 1
            if errors == 1 or errors % 30 == 0:
                print(f"Cannot reach {url} (attempt {errors})")

        elapsed = time.time() - start
        if elapsed < frame_time:
            time.sleep(frame_time - elapsed)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
