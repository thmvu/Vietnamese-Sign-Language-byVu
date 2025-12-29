import cv2
import os

VIDEO_DIR = "Dataset/Videos"
FRAME_DIR = "Dataset/Frames"
os.makedirs(FRAME_DIR, exist_ok=True)

TARGET_FRAMES = 60

for video in os.listdir(VIDEO_DIR):
    if not video.endswith(".mp4"):
        continue

    video_path = os.path.join(VIDEO_DIR, video)
    cap = cv2.VideoCapture(video_path)

    out_dir = os.path.join(FRAME_DIR, video.replace(".mp4", ""))
    os.makedirs(out_dir, exist_ok=True)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total // TARGET_FRAMES, 1)

    idx, saved = 0, 0
    while cap.isOpened() and saved < TARGET_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frame = cv2.resize(frame, (480, 480))
            cv2.imwrite(f"{out_dir}/{saved:03d}.jpg", frame)
            saved += 1
        idx += 1

    cap.release()
    print(f"✅ {video}: {saved} frames")
