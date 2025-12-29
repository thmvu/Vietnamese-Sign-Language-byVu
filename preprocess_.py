import os
import cv2
import json
import numpy as np
import pandas as pd
import mediapipe as mp
from tqdm import tqdm

# ================= CONFIG =================
FRAME_DIR = "Dataset/Frames"
LABEL_CSV = "Dataset/Text/label_clean.csv"
LABEL_MAP_PATH = "Logs/label_map.json"
OUTPUT_DIR = "Data"

SEQUENCE_LEN = 60
FEATURE_DIM = 258

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= LOAD LABEL MAP =================
with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
    label_map = json.load(f)

df = pd.read_csv(LABEL_CSV)

# ================= MEDIAPIPE =================
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=True,
    model_complexity=1
)

def extract_keypoints(results):
    pose = np.zeros(33 * 4)
    lh   = np.zeros(21 * 3)
    rh   = np.zeros(21 * 3)

    if results.pose_landmarks:
        pose = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]
        ).flatten()

    if results.left_hand_landmarks:
        lh = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
        ).flatten()

    if results.right_hand_landmarks:
        rh = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]
        ).flatten()

    features = np.concatenate([pose, lh, rh])
    return features[:FEATURE_DIM]

# ================= PROCESS =================
for _, row in tqdm(df.iterrows(), total=len(df)):
    video = row["VIDEO"].replace(".mp4", "")
    label = row["LABEL"]

    frame_folder = os.path.join(FRAME_DIR, video)
    if not os.path.exists(frame_folder):
        print(f"⚠️ Missing frames: {video}")
        continue

    sequence = []

    frames = sorted(os.listdir(frame_folder))[:SEQUENCE_LEN]
    for img in frames:
        img_path = os.path.join(frame_folder, img)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = holistic.process(image)
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)

    # Padding nếu thiếu frame
    while len(sequence) < SEQUENCE_LEN:
        sequence.append(np.zeros(FEATURE_DIM))

    sequence = np.array(sequence, dtype=np.float32)
    label_id = label_map[label]

    np.savez(
        os.path.join(OUTPUT_DIR, f"{video}.npz"),
        sequence=sequence,
        label=label_id
    )

holistic.close()
print("✅ DONE: Generated .npz files")
