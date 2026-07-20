import cv2
import mediapipe as mp
import numpy as np


def angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)

    ba = a - b
    bc = c - b

    cos_a = np.dot(ba, bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
    )

    return np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))


def extract_clip_metrics(video_path, output_video_path=None):

    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None

    if output_video_path is not None:

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            output_video_path,
            fourcc,
            fps,
            (width, height)
        )
    frame_idx = 0

    frames = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = pose.process(rgb)
            if result.pose_landmarks:

                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    result.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

            if not result.pose_landmarks:

                frames.append({

                    # Metadata
                    "frame": frame_idx,
                    "time_sec": frame_idx / fps,

                    # Detection flag
                    "pose_detected": False,

                    # Left landmarks
                    "L_hip_x": np.nan,
                    "L_hip_y": np.nan,
                    "L_hip_visibility": np.nan,

                    "L_knee_x": np.nan,
                    "L_knee_y": np.nan,
                    "L_knee_visibility": np.nan,

                    "L_ankle_x": np.nan,
                    "L_ankle_y": np.nan,
                    "L_ankle_visibility": np.nan,

                    "L_heel_x": np.nan,
                    "L_heel_y": np.nan,
                    "L_heel_visibility": np.nan,

                    # Right landmarks
                    "R_hip_x": np.nan,
                    "R_hip_y": np.nan,
                    "R_hip_visibility": np.nan,

                    "R_knee_x": np.nan,
                    "R_knee_y": np.nan,
                    "R_knee_visibility": np.nan,

                    "R_ankle_x": np.nan,
                    "R_ankle_y": np.nan,
                    "R_ankle_visibility": np.nan,

                    "R_heel_x": np.nan,
                    "R_heel_y": np.nan,
                    "R_heel_visibility": np.nan,

                    # Derived features
                    "L_knee_angle": np.nan,
                    "R_knee_angle": np.nan,

                    "L_shank_len": np.nan,
                    "R_shank_len": np.nan,

                })

                frame_idx += 1
                if writer is not None:
                    writer.write(frame)

                continue
            
            
            lm = result.pose_landmarks.landmark

            h, w = frame.shape[:2]

            def pt(idx):
                return (
                    lm[idx].x * w,
                    lm[idx].y * h,
                    lm[idx].visibility
                )

            # LEFT

            L_hip   = pt(23)
            L_knee  = pt(25)
            L_ankle = pt(27)
            L_heel  = pt(29)

            # RIGHT

            R_hip   = pt(24)
            R_knee  = pt(26)
            R_ankle = pt(28)
            R_heel  = pt(30)

            frames.append({

                # -----------------------
                # Metadata
                # -----------------------

                "frame": frame_idx,

                "time_sec": frame_idx / fps,

                # -----------------------
                # Left landmarks
                # -----------------------

                "L_hip_x":L_hip[0],
                "L_hip_y":L_hip[1],
                "L_hip_visibility":L_hip[2],

                "L_knee_x":L_knee[0],
                "L_knee_y":L_knee[1],
                "L_knee_visibility":L_knee[2],

                "L_ankle_x":L_ankle[0],
                "L_ankle_y":L_ankle[1],
                "L_ankle_visibility":L_ankle[2],

                "L_heel_x":L_heel[0],
                "L_heel_y":L_heel[1],
                "L_heel_visibility":L_heel[2],

                # -----------------------
                # Right landmarks
                # -----------------------

                "R_hip_x":R_hip[0],
                "R_hip_y":R_hip[1],
                "R_hip_visibility":R_hip[2],

                "R_knee_x":R_knee[0],
                "R_knee_y":R_knee[1],
                "R_knee_visibility":R_knee[2],

                "R_ankle_x":R_ankle[0],
                "R_ankle_y":R_ankle[1],
                "R_ankle_visibility":R_ankle[2],

                "R_heel_x":R_heel[0],
                "R_heel_y":R_heel[1],
                "R_heel_visibility":R_heel[2],

                # -----------------------
                # Derived Features
                # -----------------------

                "L_knee_angle": angle(
                    L_hip[:2],
                    L_knee[:2],
                    L_ankle[:2]
                ),

                "R_knee_angle": angle(
                    R_hip[:2],
                    R_knee[:2],
                    R_ankle[:2]
                ),

                "L_shank_len": np.linalg.norm(
                    np.array(L_knee[:2]) -
                    np.array(L_ankle[:2])
                ),

                "R_shank_len": np.linalg.norm(
                    np.array(R_knee[:2]) -
                    np.array(R_ankle[:2])
                ),

            })

            frame_idx += 1

    cap.release()

    if writer is not None:
        writer.release()

    return frames

video_path = r"E:\campire\pose\input_vid\legal\angle1.mp4"

frames = extract_clip_metrics(video_path)

print("Frames processed:", len(frames))

print("\nFirst valid frame:")

for f in frames:
    if f is not None:
        print(f)
        break

import pandas as pd

rows = [f for f in frames if f is not None]

df = pd.DataFrame(rows)

csv_path = r"E:\campire\pose\angle2_metrics.csv"

df.to_csv(csv_path, index=False)

print(df.head())
print(f"CSV saved to:\n{csv_path}")

import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))

plt.plot(df["L_knee_angle"], label="Left Knee")
plt.plot(df["R_knee_angle"], label="Right Knee")

plt.xlabel("Frame")
plt.ylabel("Angle (degrees)")
plt.title("Knee Angle Throughout Delivery")

plt.legend()
plt.grid(True)

plt.show()

plt.figure(figsize=(12,5))

plt.plot(df["L_heel_y"], label="Left Heel")
plt.plot(df["R_heel_y"], label="Right Heel")

plt.xlabel("Frame")
plt.ylabel("Y Position (pixels)")

plt.title("Heel Vertical Position")

plt.legend()
plt.grid(True)

plt.show()