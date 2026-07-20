import os
import pandas as pd

from pose_extractor import extract_clip_metrics

# -----------------------------
# Paths
# -----------------------------

INPUT_ROOT = r"E:\campire\pose\input_vid"
OUTPUT_ROOT = r"E:\campire\pose\output"

CLASSES = ["legal", "illegal"]

# -----------------------------
# Process each class
# -----------------------------

for cls in CLASSES:

    input_dir = os.path.join(INPUT_ROOT, cls)

    csv_dir = os.path.join(OUTPUT_ROOT, "csv", cls)
    pose_dir = os.path.join(OUTPUT_ROOT, "pose_video", cls)

    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(pose_dir, exist_ok=True)

    videos = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    ])

    print(f"\n{'='*60}")
    print(f"{cls.upper()} : {len(videos)} videos")
    print(f"{'='*60}")

    for i, video_name in enumerate(videos, start=1):

        video_path = os.path.join(input_dir, video_name)

        base_name = os.path.splitext(video_name)[0]

        csv_path = os.path.join(
            csv_dir,
            base_name + ".csv"
        )

        pose_video_path = os.path.join(
            pose_dir,
            base_name + "_pose.mp4"
        )

        print(f"\n[{i}/{len(videos)}] {video_name}")

        try:

            frames = extract_clip_metrics(
                video_path,
                output_video_path=pose_video_path
            )

            df = pd.DataFrame(frames)

            df.to_csv(csv_path, index=False)

            print("   ✓ CSV saved")
            print("   ✓ Pose video saved")

        except Exception as e:

            print(f"   ✗ ERROR : {e}")

print("\nAll videos processed successfully.")