import os

# Folder containing downloaded videos
folder = r"E:\campire\pose\input_vid\legal"

# Change this to any prefix you want
prefix = "angle"

# Starting number
start = 1

videos = sorted([
    f for f in os.listdir(folder)
    if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))
])

for i, old_name in enumerate(videos, start=start):

    ext = os.path.splitext(old_name)[1]

    new_name = f"{prefix}{i}{ext}"

    old_path = os.path.join(folder, old_name)
    new_path = os.path.join(folder, new_name)

    os.rename(old_path, new_path)

    print(f"{old_name}  -->  {new_name}")

print("Done.")