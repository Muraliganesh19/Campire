import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# Select CSV
# =====================================================

CSV_PATH = r"E:\campire\pose\output\csv\legal\angle1.csv"

# =====================================================
# Load
# =====================================================

df = pd.read_csv(CSV_PATH)

print("=" * 60)
print(df.head())

print("\n")
print("=" * 60)
print(df.describe())

print("\n")
print("=" * 60)
print(df.info())

# =====================================================
# Keep only detected pose frames
# =====================================================

if "pose_detected" in df.columns:
    df = df[df["pose_detected"] == True]

# =====================================================
# Knee Angle Plot
# =====================================================

plt.figure(figsize=(12,5))

plt.plot(df["frame"],
         df["L_knee_angle"],
         label="Left Knee")

plt.plot(df["frame"],
         df["R_knee_angle"],
         label="Right Knee")

plt.title("Knee Angle")

plt.xlabel("Frame")
plt.ylabel("Degrees")

plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# =====================================================
# Heel Position Plot
# =====================================================

plt.figure(figsize=(12,5))

plt.plot(df["frame"],
         df["L_heel_y"],
         label="Left Heel")

plt.plot(df["frame"],
         df["R_heel_y"],
         label="Right Heel")

plt.title("Heel Position")

plt.xlabel("Frame")
plt.ylabel("Pixels")

plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# =====================================================
# Shank Length
# =====================================================

plt.figure(figsize=(12,5))

plt.plot(df["frame"],
         df["L_shank_len"],
         label="Left")

plt.plot(df["frame"],
         df["R_shank_len"],
         label="Right")

plt.title("Shank Length")

plt.xlabel("Frame")
plt.ylabel("Pixels")

plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# =====================================================
# Visibility
# =====================================================

plt.figure(figsize=(12,5))

plt.plot(df["frame"],
         df["L_knee_visibility"],
         label="Left Knee")

plt.plot(df["frame"],
         df["R_knee_visibility"],
         label="Right Knee")

plt.plot(df["frame"],
         df["L_ankle_visibility"],
         label="Left Ankle")

plt.plot(df["frame"],
         df["R_ankle_visibility"],
         label="Right Ankle")

plt.title("Landmark Visibility")

plt.xlabel("Frame")
plt.ylabel("Visibility")

plt.ylim(0,1.05)

plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# =====================================================
# Simple Statistics
# =====================================================

print("\nKnee Angle Statistics")

print(df[[
    "L_knee_angle",
    "R_knee_angle"
]].describe())

print("\nHeel Position Statistics")

print(df[[
    "L_heel_y",
    "R_heel_y"
]].describe())

print("\nShank Length Statistics")

print(df[[
    "L_shank_len",
    "R_shank_len"
]].describe())