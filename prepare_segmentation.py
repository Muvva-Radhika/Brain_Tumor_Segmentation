import os
import glob
import random

DATA_FOLDER = "brats_dataset/BraTS2020_training_data/content/data"

# Find all H5 files
files = glob.glob(os.path.join(DATA_FOLDER, "*.h5"))

print("Total H5 files:", len(files))

# Group files according to volume number
volumes = {}

for file_path in files:

    file_name = os.path.basename(file_path)

    # Example:
    # volume_1_slice_100.h5
    parts = file_name.split("_")

    volume_number = parts[1]

    if volume_number not in volumes:
        volumes[volume_number] = []

    volumes[volume_number].append(file_path)


print("Total volumes:", len(volumes))


# Convert dictionary to list
volume_ids = list(volumes.keys())

# Shuffle volumes
random.seed(42)
random.shuffle(volume_ids)


# 80% training and 20% validation
split_index = int(0.8 * len(volume_ids))

train_volumes = volume_ids[:split_index]
val_volumes = volume_ids[split_index:]


# Create training file list
train_files = []

for volume_id in train_volumes:
    train_files.extend(volumes[volume_id])


# Create validation file list
val_files = []

for volume_id in val_volumes:
    val_files.extend(volumes[volume_id])


# Save training file paths
with open("train_files.txt", "w") as f:

    for file_path in train_files:
        f.write(file_path + "\n")


# Save validation file paths
with open("val_files.txt", "w") as f:

    for file_path in val_files:
        f.write(file_path + "\n")


print()
print("Training volumes:", len(train_volumes))
print("Validation volumes:", len(val_volumes))

print("Training slices:", len(train_files))
print("Validation slices:", len(val_files))

print()
print("Training and validation files created successfully.")