import h5py
import numpy as np
import matplotlib.pyplot as plt

file_path = "brats_dataset/BraTS2020_training_data/content/data/volume_1_slice_100.h5"

with h5py.File(file_path, "r") as f:
    image = f["image"][:]
    mask = f["mask"][:]

print("Image shape:", image.shape)
print("Mask shape:", mask.shape)

print("Mask minimum:", mask.min())
print("Mask maximum:", mask.max())
print("Mask unique values:", np.unique(mask))

# Combine the 3 mask channels
mask_combined = np.max(mask, axis=2)

print("Combined mask shape:", mask_combined.shape)
print("Combined mask sum:", mask_combined.sum())

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image[:, :, 0], cmap="gray")
plt.title("MRI Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(mask_combined, cmap="gray")
plt.title("Tumor Mask")
plt.axis("off")

plt.show()