import h5py
import matplotlib.pyplot as plt

file_path = "brats_dataset/BraTS2020_training_data/content/data/volume_1_slice_0.h5"

with h5py.File(file_path, "r") as f:

    image = f["image"][:]
    mask = f["mask"][:]

print("Image shape:", image.shape)
print("Mask shape:", mask.shape)

# Display first MRI channel
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image[:, :, 0], cmap="gray")
plt.title("MRI Image")
plt.axis("off")

# Display mask
plt.subplot(1, 2, 2)
plt.imshow(mask)
plt.title("Tumor Mask")
plt.axis("off")

plt.show()