import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# Load trained U-Net model
model = tf.keras.models.load_model(
    "unet_brain_tumor.keras",
    compile=False
)

print("U-Net model loaded successfully.")


# MRI slice file
file_path = "brats_dataset/BraTS2020_training_data/content/data/volume_1_slice_100.h5"


# Read MRI image
with h5py.File(file_path, "r") as f:

    image = np.array(f["image"])


print("Original image shape:", image.shape)


# Normalize image
image = image.astype(np.float32)

image_min = image.min()
image_max = image.max()

if image_max > image_min:
    image = (image - image_min) / (image_max - image_min)


# Add batch dimension
image_input = np.expand_dims(image, axis=0)

print("Input shape:", image_input.shape)


# Predict tumor mask
prediction = model.predict(image_input, verbose=0)

print("Prediction shape:", prediction.shape)


# Remove batch and channel dimensions
predicted_mask = prediction[0, :, :, 0]


# Convert probability to binary mask
predicted_mask = (predicted_mask > 0.5).astype(np.uint8)


print("Predicted mask shape:", predicted_mask.shape)
print("Predicted mask pixels:", np.sum(predicted_mask))


# Display MRI and predicted mask
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image[:, :, 0], cmap="gray")
plt.title("MRI Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(predicted_mask, cmap="gray")
plt.title("Predicted Tumor Mask")
plt.axis("off")

plt.tight_layout()
plt.show()