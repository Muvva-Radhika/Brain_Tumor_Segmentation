import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# =========================================================
# 1. LOAD TRAINED U-NET
# =========================================================

model = tf.keras.models.load_model(
    "unet_brain_tumor.keras",
    compile=False
)

print("U-Net loaded successfully.")


# =========================================================
# 2. SELECT MRI SLICE
# =========================================================

file_path = (
    "brats_dataset/BraTS2020_training_data/"
    "content/data/volume_1_slice_100.h5"
)


# =========================================================
# 3. READ MRI IMAGE AND ACTUAL MASK
# =========================================================

with h5py.File(file_path, "r") as f:

    image = np.array(f["image"])
    mask = np.array(f["mask"])


print("Image shape:", image.shape)
print("Mask shape:", mask.shape)


# =========================================================
# 4. NORMALIZE MRI IMAGE
# =========================================================

image = image.astype(np.float32)

image_min = image.min()
image_max = image.max()

if image_max > image_min:
    image = (image - image_min) / (image_max - image_min)


# =========================================================
# 5. PREPARE IMAGE FOR U-NET
# =========================================================

# Original:
# (240, 240, 4)

# Add batch dimension:
# (1, 240, 240, 4)

image_input = np.expand_dims(image, axis=0)

print("U-Net input shape:", image_input.shape)


# =========================================================
# 6. GET U-NET PREDICTION
# =========================================================

prediction = model.predict(
    image_input,
    verbose=0
)


# Prediction shape:
# (1, 240, 240, 1)

predicted_probability = prediction[0, :, :, 0]


# =========================================================
# 7. CHECK PREDICTION VALUES
# =========================================================

print()
print("Prediction minimum:", predicted_probability.min())
print("Prediction maximum:", predicted_probability.max())
print("Prediction mean:", predicted_probability.mean())


# =========================================================
# 8. PREPARE ACTUAL TUMOR MASK
# =========================================================

# Original mask:
# (240, 240, 3)

# Combine the 3 mask channels
actual_mask = np.max(mask, axis=2)

# Convert to binary
actual_mask = (
    actual_mask > 0
).astype(np.uint8)


print("Actual tumor pixels:", np.sum(actual_mask))


# =========================================================
# 9. CREATE BINARY U-NET MASK
# =========================================================

threshold = 0.5

predicted_mask = (
    predicted_probability >= threshold
).astype(np.uint8)


print("Threshold:", threshold)
print("Predicted tumor pixels:", np.sum(predicted_mask))


# =========================================================
# 10. DISPLAY MRI IMAGE
# =========================================================

plt.figure(figsize=(7, 7))

plt.imshow(
    image[:, :, 0],
    cmap="gray"
)

plt.title("MRI Image", fontsize=16)
plt.axis("off")

plt.tight_layout()
plt.show()


# =========================================================
# 11. DISPLAY ACTUAL TUMOR MASK
# =========================================================

plt.figure(figsize=(7, 7))

plt.imshow(
    actual_mask,
    cmap="gray"
)

plt.title("Actual Tumor Mask", fontsize=16)
plt.axis("off")

plt.tight_layout()
plt.show()


# =========================================================
# 12. DISPLAY U-NET PROBABILITY MAP
# =========================================================

plt.figure(figsize=(7, 7))

plt.imshow(
    predicted_probability,
    cmap="gray",
    vmin=0,
    vmax=1
)

plt.title(
    "U-Net Prediction Probability",
    fontsize=16
)

plt.colorbar(label="Probability")

plt.axis("off")

plt.tight_layout()
plt.show()


# =========================================================
# 13. DISPLAY U-NET BINARY MASK
# =========================================================

plt.figure(figsize=(7, 7))

plt.imshow(
    predicted_mask,
    cmap="gray"
)

plt.title(
    "U-Net Predicted Tumor Mask",
    fontsize=16
)

plt.axis("off")

plt.tight_layout()
plt.show()


print()
print("Comparison completed successfully.")