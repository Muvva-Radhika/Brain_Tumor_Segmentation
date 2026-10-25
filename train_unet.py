import tensorflow as tf
from unet_model import build_unet
from unet_data_loader import BrainTumorDataset


# --------------------------------
# Load training file paths
# --------------------------------

with open("train_files.txt", "r") as f:
    train_files = [line.strip() for line in f.readlines()]


# --------------------------------
# Small test dataset
# --------------------------------

train_files = train_files[:1000]

print("Training files used:", len(train_files))


# --------------------------------
# Create data loader
# --------------------------------

train_dataset = BrainTumorDataset(
    train_files,
    batch_size=8
)


# --------------------------------
# Create U-Net
# --------------------------------

model = build_unet()


# --------------------------------
# Dice coefficient
# --------------------------------

def dice_coefficient(y_true, y_pred):

    y_pred = tf.cast(y_pred > 0.5, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred)

    return (
        (2.0 * intersection + 1.0)
        /
        (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1.0)
    )


# --------------------------------
# Compile model
# --------------------------------

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[dice_coefficient]
)


# --------------------------------
# Train
# --------------------------------

history = model.fit(
    train_dataset,
    epochs=2
)


# --------------------------------
# Save model
# --------------------------------

model.save("unet_brain_tumor.keras")

print("U-Net training completed.")
print("Model saved as unet_brain_tumor.keras")