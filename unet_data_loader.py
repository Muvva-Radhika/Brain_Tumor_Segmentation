import h5py
import numpy as np
import tensorflow as tf


IMAGE_CHANNELS = 4


class BrainTumorDataset(tf.keras.utils.Sequence):

    def __init__(self, file_list, batch_size=8):
        self.file_list = file_list
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.file_list) / self.batch_size))

    def __getitem__(self, index):

        batch_files = self.file_list[
            index * self.batch_size:
            (index + 1) * self.batch_size
        ]

        images = []
        masks = []

        for file_path in batch_files:

            with h5py.File(file_path, "r") as f:

                image = f["image"][:]
                mask = f["mask"][:]

            # Normalize image
            image = image.astype(np.float32)

            min_value = image.min()
            max_value = image.max()

            if max_value > min_value:
                image = (image - min_value) / (max_value - min_value)

            # Combine 3 mask channels
            mask = np.max(mask, axis=2)

            # Convert mask to binary
            mask = (mask > 0).astype(np.float32)

            # Add channel dimension
            mask = np.expand_dims(mask, axis=-1)

            images.append(image)
            masks.append(mask)

        return np.array(images), np.array(masks)


# -----------------------------
# Load training file paths
# -----------------------------

with open("train_files.txt", "r") as f:
    train_files = [line.strip() for line in f.readlines()]


# -----------------------------
# Create dataset
# -----------------------------

train_dataset = BrainTumorDataset(
    train_files,
    batch_size=8
)


# -----------------------------
# Test one batch
# -----------------------------

images, masks = train_dataset[0]

print("Images shape:", images.shape)
print("Masks shape:", masks.shape)

print("Image minimum:", images.min())
print("Image maximum:", images.max())

print("Mask minimum:", masks.min())
print("Mask maximum:", masks.max())