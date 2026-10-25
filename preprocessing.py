import cv2
import numpy as np
import albumentations as A

# Image size required for the model
IMAGE_SIZE = 224

# Preprocessing and augmentation
transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2)
])


def preprocess_image(image_path):

    # Read image
    image = cv2.imread(image_path)

    if image is None:
        print("Image could not be read:", image_path)
        return None

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply preprocessing
    transformed = transform(image=image)

    image = transformed["image"]

    # Convert pixel values from 0-255 to 0-1
    image = image.astype(np.float32) / 255.0

    return image


# Test one image
image_path = "dataset/Training/glioma"

files = __import__("os").listdir(image_path)

if len(files) > 0:

    first_image = files[0]

    full_path = image_path + "/" + first_image

    processed_image = preprocess_image(full_path)

    if processed_image is not None:

        print("Original image:", first_image)
        print("Preprocessed image shape:", processed_image.shape)
        print("Minimum pixel value:", processed_image.min())
        print("Maximum pixel value:", processed_image.max())