import tensorflow as tf
from tensorflow.keras import layers, Model


def build_unet(input_shape=(240, 240, 4)):

    inputs = layers.Input(input_shape)

    # -------------------------
    # Encoder Block 1
    # -------------------------
    c1 = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    c1 = layers.Conv2D(32, 3, activation="relu", padding="same")(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    # -------------------------
    # Encoder Block 2
    # -------------------------
    c2 = layers.Conv2D(64, 3, activation="relu", padding="same")(p1)
    c2 = layers.Conv2D(64, 3, activation="relu", padding="same")(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # -------------------------
    # Encoder Block 3
    # -------------------------
    c3 = layers.Conv2D(128, 3, activation="relu", padding="same")(p2)
    c3 = layers.Conv2D(128, 3, activation="relu", padding="same")(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # -------------------------
    # Bottleneck
    # -------------------------
    c4 = layers.Conv2D(256, 3, activation="relu", padding="same")(p3)
    c4 = layers.Conv2D(256, 3, activation="relu", padding="same")(c4)

    # -------------------------
    # Decoder Block 1
    # -------------------------
    u5 = layers.UpSampling2D((2, 2))(c4)
    u5 = layers.Concatenate()([u5, c3])

    c5 = layers.Conv2D(128, 3, activation="relu", padding="same")(u5)
    c5 = layers.Conv2D(128, 3, activation="relu", padding="same")(c5)

    # -------------------------
    # Decoder Block 2
    # -------------------------
    u6 = layers.UpSampling2D((2, 2))(c5)
    u6 = layers.Concatenate()([u6, c2])

    c6 = layers.Conv2D(64, 3, activation="relu", padding="same")(u6)
    c6 = layers.Conv2D(64, 3, activation="relu", padding="same")(c6)

    # -------------------------
    # Decoder Block 3
    # -------------------------
    u7 = layers.UpSampling2D((2, 2))(c6)
    u7 = layers.Concatenate()([u7, c1])

    c7 = layers.Conv2D(32, 3, activation="relu", padding="same")(u7)
    c7 = layers.Conv2D(32, 3, activation="relu", padding="same")(c7)

    # -------------------------
    # Output
    # -------------------------
    outputs = layers.Conv2D(
        1,
        1,
        activation="sigmoid"
    )(c7)

    model = Model(inputs, outputs)

    return model


# Create U-Net
model = build_unet()

# Display model information
model.summary()