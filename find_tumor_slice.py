import h5py
import glob
import numpy as np

folder = "brats_dataset/BraTS2020_training_data/content/data"

files = glob.glob(folder + "/volume_1_slice_*.h5")

print("Total slices found:", len(files))

for file_path in files:

    with h5py.File(file_path, "r") as f:

        image = f["image"][:]
        mask = f["mask"][:]

        mask_sum = np.sum(mask)

        if mask_sum > 0:

            print("\nTumor slice found!")
            print("File:", file_path)
            print("Image shape:", image.shape)
            print("Mask shape:", mask.shape)
            print("Mask sum:", mask_sum)

            print("Image minimum:", image.min())
            print("Image maximum:", image.max())

            break