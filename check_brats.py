import h5py

file_path = "brats_dataset/BraTS2020_training_data/content/data/volume_1_slice_0.h5"

with h5py.File(file_path, "r") as f:

    print("Keys inside file:")
    print(list(f.keys()))

    for key in f.keys():
        print(key, "shape:", f[key].shape)