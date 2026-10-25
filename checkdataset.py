import os

dataset_path = "dataset"

classes = ["glioma", "meningioma", "notumor", "pituitary"]

for folder in ["Training", "Testing"]:

    print("\n" + folder)
    print("-" * 30)

    for cls in classes:

        path = os.path.join(dataset_path, folder, cls)

        if os.path.exists(path):

            images = os.listdir(path)

            print(cls, ":", len(images), "images")

        else:

            print(cls, ": Folder not found")