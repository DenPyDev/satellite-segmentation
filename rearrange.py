import os
import shutil


def dataset_rearranger(dataset_path, image_folder="img", mask_folder="mask", f_ext=".png", output_path="ds"):
    images_raw = []
    masks_raw = []

    for f in os.listdir(os.path.join(dataset_path, image_folder)):
        if f.endswith(f_ext):
            images_raw.append(f)

    for f in os.listdir(os.path.join(dataset_path, mask_folder)):
        if f.endswith(f_ext):
            masks_raw.append(f)

    # files present in both directories
    files = list(set(images_raw).intersection(masks_raw))

    all_amount = len(files)
    train_amount = int(all_amount * 0.7)
    val_amount = int(all_amount * 0.2)
    test_amount = int(all_amount - train_amount - val_amount)

    train_files = files[:train_amount]
    val_files = files[train_amount:train_amount + val_amount]
    test_files = files[train_amount + val_amount:]

    print("train_files", len(train_files))
    print("val_files", len(val_files))
    print("test_files", len(test_files))

    os.makedirs(os.path.join(output_path, "train", "img"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "train", "mask"), exist_ok=True)

    os.makedirs(os.path.join(output_path, "val", "img"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "val", "mask"), exist_ok=True)

    os.makedirs(os.path.join(output_path, "test", "img"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "test", "mask"), exist_ok=True)



    for f in train_files:
        src = os.path.join(dataset_path, image_folder, f)
        dst = os.path.join(output_path, "train", "img", f)
        shutil.move(src, dst)

        src = os.path.join(dataset_path, mask_folder, f)
        dst = os.path.join(output_path, "train", "mask", f)

        shutil.move(src, dst)

    for f in val_files:
        src = os.path.join(dataset_path, image_folder, f)
        dst = os.path.join(output_path, "val", "img", f)
        shutil.move(src, dst)
        src = os.path.join(dataset_path, mask_folder, f)
        dst = os.path.join(output_path, "val", "mask", f)

        shutil.move(src, dst)

    for f in test_files:
        src = os.path.join(dataset_path, image_folder, f)
        dst = os.path.join(output_path, "test", "img", f)
        shutil.move(src, dst)

        src = os.path.join(dataset_path, mask_folder, f)
        dst = os.path.join(output_path, "test", "mask", f)

        shutil.move(src, dst)

if __name__ == '__main__':
    dataset_path = "bw_dataset"
    dataset_rearranger(dataset_path)
