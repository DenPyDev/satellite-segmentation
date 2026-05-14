import os
import glob
import random
import cv2
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import kagglehub
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from unet_sat import unet, trainGenerator, ToMask

CROP_SIZE       = 256
CROPS_PER_IMG   = 200
MIN_BUILDING_PX = 500
VAL_RATIO       = 0.2
N_IMAGES        = 5
BATCH_SIZE      = 16
EPOCHS          = 10
RANDOM_SEED     = 42
TRAIN_PATH      = 'inria_crops/train'
VAL_PATH        = 'inria_crops/val'
MODEL_SAVE      = 'segmentation_inria.h5'
RESULTS_DIR     = 'results'

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def find_inria_paths(dataset_path, n=N_IMAGES):
    img_paths = sorted(glob.glob(os.path.join(dataset_path, '**/train/images/*.tif'), recursive=True))
    gt_paths  = sorted(glob.glob(os.path.join(dataset_path, '**/train/gt/*.tif'),     recursive=True))
    if not img_paths:
        img_paths = sorted(glob.glob(os.path.join(dataset_path, '**/images/*.tif'), recursive=True))
        gt_paths  = sorted(glob.glob(os.path.join(dataset_path, '**/gt/*.tif'),     recursive=True))
    return img_paths[:n], gt_paths[:n]


def generate_crops(img_paths, gt_paths):
    for split in (TRAIN_PATH, VAL_PATH):
        os.makedirs(os.path.join(split, 'img'),  exist_ok=True)
        os.makedirs(os.path.join(split, 'mask'), exist_ok=True)

    all_crops = []
    for img_p, gt_p in zip(img_paths, gt_paths):
        img_full  = tiff.imread(img_p)
        mask_full = tiff.imread(gt_p)

        if img_full.ndim == 2:
            img_full = np.stack([img_full] * 3, axis=-1)
        if img_full.shape[2] > 3:
            img_full = img_full[:, :, :3]
        if mask_full.ndim == 3:
            mask_full = mask_full[:, :, 0]

        h, w = img_full.shape[:2]
        max_x, max_y = w - CROP_SIZE, h - CROP_SIZE
        if max_x <= 0 or max_y <= 0:
            print(f"Image too small, skipping: {img_p}")
            continue

        crops_got, attempts = 0, 0
        while crops_got < CROPS_PER_IMG and attempts < CROPS_PER_IMG * 20:
            attempts += 1
            x = np.random.randint(0, max_x)
            y = np.random.randint(0, max_y)
            img_crop  = img_full[y:y+CROP_SIZE, x:x+CROP_SIZE]
            mask_crop = mask_full[y:y+CROP_SIZE, x:x+CROP_SIZE]
            if np.count_nonzero(mask_crop) < MIN_BUILDING_PX and crops_got <= CROPS_PER_IMG // 2:
                continue
            _, mask_bin = cv2.threshold(mask_crop.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
            all_crops.append((img_crop, mask_bin))
            crops_got += 1
        print(f"{os.path.basename(img_p)}: {crops_got} crops")

    random.shuffle(all_crops)
    val_count   = int(len(all_crops) * VAL_RATIO)
    val_crops   = all_crops[:val_count]
    train_crops = all_crops[val_count:]

    for i, (img_c, mask_c) in enumerate(train_crops):
        cv2.imwrite(os.path.join(TRAIN_PATH, 'img',  f'{i:05d}.png'), cv2.cvtColor(img_c, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(TRAIN_PATH, 'mask', f'{i:05d}.png'), mask_c)
    for i, (img_c, mask_c) in enumerate(val_crops):
        cv2.imwrite(os.path.join(VAL_PATH, 'img',  f'{i:05d}.png'), cv2.cvtColor(img_c, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(VAL_PATH, 'mask', f'{i:05d}.png'), mask_c)

    print(f"train: {len(train_crops)}, val: {len(val_crops)}")
    return len(train_crops), len(val_crops)


def load_val_dataset(batch_size=128, target_size=(256, 256)):
    n_val = len(os.listdir(os.path.join(VAL_PATH, 'img')))
    bs = min(batch_size, n_val)
    image_gen = ImageDataGenerator().flow_from_directory(
        VAL_PATH, classes=['img'], class_mode=None,
        color_mode='rgb', target_size=target_size, batch_size=bs, seed=2, shuffle=False)
    mask_gen = ImageDataGenerator().flow_from_directory(
        VAL_PATH, classes=['mask'], class_mode=None,
        color_mode='grayscale', target_size=target_size, batch_size=bs, seed=2, shuffle=False)
    imgs, masks = next(zip(image_gen, mask_gen))
    return imgs / 255.0, (masks / 255.0).clip(0, 1)


def lr_scheduler(epoch, lr):
    if epoch % 2 == 0 and epoch:
        return lr * 0.9
    return lr


def save_preview_crops(n=4):
    img_names  = sorted(os.listdir(os.path.join(TRAIN_PATH, 'img')))[:n]
    mask_names = sorted(os.listdir(os.path.join(TRAIN_PATH, 'mask')))[:n]
    fig, axs = plt.subplots(2, n, figsize=(4 * n, 8))
    for col, (fn_i, fn_m) in enumerate(zip(img_names, mask_names)):
        img_c  = cv2.imread(os.path.join(TRAIN_PATH, 'img',  fn_i))
        mask_c = cv2.imread(os.path.join(TRAIN_PATH, 'mask', fn_m), cv2.IMREAD_GRAYSCALE)
        axs[0, col].imshow(cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB))
        axs[0, col].set_title('image')
        axs[0, col].axis('off')
        axs[1, col].imshow(mask_c, cmap='gray')
        axs[1, col].set_title('mask')
        axs[1, col].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'preview_crops.png'), dpi=100)
    plt.close()


def save_training_curves(history):
    d = history.history
    fig, axs = plt.subplots(2, sharex=True, figsize=(15, 10))
    axs[0].plot(d['loss'],     color='r', label='train loss')
    axs[0].plot(d['val_loss'], color='b', label='val loss')
    axs[0].legend(loc='upper right')
    axs[0].set_title('loss')
    auc_key = [k for k in d if 'auc' in k and 'val' not in k]
    val_auc_key = [k for k in d if 'val_auc' in k]
    if auc_key and val_auc_key:
        axs[1].plot(d[auc_key[0]],     color='r', label='train auc')
        axs[1].plot(d[val_auc_key[0]], color='b', label='val auc')
        axs[1].legend(loc='upper left')
        axs[1].set_title('auc')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'training_curves.png'), dpi=100)
    plt.close()


def save_predictions(model, val_x, val_y, n=16):
    results = model.predict(val_x[:n])
    for i in range(min(n, len(val_x))):
        fig, axs = plt.subplots(1, 4, figsize=(16, 4))
        axs[0].imshow(val_x[i])
        axs[0].set_title('image')
        axs[0].axis('off')
        axs[1].imshow(val_y[i].reshape(CROP_SIZE, CROP_SIZE), cmap='gray')
        axs[1].set_title('gt mask')
        axs[1].axis('off')
        axs[2].imshow(results[i].reshape(CROP_SIZE, CROP_SIZE), cmap='gray')
        axs[2].set_title('pred raw')
        axs[2].axis('off')
        axs[3].imshow(ToMask(results[i]), cmap='gray')
        axs[3].set_title('pred binary')
        axs[3].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'pred_{i:02d}.png'), dpi=100)
        plt.close()


if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)

    dataset_path = kagglehub.dataset_download('sagar100rathod/inria-aerial-image-labeling-dataset')
    print("Dataset root:", dataset_path)

    img_paths, gt_paths = find_inria_paths(dataset_path)
    print(f"Found {len(img_paths)} images, {len(gt_paths)} masks")
    if not img_paths:
        raise RuntimeError("No .tif images found in dataset.")

    train_img_dir = os.path.join(TRAIN_PATH, 'img')
    if not os.path.exists(train_img_dir) or not os.listdir(train_img_dir):
        n_train, n_val = generate_crops(img_paths, gt_paths)
    else:
        n_train = len(os.listdir(train_img_dir))
        n_val   = len(os.listdir(os.path.join(VAL_PATH, 'img')))
        print(f"Using existing crops: train={n_train}, val={n_val}")

    save_preview_crops()

    model = unet()
    model.summary()

    steps_per_epoch = max(1, n_train // BATCH_SIZE)
    my_gene = trainGenerator(BATCH_SIZE, TRAIN_PATH, 'img', 'mask', save_to_dir=None)
    val_x, val_y = load_val_dataset()

    callbacks = [
        ModelCheckpoint(MODEL_SAVE, monitor='val_auc', mode='max', save_best_only=True, verbose=1),
        LearningRateScheduler(lr_scheduler, verbose=1),
    ]

    history = model.fit(
        my_gene,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=(val_x, val_y),
        callbacks=callbacks,
    )

    save_training_curves(history)

    model.load_weights(MODEL_SAVE)
    save_predictions(model, val_x, val_y)
