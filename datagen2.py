from google.colab import drive
drive.mount('/content/drive')

# !unzip /content/drive/MyDrive/set_data.zip -d /content/set_data
# !cp /content/drive/MyDrive/rot_api.py  /content/
#
# !rm -rf dataset
# !mkdir dataset
# !mkdir dataset/images_train
# !mkdir dataset/masks_train

import cv2, math
import numpy as np
from rot_api import crop_around_center, largest_rotated_rect, rotate_image

path = r"set_data"
out_path = r"dataset"
bw_samples_per_rotation = 20

max_iter = 5000

def get_random_crop(image, x, y, crop_height, crop_width):

    crop = image[y: y + crop_height, x: x + crop_width]
    return crop


def rotate_img(image, angle):
    image_height, image_width = image.shape[:2]
    image_rotated = rotate_image(image, angle)
    image_rotated_cropped = crop_around_center(
        image_rotated,
        *largest_rotated_rect(
            image_width,
            image_height,
            math.radians(angle)
        )
    )
    return image_rotated_cropped
# 1 "2", "3"
for num in ["1"]:
    pic_path = f"{path}/{num}_pic.jpg"
    mask_path = f"{path}/{num}_mask.png"

    pic_bmp = cv2.imread(pic_path)
    mask_bmp = cv2.imread(mask_path)
    print(pic_bmp.shape, mask_bmp.shape)

    for ang in range(0, 360, 5):
        print("iter")

        print(ang)
        pic_bmp_rot = rotate_img(pic_bmp, ang)
        mask_bmp_rot = rotate_img(mask_bmp, ang)

        crop_width = crop_height = 500

        max_x = pic_bmp_rot.shape[1] - crop_width
        max_y = pic_bmp_rot.shape[0] - crop_height

        blacks = 0
        whites = 0
        max_iter_c = 0

        while whites< bw_samples_per_rotation:
        # while blacks<bw_samples_per_rotation or whites< bw_samples_per_rotation:
            x = np.random.randint(0, max_x)
            y = np.random.randint(0, max_y)

            mask_bmp_rot_crop = get_random_crop(mask_bmp_rot, x, y, crop_width, crop_height)
            pic_bmp_rot_crop = get_random_crop(pic_bmp_rot, x, y, crop_width, crop_height)

            if np.count_nonzero(mask_bmp_rot_crop.flatten()) > 100:

                if whites < bw_samples_per_rotation:
                    _, mask_bmp_rot_crop = cv2.threshold(mask_bmp_rot_crop,127,255,cv2.THRESH_BINARY)
                    out_pic_path = f"{out_path}/images_train/{num}_{ang}_{x}_{y}.png"
                    out_mask_path = f"{out_path}/masks_train/{num}_{ang}_{x}_{y}.png"
                    cv2.imwrite(out_pic_path, pic_bmp_rot_crop)
                    cv2.imwrite(out_mask_path, mask_bmp_rot_crop)

                    whites += 1

            else:
                if blacks < bw_samples_per_rotation:
                    pass
                    # _, mask_bmp_rot_crop = cv2.threshold(mask_bmp_rot_crop,127,255,cv2.THRESH_BINARY)
                    # pic_bmp_rot_crop = get_random_crop(pic_bmp_rot, x, y, 256, 256)
                    # out_pic_path = f"{out_path}/images_train/{num}_{ang}_{x}_{y}.png"
                    # out_mask_path = f"{out_path}/masks_train/{num}_{ang}_{x}_{y}.png"
                    # cv2.imwrite(out_pic_path, pic_bmp_rot_crop)
                    # cv2.imwrite(out_mask_path, mask_bmp_rot_crop)

                    # blacks += 1
            print(num,ang, whites, blacks, max_iter_c)
            max_iter_c +=1
            if max_iter_c > max_iter:
              break