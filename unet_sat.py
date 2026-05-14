import os
from collections import defaultdict
import skimage.io as io
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
import cv2
from shapely.geometry import MultiPolygon, Polygon

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.metrics import AUC
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def unet(pretrained_weights=None, input_size=(256, 256, 3)):
    inputs = Input(input_size)
    kern_c = 8

    conv1 = Conv2D(kern_c, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(kern_c, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    kern_c *= 2
    conv2 = Conv2D(kern_c, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(kern_c, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    kern_c *= 2
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(kern_c, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(kern_c, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    kern_c *= 2
    conv4 = Conv2D(kern_c, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(kern_c, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    up6 = Conv2D(kern_c, 2, activation='relu', padding='same', kernel_initializer='he_normal')( UpSampling2D(size=(2, 2))(pool4))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(kern_c, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(kern_c, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    kern_c //= 2
    up7 = Conv2D(kern_c, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(kern_c, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(kern_c, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    kern_c //= 2
    up8 = Conv2D(kern_c, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(kern_c, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(kern_c, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    kern_c //= 2
    up9 = Conv2D(kern_c, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(kern_c, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(kern_c, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=[AUC()])

    if pretrained_weights:
        model.load_weights(pretrained_weights)
    return model


def adjustData(img, mask):
  img = img / 255.0
  mask[mask>0] = 1
  mask[mask<1] = 0
  return (img, mask)


def trainGenerator(batch_size,train_path,image_folder,mask_folder,image_color_mode = "rgb",
                  mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                  save_to_dir = None,target_size = (256,256),seed = 1):
  aug_dict = dict(rotation_range=0.2,
                  width_shift_range=0.05,
                  height_shift_range=0.05,
                  horizontal_flip=True,
                  fill_mode='nearest')
  image_datagen = ImageDataGenerator(**aug_dict)
  mask_datagen = ImageDataGenerator(**aug_dict)
  image_generator = image_datagen.flow_from_directory(
      train_path,
      classes = [image_folder],
      class_mode = None,
      color_mode = image_color_mode,
      target_size = target_size,
      batch_size = batch_size,
      save_to_dir = save_to_dir,
      save_prefix  = image_save_prefix,
      seed = seed)
  
  mask_generator = mask_datagen.flow_from_directory(
      train_path,
      classes = [mask_folder],
      class_mode = None,
      color_mode = mask_color_mode,
      target_size = target_size,
      batch_size = batch_size,
      save_to_dir = save_to_dir,
      save_prefix  = mask_save_prefix,
      seed = seed)
  train_generator = zip(image_generator, mask_generator)
  for (img,mask) in train_generator:
      img,mask = adjustData(img,mask)
      yield (img,mask)


def testGenerator(test_path,num_image = 30, target_size = (256,256)):
    image_datagen = ImageDataGenerator()
    image_generator = image_datagen.flow_from_directory(
        test_path,
        classes = ['img'],
        class_mode = None,
        color_mode = 'rgb',
        target_size = target_size,
        batch_size = num_image,
        save_to_dir = None,
        save_prefix  = '',
        seed = 2)
    for img in image_generator:
        img /= 255.0
        yield img


def GenTrainDataset(generator):
  xTrain = []
  yTrain = []
  tupleImg = next(generator)
  
  for img in tupleImg[0]:
    xTrain.append(img.reshape((256,256,3)))
  for mask in tupleImg[1]:
    yTrain.append(mask.reshape((256,256,1)))
  
  return np.array(xTrain), np.array(yTrain)

def ToMask(img1, threshold=0.4):
    im = img1.reshape((256, 256)).copy()
    im[im > threshold] = 1
    im[im <= threshold] = 0
    return im


def test_pair(ds_path, i=0, target_size=(256, 256)):
    im_path = ds_path + "/img/"
    mask_path = ds_path + "/mask/"
    names_img = os.listdir(im_path)
    p_img = resize(io.imread(im_path + names_img[i], as_gray=False), target_size)
    p_mask = resize(io.imread(mask_path + names_img[i], as_gray=True), target_size)
    return p_img, p_mask


def mask_to_polygons(mask, epsilon=10., min_area=10.):
    temp = ((mask == 1) * 255).astype(np.uint8)
    contours, hierarchy = cv2.findContours(temp, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    if not contours:
        return MultiPolygon()
    approx_contours = [cv2.approxPolyDP(cnt, epsilon, True) for cnt in contours]
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(approx_contours[idx])
    all_polygons = []
    for idx, cnt in enumerate(approx_contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            poly = Polygon(
                shell=cnt[:, 0, :],
                holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                       if cv2.contourArea(c) >= min_area])
            all_polygons.append(poly)
    all_polygons = MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        if all_polygons.type == 'Polygon':
            all_polygons = MultiPolygon([all_polygons])
    return all_polygons


def mask_for_polygons(polygons, im_size):
    img_mask = np.zeros(im_size, np.uint8)
    if not polygons:
        return img_mask
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [int_coords(pi.coords) for poly in polygons for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask


if __name__ == '__main__':
  myGene = trainGenerator(32,'bw_dataset','img','mask',save_to_dir = None)
  model = unet()
  model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss',verbose=1, save_best_only=True)
  history = model.fit(myGene,steps_per_epoch=100,epochs=15,callbacks=[model_checkpoint])

  generator = trainGenerator(128,'ds_val','img','mask',save_to_dir=None, seed=6)
  dataset = GenTrainDataset(generator)
  model.evaluate(dataset[0], dataset[1])

  keys = list(history.history.keys())
  print(keys)
  plt.figure(figsize=(10, 5))
  plt.plot(history.history[keys[0]], color='r')
  plt.plot(history.history[keys[1]], color='b')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.show()

  model.save_weights('weights.h5')

