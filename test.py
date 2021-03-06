import argparse
import Models, LoadBatches
import glob, os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import keras
from Models.Segnet import *
from Models.Unet import *


parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type=str)
parser.add_argument("--test_images", type=str)
parser.add_argument("--output_path", type=str)
parser.add_argument("--img_height", type=int, default=224)
parser.add_argument("--img_width", type=int, default=224)
parser.add_argument("--vis", type=bool, default=False)

parser.add_argument("--model", type=str, default='segnet_basic')

args = parser.parse_args()

images_path = args.test_images
img_width =  args.img_width
img_height = args.img_height

batch_size = 32

model_zoo = {'segnet': segnet, 'segnet_basic': segnet_basic, 
             'unet_mini': unet_mini, 'unet': unet}
print('Testing on '+args.model)


def output_test_image(x, pr, gt, path):
    fig=plt.figure(figsize=(12, 4))
    fig.add_subplot(1, 3, 1)
    plt.imshow(pr)
    fig.add_subplot(1, 3, 2)
    plt.imshow(x)
    fig.add_subplot(1, 3, 3)
    plt.imshow(gt)
    plt.savefig(path)
    plt.close()


m = model_zoo[args.model](input_shape=(img_height, img_width, 3))
m.compile(loss='mean_squared_error',
          optimizer= keras.optimizers.Adam(lr=1e-4),
          metrics=['accuracy'])

m.load_weights(args.save_weights_path)

test_images = glob.glob(images_path + '*.jpg')
test_images.sort()
test_gen = LoadBatches.imageSegmentationGenerator(images_path, 
               images_path, batch_size, img_height, img_width)

pred = m.predict_generator(test_gen, steps=len(test_images)//batch_size, verbose=1)
pred = np.argmax(pred, axis=-1)
print(pred.shape)

gt_seg = glob.glob(images_path + '*.png')
gt_seg.sort()

iou_score = 0.0
for idx in range(pred.shape[0]):
    gt = LoadBatches.getSegmentationArr(gt_seg[idx], img_width, img_height)
    gt = np.argmax(gt, axis=-1)
    pred_label = pred[idx, :, :]
    intersection = np.logical_and(gt, pred_label)
    union = np.logical_or(gt, pred_label)
    iou_score += np.sum(intersection,dtype=np.float) / np.sum(union,dtype=np.float)

print('mIOU:' + str(iou_score/pred.shape[0]))

if args.vis:
    output_path = args.model+'_predict/'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for i, imgName in enumerate(test_images):
        outName = imgName.replace(images_path, output_path)
        X = LoadBatches.getImageArr(imgName, img_width, img_height)
        pr = pred[i, :,:]
        gt = LoadBatches.getSegmentationArr(gt_seg[i], img_width, img_height)
        gt = np.argmax(gt, axis=-1)
        output_test_image(X, pr, gt, outName)
        
