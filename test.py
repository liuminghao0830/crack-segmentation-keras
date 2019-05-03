import argparse
import Models, LoadBatches
import glob, os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import keras
from Models.Segnet import segnet
from Models.Segnet_basic import segnet_basic


parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type=str)
parser.add_argument("--test_images", type=str)
parser.add_argument("--output_path", type=str)
parser.add_argument("--img_height", type=int, default=224)
parser.add_argument("--img_width", type=int, default=224)
parser.add_argument("--vis", default=False)

parser.add_argument("--model", type=str, default='segnet_basic')

args = parser.parse_args()

images_path = args.test_images
img_width =  args.img_width
img_height = args.img_height

batch_size = 32

model_zoo = {'segnet': segnet, 'segnet_basic': segnet_basic}
print('Testing on '+args.model)

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
pred[pred > 0.5] = 1.
pred[pred < 0.5] = 0.

gt_seg = glob.glob(images_path + '*.png')
gt_seg.sort()

iou_score = 0.0
for idx in range(pred.shape[0]):
    gt = LoadBatches.getSegmentationArr(gt_seg[idx], img_width, img_height)[:,:,0]
    gt[gt > 0.5] = 1.
    gt[gt < 0.5] = 0.
    pred_label = pred[idx, :, :, 0]
    intersection = np.logical_and(gt, pred_label)
    union = np.logical_or(gt, pred_label)
    iou_score += np.sum(intersection,dtype=np.float) / np.sum(union,dtype=np.float)

print('mIOU:' + str(iou_score/pred.shape[0]))

if args.vis:
    output_path = args.model+'_predict/'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for imgName, gtName in zip(test_images, gt_seg):
        outName = imgName.replace(images_path, output_path)
        X = LoadBatches.getImageArr(imgName, img_width, img_height)
        pr = m.predict(X[np.newaxis,:,:,:])[0]
        gt = LoadBatches.getSegmentationArr(gtName, img_width, img_height)[:,:,0]
        fig=plt.figure(figsize=(12, 4))
        fig.add_subplot(1, 3, 1)
        plt.imshow(pr[:,:,0])
        fig.add_subplot(1, 3, 2)
        plt.imshow(X)
        fig.add_subplot(1, 3, 3)
        plt.imshow(gt)
        plt.savefig(outName)
