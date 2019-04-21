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

parser.add_argument("--model", type=str, default='segnet')

args = parser.parse_args()

images_path = args.test_images
img_width =  args.img_width
img_height = args.img_height

model_zoo = {'segnet': segnet, 'segnet_basic': segnet_basic}
print('Training on '+args.model)

m = model_zoo[args.model](input_shape=(img_height, img_width, 3))
m.compile(loss='mean_squared_error',
          optimizer= keras.optimizers.Adam(lr=1e-4),
          metrics=['accuracy'])

m.load_weights(args.save_weights_path)

images = glob.glob(images_path + '*.jpg')
images.sort()

output_path = args.model+'_predict/'
if not os.path.exists(output_path):
    os.mkdir(output_path)

images = np.random.choice(images, 100, replace=False)

for imgName in images:
    outName = imgName.replace(images_path, output_path)
    X = LoadBatches.getImageArr(imgName, img_width, img_height)
    pr = m.predict(X[np.newaxis,:,:,:])[0]
    fig=plt.figure(figsize=(8, 4))
    fig.add_subplot(1, 2, 1)
    plt.imshow(pr[:,:,0])
    fig.add_subplot(1, 2, 2)
    plt.imshow(X)
    plt.savefig(outName)
