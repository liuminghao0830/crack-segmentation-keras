import argparse
import Models , LoadBatches
from Models.Segnet import segnet
import glob
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type = str)
parser.add_argument("--test_images", type = str)
parser.add_argument("--output_path", type = str)
parser.add_argument("--img_height", type=int, default = 224)
parser.add_argument("--img_width", type=int, default = 224)

args = parser.parse_args()

images_path = args.test_images
img_width =  args.img_width
img_height = args.img_height

m = segnet(input_height=img_height, input_width=img_width)
m.load_weights(args.save_weights_path)
m.compile(loss='categorical_crossentropy',
      				optimizer= 'adam',
      				metrics=['accuracy'])

images = glob.glob(images_path + '*.jpg')
images.sort()

images = np.random.choice(images, 10, replace=False)

for imgName in images:
    #outName = imgName.replace(images_path, args.output_path)
    X = LoadBatches.getImageArr(imgName, img_width, img_height)
    pr = m.predict(X[np.newaxis,:,:,:])[0]
    fig=plt.figure(figsize=(8, 4))
    fig.add_subplot(1, 2, 1)
    plt.imshow(pr[:,:,0])
    fig.add_subplot(1, 2, 2)
    plt.imshow(X)
    plt.show()
    #cv2.imwrite(outName, seg_img)

