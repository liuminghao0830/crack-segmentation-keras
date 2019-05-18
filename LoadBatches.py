import numpy as np
import cv2
import glob
import itertools

def getImageArr(path, width, height, imgNorm='divide', 
							 odering='channels_last'):
    img = cv2.imread(path, 1)

    if imgNorm == 'sub_and_divide':
        img = np.float32(cv2.resize(img, (width, height))) / 127.5 - 1
    elif imgNorm == 'sub_mean':
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        img[:,:,0] -= 103.939
        img[:,:,1] -= 116.779
        img[:,:,2] -= 123.68
    elif imgNorm == 'divide':
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        img = img/255.0

    if odering == 'channels_first':
        img = np.rollaxis(img, 2, 0)
    return img



def getSegmentationArr(path, width, height):

    # seg_labels = np.zeros((height, width, 2))

    img = cv2.imread(path, 0)
    img = cv2.resize(img, (width, height))
    seg_labels = img / 255.0
    
    #seg_labels = np.reshape(seg_labels, (width*height, 1))
    
    return seg_labels[:,:,np.newaxis]


def imageSegmentationGenerator(images_path, segs_path, batch_size, 
                                           img_height, img_width):
    
    assert images_path[-1] == '/'
    assert segs_path[-1] == '/'

    images = glob.glob(images_path + '*.jpg')
    images.sort()
    segmentations = glob.glob(segs_path + '*.png')
    segmentations.sort()

    assert len(images) == len(segmentations)
    for im , seg in zip(images,segmentations):
        assert(im.split('/')[-1].split('.')[0] == seg.split('/')[-1].split('.')[0])

    zipped = itertools.cycle(zip(images,segmentations))

    while True:
        X = []
        Y = []
        for _ in range(batch_size):
            im , seg = next(zipped)
            X.append(getImageArr(im, img_width, img_height))
            Y.append(getSegmentationArr(seg, img_width, img_height))

        yield np.array(X) , np.array(Y)


