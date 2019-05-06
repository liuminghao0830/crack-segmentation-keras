import argparse, glob
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam 
import Models, LoadBatches
from Models.Segnet import *
from Models.Unet import *
import keras.backend as K

parser = argparse.ArgumentParser()
parser.add_argument("--train_images", type=str)
parser.add_argument("--train_annotations", type=str)
parser.add_argument("--img_height", type=int, default=224)
parser.add_argument("--img_width", type=int, default=224)

parser.add_argument("--val_images", type=str, default=None)
parser.add_argument("--val_annotations", type=str, default=None)

parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--load_weights", type=str, default=None)

parser.add_argument("--model", type=str, default='segnet_basic')


args = parser.parse_args()

train_images_path = args.train_images
train_segs_path = args.train_annotations
batch_size = args.batch_size

img_height = args.img_height
img_width = args.img_width

epochs = args.epochs
load_weights = args.load_weights

val_images_path = args.val_images
val_segs_path = args.val_annotations

num_train_images = len(glob.glob(train_images_path + '*.jpg'))
num_valid_images = len(glob.glob(val_images_path + '*.jpg'))

model_zoo = {'segnet': segnet, 'segnet_basic': segnet_basic, 
             'unet_mini': unet_mini, 'unet': unet}

print('Training on '+args.model)

m = model_zoo[args.model](input_shape = (img_height,img_width,3))
m.compile(loss='categorical_crossentropy',
                optimizer= Adam(lr=1e-3),
                metrics=['accuracy'])

if load_weights: m.load_weights(load_weights)


print("Model output shape: {}".format(m.output_shape))


train_gen = LoadBatches.imageSegmentationGenerator(train_images_path, 
                  train_segs_path, batch_size, img_height, img_width)

val_gen = LoadBatches.imageSegmentationGenerator(val_images_path, 
                val_segs_path, batch_size, img_height, img_width)


checkpoint = ModelCheckpoint(args.model+'.h5', monitor='val_loss', verbose=1, 
                    save_best_only=True, mode='min', save_weights_only=True)
reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5,  
                                   verbose=1, mode='auto', epsilon=0.0001)

m.fit_generator(train_gen,
                steps_per_epoch = num_train_images//batch_size,
                validation_data = val_gen,
                validation_steps = num_valid_images//batch_size,
                epochs = epochs, 
                verbose = 1, 
                callbacks = [checkpoint, reduceLROnPlat])
