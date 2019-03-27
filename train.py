import argparse
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam 
import Models, LoadBatches
from Models.Segnet import segnet

parser = argparse.ArgumentParser()
parser.add_argument("--train_images", type = str)
parser.add_argument("--train_annotations", type = str)
parser.add_argument("--img_height", type=int, default = 224)
parser.add_argument("--img_width", type=int, default = 224)

parser.add_argument("--val_images", type = str, default = None)
parser.add_argument("--val_annotations", type = str, default = None)

parser.add_argument("--epochs", type = int, default = 50)
parser.add_argument("--batch_size", type = int, default = 16)
parser.add_argument("--load_weights", type = str, default = None)


args = parser.parse_args()

train_images_path = args.train_images
train_segs_path = args.train_annotations
batch_size = args.batch_size

img_height = args.img_height
img_width = args.img_width

epochs = args.epochs
load_weights = args.load_weights

if args.val_images:
    val_images_path = args.val_images
    val_segs_path = args.val_annotations

m = segnet(input_height=img_height, input_width=img_width)
m.compile(loss='mean_squared_error',
                optimizer= Adam(lr=1e-4),
                metrics=['accuracy'])

if load_weights:
    m.load_weights(load_weights)


print("Model output shape:", m.output_shape)

train_gen = LoadBatches.imageSegmentationGenerator(train_images_path, 
            train_segs_path, train_batch_size, img_height, img_width)

if args.val_images:
    val_gen = LoadBatches.imageSegmentationGenerator(val_images_path, 
                val_segs_path, val_batch_size, img_height, img_width)


checkpoint = ModelCheckpoint(model_name+'.h5', monitor='val_acc', verbose=1, 
                    save_best_only=True, mode='max', save_weights_only=True)
reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5,  
                                   verbose=1, mode='auto', epsilon=0.0001)

m.fit_generator(train_gen,
                steps_per_epoch = 1896//train_batch_size,
                validation_data = val_gen,
                validation_steps = 348//val_batch_size,
                epochs = epochs, 
                verbose = 1, 
                callbacks = [checkpoint, reduceLROnPlat])