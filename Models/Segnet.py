import numpy as np
import keras
import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras import backend as K

K.set_image_data_format('channels_last')


def segnet(input_height=224, input_width=224):
    kernel = 3
    pool_size = 2
    input_shape = (input_width, input_height, 3)

    encoding_layers = [
        Conv2D(64, kernel_size=kernel, strides=1, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(64, kernel_size=kernel, strides=1, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=pool_size, strides=2),

        Conv2D(128, kernel_size=kernel, strides=1, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(128, kernel_size=kernel, strides=1, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=pool_size, strides=2),

        Conv2D(256, kernel_size=kernel, strides=1, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(256, kernel_size=kernel, strides=1, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(256, kernel_size=kernel, strides=1, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=pool_size, strides=2),

        Conv2D(512, kernel_size=kernel, strides=1, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(512, kernel_size=kernel, strides=1, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(512, kernel_size=kernel, strides=1, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=pool_size, strides=2),

        Conv2D(512, kernel_size=kernel, strides=1, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(512, kernel_size=kernel, strides=1, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(512, kernel_size=kernel, strides=1, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=pool_size, strides=2)
    ]

    decoding_layers = [
        UpSampling2D(size=(pool_size,pool_size)),
        Conv2D(512, kernel_size=kernel, strides=1, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(512, kernel_size=kernel, strides=1, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(512, kernel_size=kernel, strides=1, padding='same'),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(size=(pool_size,pool_size)),
        Conv2D(512, kernel_size=kernel, strides=1, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(512, kernel_size=kernel, strides=1, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(256, kernel_size=kernel, strides=1, padding='same'),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(size=(pool_size,pool_size)),
        Conv2D(256, kernel_size=kernel, strides=1, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(256, kernel_size=kernel, strides=1, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(128, kernel_size=kernel, strides=1, padding='same'),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(size=(pool_size,pool_size)),
        Conv2D(128, kernel_size=kernel, strides=1, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(64, kernel_size=kernel, strides=1, padding='same'),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(size=(pool_size,pool_size)),
        Conv2D(64, kernel_size=kernel, strides=1, padding='same'),
        BatchNormalization(),
        Activation('relu'),

        Conv2D(1, kernel_size=1, strides=1, padding='valid'),
        BatchNormalization(),
    ]


    model = Sequential()

    model.add(Layer(input_shape=input_shape))

    model.encoding_layers = encoding_layers
    for layer in model.encoding_layers:
        model.add(layer)


    model.decoding_layers = decoding_layers
    for layer in model.decoding_layers:
        model.add(layer)

    '''
    model.add(Convolution2D(nClasses, kernel_size=1, strides=1, padding='valid'))

    model.add(Reshape((nClasses, input_height*input_width), 
        input_shape=(nClasses, input_width, input_height)))
    
    model.add(Permute((2, 1)))
    model.add(Activation('softmax'))
    '''
    model.summary()
    return model

