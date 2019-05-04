from keras.models import Model
from keras.layers import *
from keras import backend as K

K.set_image_data_format('channels_last')

def unet(input_shape):
    inputs = Input(shape=input_shape)
    conv1 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    conv3 = UpSampling2D(size=(2, 2))(conv3)

    up1 = Concatenate(axis=-1)([conv3, conv2])
    
    conv4 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(conv4)
    conv4 = UpSampling2D(size=(2, 2))(conv4)

    up2 = Concatenate(axis=-1)([conv4, conv1])
    conv5 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(conv5)
    
    conv6 = Conv2D(2, 1, 1, activation='softmax',border_mode='same')(conv5)

    model = Model(inputs, conv6)
    model.summary()

    return model
	
	
	

