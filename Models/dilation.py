from keras import backend as K
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.layers import Dropout, UpSampling2D, ZeroPadding2D

def dilatenet(input_shape):
    classes = 1
    model_in = Input(shape=input_shape)

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(model_in)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(x)
    
    x = Conv2D(512, (3, 3), dilation_rate=(2, 2), activation='relu', padding='same', name='conv5_1')(x)
    x = Conv2D(512, (3, 3), dilation_rate=(2, 2), activation='relu', padding='same', name='conv5_2')(x)
    x = Conv2D(512, (3, 3), dilation_rate=(2, 2), activation='relu', padding='same', name='conv5_3')(x)
    x = Conv2D(4096, (7, 7), dilation_rate=(4, 4), activation='relu', padding='same', name='fc6')(x)
    x = Dropout(0.5, name='drop6')(x)
    
    x = Conv2D(4096, (1, 1), activation='relu', name='fc7')(x)
    x = Dropout(0.5, name='drop7')(x)
    x = Conv2D(classes, (1, 1), name='final')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(classes, (3, 3), activation='relu', name='ctx_conv1_1')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(classes, (3, 3), activation='relu', name='ctx_conv1_2')(x)
    x = ZeroPadding2D(padding=(2, 2))(x)
    
    x = Conv2D(classes, (3, 3), dilation_rate=(2, 2), activation='relu', name='ctx_conv2_1')(x)
    x = ZeroPadding2D(padding=(4, 4))(x)
    x = Conv2D(classes, (3, 3), dilation_rate=(4, 4), activation='relu', name='ctx_conv3_1')(x)
    x = ZeroPadding2D(padding=(8, 8))(x)
    x = Conv2D(classes, (3, 3), dilation_rate=(8, 8), activation='relu', name='ctx_conv4_1')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    
    x = Conv2D(classes, (3, 3), activation='relu', name='ctx_fc1')(x)

    x = UpSampling2D(size=(8, 8))(x)
    model_out = Conv2D(classes, (1, 1), activation='sigmoid', name='ctx_final')(x)
    
    model = Model(input=model_in, output=model_out, name='dilation')
    model.summary()

    return model
