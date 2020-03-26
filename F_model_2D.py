
from keras.applications import vgg16
from keras.models import Model, Sequential
from keras.layers import Conv2D, Conv2DTranspose, Input, Cropping2D, add, Dropout, Reshape, Activation
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import os
import sys
from functools import partial
from keras.models import Model
from keras.regularizers import l2
from keras.layers import *
from keras.engine import Layer
from keras.applications.vgg16 import *
from keras.models import *
from keras.applications.imagenet_utils import _obtain_input_shape
import keras.backend as K
import tensorflow as tf
from keras import optimizers

nClasses = 2
sgd = optimizers.SGD(lr=1E-2, decay=5**(-4), momentum=0.9, nesterov=True)

# compute dsc
def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# proposed loss function
def dice_coef_loss(y_true, y_pred):
    distance = 0
    for label_index in range(1,nClasses):
        dice_coef_class = dice_coef(y_true[:,:,:,label_index], y_pred[:, :,:,label_index])
        distance = 1 - dice_coef_class + distance
    return distance

# dsc per class
def label_wise_dice_coefficient(y_true, y_pred, label_index):
    return dice_coef(y_true[:,:,:,label_index], y_pred[:, :,:,label_index])

# get label dsc
def get_label_dice_coefficient_function(label_index):
    f = partial(label_wise_dice_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'label_{0}_dice_coef'.format(label_index))
    return f


def focal_loss(gamma=2., alpha=.25):
	def focal_loss_fixed(y_true, y_pred):
		y_pred = K.clip(y_pred, 0.0001, 1-0.0001)
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
	return focal_loss_fixed
    
def FCN8(nClasses, input_height=192, input_width=192):
    ## input_height and width must be devisible by 32 because maxpooling with filter size = (2,2) is operated 5 times,
    ## which makes the input_height and width 2^5 = 32 times smaller
    img_input = Input(shape=(input_height, input_width,
                             3))  ## Assume 224,224,3
    IMAGE_ORDERING = "channels_last"
    ## Block 1
    x = Conv2D(64, (3, 3),
               activation='relu',
               padding='same',
               name='block1_conv1',
               data_format=IMAGE_ORDERING)(img_input)
    x = Conv2D(64, (3, 3),
               activation='relu',
               padding='same',
               name='block1_conv2',
               data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2),
                     strides=(2, 2),
                     name='block1_pool',
                     data_format=IMAGE_ORDERING)(x)
    f1 = x

    # Block 2
    x = Conv2D(128, (3, 3),
               activation='relu',
               padding='same',
               name='block2_conv1',
               data_format=IMAGE_ORDERING)(x)
    x = Conv2D(128, (3, 3),
               activation='relu',
               padding='same',
               name='block2_conv2',
               data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2),
                     strides=(2, 2),
                     name='block2_pool',
                     data_format=IMAGE_ORDERING)(x)
    f2 = x

    # Block 3
    x = Conv2D(256, (3, 3),
               activation='relu',
               padding='same',
               name='block3_conv1',
               data_format=IMAGE_ORDERING)(x)
    x = Conv2D(256, (3, 3),
               activation='relu',
               padding='same',
               name='block3_conv2',
               data_format=IMAGE_ORDERING)(x)
    x = Conv2D(256, (3, 3),
               activation='relu',
               padding='same',
               name='block3_conv3',
               data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2),
                     strides=(2, 2),
                     name='block3_pool',
                     data_format=IMAGE_ORDERING)(x)
    pool3 = x

    # Block 4
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block4_conv1',
               data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block4_conv2',
               data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block4_conv3',
               data_format=IMAGE_ORDERING)(x)
    pool4 = MaxPooling2D((2, 2),
                         strides=(2, 2),
                         name='block4_pool',
                         data_format=IMAGE_ORDERING)(x)  ## (None, 14, 14, 512)

    # Block 5
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block5_conv1',
               data_format=IMAGE_ORDERING)(pool4)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block5_conv2',
               data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block5_conv3',
               data_format=IMAGE_ORDERING)(x)
    pool5 = MaxPooling2D((2, 2),
                         strides=(2, 2),
                         name='block5_pool',
                         data_format=IMAGE_ORDERING)(x)  ## (None, 7, 7, 512)

    n = 4096
    o = (Conv2D(n, (7, 7),
                activation='relu',
                padding='same',
                name="conv6",
                data_format=IMAGE_ORDERING))(pool5)
    conv7 = (Conv2D(n, (1, 1),
                    activation='relu',
                    padding='same',
                    name="conv7",
                    data_format=IMAGE_ORDERING))(o)

    ## 4 times upsamping for pool4 layer
    conv7_4 = Conv2DTranspose(nClasses,
                              kernel_size=(4, 4),
                              strides=(4, 4),
                              use_bias=False,
                              data_format=IMAGE_ORDERING)(conv7)
    ## (None, 224, 224, 10)
    ## 2 times upsampling for pool411
    pool411 = (Conv2D(nClasses, (1, 1),
                      activation='relu',
                      padding='same',
                      name="pool4_11",
                      data_format=IMAGE_ORDERING))(pool4)
    pool411_2 = (Conv2DTranspose(nClasses,
                                 kernel_size=(2, 2),
                                 strides=(2, 2),
                                 use_bias=False,
                                 data_format=IMAGE_ORDERING))(pool411)

    pool311 = (Conv2D(nClasses, (1, 1),
                      activation='relu',
                      padding='same',
                      name="pool3_11",
                      data_format=IMAGE_ORDERING))(pool3)

    o = Add(name="add")([pool411_2, pool311, conv7_4])
    o = Conv2DTranspose(nClasses,
                        kernel_size=(8, 8),
                        strides=(8, 8),
                        use_bias=False,
                        data_format=IMAGE_ORDERING)(o)

    o = (Activation('softmax'))(o)

    fine_model = Model(img_input, o)

    metrics = dice_coef
    include_label_wise_dice_coefficients = True;
    if not isinstance(metrics, list):
        metrics = [metrics]

    if include_label_wise_dice_coefficients and nClasses > 1:
        label_wise_dice_metrics = [get_label_dice_coefficient_function(index) for index in range(nClasses)]
        if metrics:
            metrics = metrics + label_wise_dice_metrics
        else:
            metrics = label_wise_dice_metrics
            

    fine_model.compile(optimizer=sgd,

                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    
    
    
    #fine_model.load_weights('/data/ydeng1/pancreatitis/fcn/fine_weights.h5')
#    fine_model.compile(optimizer=sgd,

#                  loss=[focal_loss(alpha=.25, gamma=2)],
#                  metrics=['accuracy'])

    return fine_model


fine_model = FCN8(nClasses=2, input_height=192, input_width=192)
fine_model.summary()
