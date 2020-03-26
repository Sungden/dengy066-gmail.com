
from __future__ import print_function

# import packages
from functools import partial
import os
import numpy as np
from keras.models import Model 
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose,Dropout
from keras.optimizers import Adam
from keras import callbacks
from keras import backend as K
from keras.utils import plot_model
import h5py
from keras import optimizers
from keras.models import load_model
K.set_image_data_format('channels_last')  # TF dimension ordering in this code


# init configs
image_rows = 192
image_cols = 192
num_classes = 2

# patch extraction parameters
patch_size = 192
BASE = 64
smooth = 0.1
nb_epochs  =1000#configs.NUM_EPOCHS
batch_size  = 64
unet_model_type = 'default'
PATIENCE = 50

# compute dsc
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# proposed loss function
def dice_coef_loss(y_true, y_pred):
    distance = 0
    for label_index in range(num_classes):
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



# 2D U-net depth=5
def get_unet_default():
    metrics = dice_coef
    include_label_wise_dice_coefficients = True;
    inputs = Input((patch_size, patch_size,1))
    block1_conv1 = Conv2D(BASE, (3, 3), activation='relu', padding='same')(inputs)
    block1_conv2 = Conv2D(BASE, (3, 3), activation='relu', padding='same')(block1_conv1)
    block1_pool = MaxPooling2D(pool_size=(2, 2))(block1_conv2)

    
    block2_conv1 = Conv2D(BASE*2, (3, 3), activation='relu', padding='same')(block1_pool)
    block2_conv2 = Conv2D(BASE*2, (3, 3), activation='relu', padding='same')(block2_conv1)
    block2_pool = MaxPooling2D(pool_size=(2, 2))(block2_conv2)

    block3_conv1 = Conv2D(BASE*4, (3, 3), activation='relu', padding='same')(block2_pool)
    block3_conv2 = Conv2D(BASE*4, (3, 3), activation='relu', padding='same')(block3_conv1)
    block3_conv3 = Conv2D(BASE*4, (3, 3), activation='relu', padding='same')(block3_conv2)
    block3_pool = MaxPooling2D(pool_size=(2, 2))(block3_conv3)

    block4_conv1 = Conv2D(BASE*8, (3, 3), activation='relu', padding='same')(block3_pool)
    block4_conv2 = Conv2D(BASE*8, (3, 3), activation='relu', padding='same')(block4_conv1)
    block4_conv3 = Conv2D(BASE*8, (3, 3), activation='relu', padding='same')(block4_conv2)
    block4_pool = MaxPooling2D(pool_size=(2, 2))(block4_conv3)

    block5_conv1 = Conv2D(BASE*16, (3, 3), activation='relu', padding='same')(block4_pool)
    block5_conv2 = Conv2D(BASE*16, (3, 3), activation='relu', padding='same')(block5_conv1)
    block5_conv3 = Conv2D(BASE*16, (3, 3), activation='relu', padding='same')(block5_conv2)


    up6 = concatenate([Conv2DTranspose(BASE*8, (2, 2), strides=(2, 2), padding='same')(block5_conv3), block4_conv3], axis=3)
    conv6 = Conv2D(BASE*8, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(BASE*8, (3, 3), activation='relu', padding='same')(conv6)


    up7 = concatenate([Conv2DTranspose(BASE*4, (2, 2),strides=(2, 2), padding='same')(conv6), block3_conv3], axis=3)
    conv7 = Conv2D(BASE*4, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(BASE*4, (3, 3), activation='relu', padding='same')(conv7)


    up8 = concatenate([Conv2DTranspose(BASE*2, (2, 2), strides=(2, 2), padding='same')(conv7), block2_conv2], axis=3)
    conv8 = Conv2D(BASE*2, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(BASE*2, (3, 3), activation='relu', padding='same')(conv8)


    up9 = concatenate([Conv2DTranspose(BASE, (2, 2), strides=(2, 2), padding='same')(conv8), block1_conv2], axis=3)
    conv9 = Conv2D(BASE, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(BASE, (3, 3), activation='relu', padding='same')(conv9)


    conv10 = Conv2D(num_classes, (1, 1), activation='softmax')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    
    if not isinstance(metrics, list):
        metrics = [metrics]

    if include_label_wise_dice_coefficients and num_classes > 1:
        label_wise_dice_metrics = [get_label_dice_coefficient_function(index) for index in range(num_classes)]
        if metrics:
            metrics = metrics + label_wise_dice_metrics
        else:
            metrics = label_wise_dice_metrics
     
    #model = load_model("/data/ydeng1/pancreatitis/fcn_3Dunet/F_outputs_2D/model_Unet.h5")
    
    sgd = optimizers.SGD(lr=1E-2, decay=5**(-4), momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,

                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# 2D U-net depth=4
def get_unet_reduced():
    metrics = dice_coef
    include_label_wise_dice_coefficients = True;
    inputs = Input((patch_size, patch_size, 1))
    conv1 = Conv2D(BASE, (3, 3), activation='ReLU', padding='same')(inputs)
    conv1 = Conv2D(BASE, (3, 3), activation='ReLU', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(BASE*2, (3, 3), activation='ReLU', padding='same')(pool1)
    conv2 = Conv2D(BASE*2, (3, 3), activation='ReLU', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(BASE*4, (3, 3), activation='ReLU', padding='same')(pool2)
    conv3 = Conv2D(BASE*4, (3, 3), activation='ReLU', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(BASE*8, (3, 3), activation='ReLU', padding='same')(pool3)
    conv4 = Conv2D(BASE*8, (3, 3), activation='ReLU', padding='same')(conv4)


    up7 = concatenate([Conv2DTranspose(BASE*4, (2, 2),strides=(2, 2), padding='same')(conv4), conv3], axis=3)
    conv7 = Conv2D(BASE*4, (3, 3), activation='ReLU', padding='same')(up7)
    conv7 = Conv2D(BASE*4, (3, 3), activation='ReLU' ,padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(BASE*2, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(BASE*2, (3, 3), activation='ReLU', padding='same')(up8)
    conv8 = Conv2D(BASE*2, (3, 3), activation='ReLU', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(BASE, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(BASE, (3, 3), activation='ReLU', padding='same')(up9)
    conv9 = Conv2D(BASE, (3, 3), activation='ReLU', padding='same')(conv9)

    conv10 = Conv2D(num_classes, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    #model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    
    if not isinstance(metrics, list):
        metrics = [metrics]

    if include_label_wise_dice_coefficients and num_classes > 1:
        label_wise_dice_metrics = [get_label_dice_coefficient_function(index) for index in range(num_classes)]
        if metrics:
            metrics = metrics + label_wise_dice_metrics
        else:
            metrics = label_wise_dice_metrics

    model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, metrics=metrics)
    return model

# 2D U-net depth=6
def get_unet_extended():
    metrics = dice_coef
    include_label_wise_dice_coefficients = True;
    inputs = Input((patch_size, patch_size, 1))
    conv1 = Conv2D(BASE, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(BASE, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(BASE*2, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(BASE*2, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(BASE*4, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(BASE*4, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(BASE*8, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(BASE*8, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(BASE*16, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(BASE*16, (3, 3), activation='relu', padding='same')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    
    conv5_extend = Conv2D(BASE*32, (3, 3), activation='relu', padding='same')(pool5)
    conv5_extend = Conv2D(BASE*32, (3, 3), activation='relu', padding='same')(conv5_extend)

    up6_extend = concatenate([Conv2DTranspose(BASE*16, (2, 2), strides=(2, 2), padding='same')(conv5_extend), conv5], axis=3)
    conv6_extend = Conv2D(BASE*16, (3, 3), activation='relu', padding='same')(up6_extend)
    conv6_extend = Conv2D(BASE*16, (3, 3), activation='relu', padding='same')(conv6_extend)
    
    up6 = concatenate([Conv2DTranspose(BASE*8, (2, 2), strides=(2, 2), padding='same')(conv6_extend), conv4], axis=3)
    conv6 = Conv2D(BASE*8, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(BASE*8, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(BASE*4, (2, 2),strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(BASE*4, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(BASE*4, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(BASE*2, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(BASE*2, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(BASE*2, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(BASE, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(BASE, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(BASE, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(num_classes, (1, 1), activation='softmax')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    
    if not isinstance(metrics, list):
        metrics = [metrics]

    if include_label_wise_dice_coefficients and num_classes > 1:
        label_wise_dice_metrics = [get_label_dice_coefficient_function(index) for index in range(num_classes)]
        if metrics:
            metrics = metrics + label_wise_dice_metrics
        else:
            metrics = label_wise_dice_metrics
            
    model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, metrics=metrics)
    return model


# train
def train():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train=np.load("/data/ydeng1/pancreatitis/fcn_3Dunet/F_imdbs_2d_patch_2D/X_train_resized.npy") [:,:,:,0]
    imgs_train=np.expand_dims(imgs_train,axis=3)
    imgs_gtruth_train = np.load("/data/ydeng1/pancreatitis/fcn_3Dunet/F_imdbs_2d_patch_2D/Y_train_resized.npy")
    
    print('-'*30)
    print('Loading and preprocessing validation data...')
    print('-'*30)   
    imgs_val=np.load("/data/ydeng1/pancreatitis/fcn_3Dunet/F_imdbs_2d_patch_2D/X_test_resized.npy")[:,:,:,0]
    imgs_val=np.expand_dims(imgs_val,axis=3)
    imgs_gtruth_val  = np.load("/data/ydeng1/pancreatitis/fcn_3Dunet/F_imdbs_2d_patch_2D/Y_test_resized.npy")
      
    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    
    if unet_model_type == 'default':
        model = get_unet_default()
    elif unet_model_type == 'reduced':
        model = get_unet_reduced()
    elif unet_model_type == 'extended':
        model = get_unet_extended()  
        
    model.summary()        
    model = load_model("/data/ydeng1/pancreatitis/fcn_3Dunet/F_outputs_2D/model_Unet.h5")    
    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    #============================================================================
    print('training starting..')
    log_filename = 'F_outputs_2D/' +'model_train.csv' 
    #Callback that streams epoch results to a csv file.
    
    csv_log = callbacks.CSVLogger(log_filename, separator=',', append=True)
    
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=PATIENCE, verbose=0, mode='min')
    
    #checkpoint_filepath = 'outputs/' + image_type +"_best_weight_model_{epoch:03d}_{val_loss:.4f}.hdf5"
    checkpoint_filepath = 'F_outputs_2D/' + 'weights.h5'
    
    checkpoint = callbacks.ModelCheckpoint(checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    
    #callbacks_list = [csv_log, checkpoint]
    callbacks_list = [csv_log, early_stopping, checkpoint]

    #============================================================================
    hist = model.fit(imgs_train, imgs_gtruth_train, batch_size=batch_size, nb_epoch=nb_epochs, verbose=1, validation_data=(imgs_val,imgs_gtruth_val), shuffle=True, callbacks=callbacks_list) #              validation_split=0.2,
             
    model_name = 'F_outputs_2D/' + 'model.h5'
    model.save(model_name)  # creates a HDF5 file 'my_model.h5'

	
# main
if __name__ == '__main__':
    # folder to hold outputs
    if 'F_outputs_2D' not in os.listdir(os.curdir):
        os.mkdir('F_outputs_2D')   
    train()
