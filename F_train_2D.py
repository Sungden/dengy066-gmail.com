
from extract_boundingbox import load_training_data
from sklearn.utils import shuffle
from keras import callbacks
import os,time
from keras.models import Model
from F_model_2D import fine_model
import numpy as np

Patience=30
batch_size=80
nb_epochs=500

print('-' * 30)
print('Loading and preprocessing train data...')
print('-' * 30)

def train():
    print('training starting..')
    log_filename = 'F_outputs_2D/'  + 'model_train.csv'
    #Callback that streams epoch results to a csv file.

    csv_log = callbacks.CSVLogger(log_filename, separator=',', append=True)

    early_stopping = callbacks.EarlyStopping(monitor='val_loss',
                                             min_delta=0.0001,
                                             patience=Patience,
                                             verbose=0,
                                             mode='min')


    checkpoint_filepath = 'F_outputs_2D/' + 'weights.h5'

    checkpoint = callbacks.ModelCheckpoint(checkpoint_filepath,
                                           monitor='val_loss',
                                           verbose=1,
                                           save_best_only=True,
                                           mode='min')

    #callbacks_list = [csv_log, checkpoint]
    callbacks_list = [csv_log, early_stopping, checkpoint]

    #============================================================================
    hist = fine_model.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        nb_epoch=nb_epochs,
        verbose=1,
        validation_data=(X_test,Y_test),
        shuffle=True,
        callbacks=callbacks_list)  

    model_name = 'F_outputs_2D/' + 'model.h5'
    fine_model.save(model_name)  # creates a HDF5 file 'my_model.h5'
 
if __name__ == '__main__':
    # folder to hold outputs

    if 'F_outputs_2D' not in os.listdir(os.curdir):
        os.mkdir('F_outputs_2D')
   
    gt=np.load("/data/ydeng1/pancreatitis/fcn_3Dunet/C_imdbs_2d_patch_AP/patches_training_gtruth_2d.npy")
    gt=gt[:,:,:,1]
    gt = np.array(gt,np.uint8)
    print(gt.shape)
    im=np.load("/data/ydeng1/pancreatitis/fcn_3Dunet/C_imdbs_2d_patch_AP/patches_training_imgs_2d.npy")
    im=im[:,:,:,0]
    print(im.shape)

    gt_test=np.load("/data/ydeng1/pancreatitis/fcn_3Dunet/C_imdbs_2d_patch_AP/patches_val_gtruth_2d.npy")
    gt_test=gt_test[:,:,:,1]
    gt_test = np.array(gt_test,np.uint8)
    print(gt_test.shape)

    img_test=np.load("/data/ydeng1/pancreatitis/fcn_3Dunet/C_imdbs_2d_patch_AP/patches_val_imgs_2d.npy")
    img_test=img_test[:,:,:,0]

    X_train, Y_train, X_test, Y_test=load_training_data(im,gt,img_test,gt_test)
    np.save('/data/ydeng1/pancreatitis/fcn_3Dunet/F_imdbs_2d_patch_2D/X_train_resized.npy',X_train)
    np.save('/data/ydeng1/pancreatitis/fcn_3Dunet/F_imdbs_2d_patch_2D/Y_train_resized.npy',Y_train)
    np.save('/data/ydeng1/pancreatitis/fcn_3Dunet/F_imdbs_2d_patch_2D/X_test_resized.npy',X_test)
    np.save('/data/ydeng1/pancreatitis/fcn_3Dunet/F_imdbs_2d_patch_2D/Y_test_resized.npy',Y_test)  
    train()
