
from C_data_handling import load_train_data, load_validatation_data
from sklearn.utils import shuffle
from keras import callbacks
import os,time
from keras.models import Model
from C_model import model
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

Patience=50
batch_size=12
nb_epochs=500

print('-' * 30)
print('Loading and preprocessing train data...')
print('-' * 30)

X_train, Y_train =load_train_data()
X_train=np.repeat(X_train, 3, axis = 3)
print(X_train.shape,Y_train.shape)

X_test,Y_test=load_validatation_data()
X_test=np.repeat(X_test, 3, axis = 3)
print(X_test.shape,Y_test.shape)


def train():
    print('training starting..')
    log_filename = 'C_outputs/'  + 'model_train.csv'
    #Callback that streams epoch results to a csv file.

    csv_log = callbacks.CSVLogger(log_filename, separator=',', append=True)

    early_stopping = callbacks.EarlyStopping(monitor='val_loss',
                                             min_delta=0.0001,
                                             patience=Patience,
                                             verbose=0,
                                             mode='min')


    checkpoint_filepath = 'C_outputs/' + 'weights.h5'

    checkpoint = callbacks.ModelCheckpoint(checkpoint_filepath,
                                           monitor='val_loss',
                                           verbose=1,
                                           save_best_only=True,
                                           mode='min')

    #callbacks_list = [csv_log, checkpoint]
    callbacks_list = [csv_log, early_stopping, checkpoint]

    #============================================================================
    hist = model.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        nb_epoch=nb_epochs,
        verbose=1,
        validation_data=(X_test,Y_test),
        shuffle=True,
        callbacks=callbacks_list)  

    model_name = 'C_outputs/' + 'model.h5'
    model.save(model_name)  # creates a HDF5 file 'my_model.h5'


if __name__ == '__main__':
    # folder to hold outputs

    if 'C_outputs' not in os.listdir(os.curdir):
        os.mkdir('C_outputs')
    train()
