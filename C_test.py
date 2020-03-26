import matplotlib.pyplot as plt
from C_data_handling import load_train_data, load_validatation_data
from sklearn.utils import shuffle
from keras import callbacks
import os,time
from keras.models import Model
from C_model import model
import numpy as np
import nibabel as nib
from extract_boundingbox import load_data
import imageio
import os
import keras.backend as K
import cv2
#X_test1,Y_test1 =load_train_data()
#X_test1=np.repeat(X_test1, 3, axis = 3)
# print('X_train.shape:',X_train.shape,'Y_train.shape:',Y_train.shape)

X_test1,Y_test1=load_validatation_data()
X_test1=np.repeat(X_test1, 3, axis = 3)
# print('X_test.shape:',X_test.shape,'Y_test.shape:',Y_test.shape)

model.load_weights('/data/ydeng1/pancreatitis/fcn_3Dunet/C_outputs/weights.h5')
print(X_test1.shape,'9999999')
##select specfical layer outputs
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer('block3_pool').output)
y_pred1 = intermediate_layer_model.predict(X_test1)

print(y_pred1.shape,'444444444444444444')


#y_pred1 = model.predict(X_test1)
print(y_pred1.shape,np.max(y_pred1[:,:,:,1]))
y_pred2= y_pred1[:,:,:,1].copy()
y_predi = y_pred1[:,:,:,1]
y_predi[y_predi>=0.4]=1
y_predi[y_predi<0.4]=0
y_testi=np.argmax(Y_test1, axis=3)


def computeDice(img_true,img_pre):
    intersection = np.sum(img_true * img_pre)
    dsc= (2. * intersection) / (np.sum(img_true) + np.sum(img_pre))
    return dsc
#print(computeDice(y_testi,y_predi))

####save coarse result feature map into png image####
path='/data/ydeng1/pancreatitis/fcn_3Dunet/C_result_png/Validation_pooling3'
for i in range(y_pred2.shape[0]):
    img=y_pred2[i,:,:]
    gt=y_testi[i,:,:]
    gt=cv2.resize(gt,(64, 64))
    path1=os.path.join(path,str(i),'images')
    path2=os.path.join(path,str(i),'masks')
    if not os.path.exists(path1):
        os.makedirs(path1)
    im_name=str(i)+'.png'
    imageio.imsave(os.path.join(path1,im_name),img) 
    if not os.path.exists(path2):
        os.makedirs(path2)
    gt_name=str(i)+'.png'
    imageio.imsave(os.path.join(path2,gt_name),gt)
    
#np.save('/data/ydeng1/pancreatitis/fcn_3Dunet/result/coarse_result_original.npy',y_pred1[:,:,:,1])
#np.save('/data/ydeng1/pancreatitis/fcn_3Dunet/result/coarse_result.npy',y_predi)

####save coarse result into png image####
path='/data/ydeng1/pancreatitis/fcn_3Dunet/result/coarse_result/'
for i in range(y_predi.shape[0]):
    pre=y_predi[i,:,:]
    im_name=str(i)+'.png'
    #imageio.imsave(os.path.join(path,im_name),pre) 




for i in range(X_test1.shape[0]):
    plt.subplot(2,4,1)
    plt.imshow(X_test1[i,:,:,1],'gray')
    plt.subplot(2,4,2)
    plt.imshow(y_pred2[i,:,:],'gray')
    plt.subplot(2,4,3)
    plt.imshow(y_predi[i,:,:],'gray')  
    plt.subplot(2,4,4)
    plt.imshow(y_testi[i,:,:],'gray')   
    plt.subplot(2,4,5)
    y_testi = np.array(y_testi,np.uint8)
    img1,  gt1,_,_,_,_=load_data(X_test1[i,:,:,1],y_testi[i,:,:]) 
    plt.imshow(img1,'gray')
    plt.subplot(2,4,6)
    plt.imshow(gt1,'gray')
    plt.subplot(2,4,7)
    y_predi = np.array(y_predi,np.uint8)
    img, gt,_,_,_,_=load_data(X_test1[i,:,:,1],y_predi[i,:,:])   
    plt.imshow(img,'gray')    
    plt.subplot(2,4,8)
    plt.imshow(gt,'gray')
    plt.show()




