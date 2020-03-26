import matplotlib.pyplot as plt
from C_data_handling import load_train_data, load_validatation_data
from sklearn.utils import shuffle
from keras import callbacks
import os,time
from keras.models import Model
from F_train_Unet import get_unet_default
import numpy as np
import nibabel as nib
from extract_boundingbox import load_data
import cv2

# X_train, Y_train =load_train_data()
# X_train=np.repeat(X_train, 3, axis = 3)
# print('X_train.shape:',X_train.shape,'Y_train.shape:',Y_train.shape)
X_dir='/data/ydeng1/pancreatitis/fcn_3Dunet/C_result_png/coarse_result/'
X_img=sorted(os.listdir(X_dir))

Y_dir='/data/ydeng1/pancreatitis/fcn_3Dunet/C_result_png/bbox_result/'
Y_img=sorted(os.listdir(Y_dir))

X_test1=np.zeros((len(X_img),192,192))
Y_test1=np.zeros((len(Y_img),192,192))
for img,i in zip(X_img,range(len(X_img))):
  X1=cv2.imread(os.path.join(X_dir,img))
  X2=X1[:,:,0]
  X=cv2.resize(X2,(192, 192))
  X_test1[i,:,:]=X
  
X_test1=(X_test1-np.min(X_test1))/(np.max(X_test1)-np.min(X_test1))  
X_test1=np.expand_dims(X_test1,axis=3)

for img,i in zip(Y_img,range(len(Y_img))):
  Y=cv2.imread(os.path.join(Y_dir,img))
  Y=cv2.resize(Y,(192, 192))
  Y_test1[i,:,:]=Y[:,:,0]


#X_test1=np.load('/data/ydeng1/pancreatitis/fcn_3Dunet/F_imdbs_2d_patch_2D/X_test_resized.npy')[:,:,:,0]
#X_test1=np.expand_dims(X_test1,axis=3)
#Y_test1=np.load('/data/ydeng1/pancreatitis/fcn_3Dunet/F_imdbs_2d_patch_2D/Y_test_resized.npy')
#X_test1=np.repeat(X_test1, 3, axis = 3)
# print('X_test.shape:',X_test.shape,'Y_test.shape:',Y_test.shape)

fine_model= get_unet_default()
fine_model.load_weights('/data/ydeng1/pancreatitis/fcn_3Dunet/F_outputs_2D/weights.h5')
y_pred1 = fine_model.predict(X_test1)
print(y_pred1.shape,np.max(y_pred1[:,:,:,1]))
y_pred2= y_pred1[:,:,:,1].copy()
y_predi = y_pred1[:,:,:,1]
y_predi[y_predi>=0.17]=1
y_predi[y_predi<0.17]=0
Y_test1[Y_test1>0]=1
y_testi=Y_test1
print(y_testi.shape,y_predi.shape)
print(np.sum(y_testi),np.sum(y_predi),np.sum(y_testi * y_predi))

def computeDice(img_true,img_pre):
    intersection = np.sum(img_true * img_pre)
    dsc= (2. * intersection) / (np.sum(img_true) + np.sum(img_pre))
    return dsc
print(computeDice(y_testi,y_predi))

#np.save('/data/ydeng1/pancreatitis/fcn_3Dunet/result/coarse_result_original.npy',y_pred1[:,:,:,1])
for i in range(X_test1.shape[0]):
    plt.subplot(2,4,1)
    plt.imshow(X_test1[i,:,:,0],'gray')
    plt.subplot(2,4,2)
    plt.imshow(y_pred2[i,:,:],'gray')
    plt.subplot(2,4,3)
    plt.imshow(y_predi[i,:,:],'gray')  
    plt.subplot(2,4,4)
    plt.imshow(y_testi[i,:,:],'gray')   
    plt.subplot(2,4,5)
    y_testi = np.array(y_testi,np.uint8)
    img1,  gt1,_,_,_,_=load_data(X_test1[i,:,:,0],y_testi[i,:,:]) 
    plt.imshow(img1,'gray')
    plt.subplot(2,4,6)
    plt.imshow(gt1,'gray')
    plt.subplot(2,4,7)
    y_predi = np.array(y_predi,np.uint8)
    img, gt,_,_,_,_=load_data(X_test1[i,:,:,0],y_predi[i,:,:])   
    plt.imshow(img,'gray')    
    plt.subplot(2,4,8)
    plt.imshow(gt,'gray')
    plt.show()




