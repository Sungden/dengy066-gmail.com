
import numpy as np
import nibabel as nib
import time
import matplotlib.pyplot as plt
import cv2
import matplotlib
from PIL import Image
import scipy.misc
import os

def load_data(img,gt):  
    x, y, w, h = cv2.boundingRect(gt)
    xi=x-30
    yi=y-30
    wi=w+60
    hi=h+60 
    
#     for i in range(yi,yi + hi):      
#         for j in range(xi,xi+wi):        
#             gt[i,j]=1 

    gt1=gt[yi:(yi+hi),xi:(xi+wi)]     
    img1=img[yi:(yi+hi),xi:(xi+wi)]    
    return img1, gt1,xi,yi,wi,hi

def load_training_data(im,gt,img_test,gt_test):
    #resize to 128*128
    X_train1=np.zeros((5335,192,192,3))
    Y_train1=np.zeros((5335,192,192,2))
    X_test1=np.zeros((281,192,192,3))
    Y_test1=np.zeros((281,192,192,2))
    
    for slice in range(gt.shape[0]):
        gt1=gt[slice,:,:]
        im1=im[slice,:,:]
        X_train,Y_train,xi,yi,wi,hi=load_data(im1,gt1)  
        
        X_name='X_train'+'_'+str(slice)+'.npy'
        Y_name='Y_train'+'_'+str(slice)+'.npy'  
        
        np.save(os.path.join('/data/ydeng1/pancreatitis/fcn_3Dunet/F_imdbs_2d_patch_2D/X_train',X_name),X_train)
        np.save(os.path.join('/data/ydeng1/pancreatitis/fcn_3Dunet/F_imdbs_2d_patch_2D/Y_train',Y_name),Y_train)
        
        X_train = cv2.resize(X_train,(192, 192))
        Y_train=cv2.resize(Y_train,(192, 192))
   
        
        X_train=np.expand_dims(X_train,axis=3)
        X_train=np.expand_dims(X_train,axis=0)
        X_train=np.repeat(X_train, 3, axis = 3)

        Y=np.zeros((Y_train.shape[0],Y_train.shape[1],2))
        Y[:,:,0]=(Y_train==0)
        Y[:,:,1]=(Y_train==1)
        Y_train=np.expand_dims(Y,axis=3)
        Y_train=np.expand_dims(Y,axis=0)   
        
        X_train1[slice,:,:,:]=X_train
        Y_train1[slice,:,:,:]=Y_train
 
    for slice in range(gt_test.shape[0]):
        gt_test1=gt_test[slice,:,:]
        img_test1=img_test[slice,:,:]
        X_test,Y_test,xi,yi,wi,hi=load_data(img_test1,gt_test1)  
        
        X_name='X_test'+'_'+str(slice)+'.npy'
        Y_name='Y_test'+'_'+str(slice)+'.npy'
        
        np.save(os.path.join('/data/ydeng1/pancreatitis/fcn_3Dunet/F_imdbs_2d_patch_2D/X_test',X_name),X_test)
        np.save(os.path.join('/data/ydeng1/pancreatitis/fcn_3Dunet/F_imdbs_2d_patch_2D/Y_test',Y_name),Y_test) 
        
        X_test=cv2.resize(X_test,(192, 192))
        Y_test=cv2.resize(Y_test,(192, 192))
        
        X_test=np.expand_dims(X_test,axis=3)
        X_test=np.expand_dims(X_test,axis=0)
        X_test=np.repeat(X_test, 3, axis = 3)

        Y=np.zeros((Y_test.shape[0],Y_test.shape[1],2))
        Y[:,:,0]=(Y_test==0)
        Y[:,:,1]=(Y_test==1)
        Y_test=np.expand_dims(Y,axis=3)
        Y_test=np.expand_dims(Y,axis=0)  

        
        X_test1[slice,:,:,:]=X_test
        Y_test1[slice,:,:,:]=Y_test
        
    return X_train1, Y_train1, X_test1, Y_test1






        




    
    
     

        


