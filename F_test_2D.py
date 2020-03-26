import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras import callbacks
import os,time
from keras.models import Model
from F_train_Unet import get_unet_default
from C_model import model
import numpy as np
import nibabel as nib
from extract_boundingbox import load_data
import cv2

CUDA_VISIBLE_DEVICES=1

gt_test=np.load("/data/ydeng1/pancreatitis/fcn_3Dunet/C_imdbs_2d_patch_AP/patches_val_gtruth_2d.npy")
gt_test = np.array(gt_test,np.uint8)
print(gt_test.shape)

img_test=np.load("/data/ydeng1/pancreatitis/fcn_3Dunet/C_imdbs_2d_patch_AP/patches_val_imgs_2d.npy")
img_test=np.repeat(img_test, 3, axis = 3)

img_name='test'  
seg_name=img_name+'_fine_result.npy'
coarse_name=img_name+'_coarse_result.npy'
start_time = time.time()

outputfile_coarse=os.path.join('/data/ydeng1/pancreatitis/fcn_3Dunet/result/',coarse_name)
outputfile_fine=os.path.join('/data/ydeng1/pancreatitis/fcn_3Dunet/result/',seg_name)

X_test=img_test #x,y,z
X_test=(X_test-np.min(X_test))/(np.max(X_test)-np.min(X_test))

model.load_weights('/data/ydeng1/pancreatitis/fcn_3Dunet/C_outputs/weights.h5')
y_pred = model.predict(X_test)
print('y_pred.shape:',y_pred.shape)

y_predi=y_pred[:,:,:,1]
y_predi[y_predi>=0.4]=1
y_predi[y_predi<0.4]=0
#y_predi = np.argmax(y_pred, axis=3)
y_predi = np.array(y_predi,np.uint8)  
print('y_pred.shape:',y_predi.shape)

segmentation_coarse=np.float32(y_predi) #x,y,z
#np.save(outputfile_coarse,segmentation_coarse )
segmentation_fine=np.zeros((X_test.shape[0],512,512)) #x,y,z

fine_model= get_unet_default()
fine_model.summary()
fine_model.load_weights('/data/ydeng1/pancreatitis/fcn_3Dunet/F_outputs_2D/weights.h5')
for i in range(y_predi.shape[0]):

    plt.subplot(2,4,1)
    plt.imshow(gt_test[i,:,:,1],'gray')
    plt.subplot(2,4,2)
    plt.imshow(img_test[i,:,:,0],'gray')
    X_test, Y_test,xi,yi,wi,hi=load_data(img_test[i,:,:,1],y_predi[i,:,:])
    X_name='X_test_coarse'+'_'+str(i)+'.npy'
    Y_name='Y_test_coarse'+'_'+str(i)+'.npy'  
    #np.save(os.path.join('/data/ydeng1/pancreatitis/fcn_3Dunet/F_imdbs_2d_patch_2D_AP/X_test_coarse',X_name),X_test)
    #np.save(os.path.join('/data/ydeng1/pancreatitis/fcn_3Dunet/F_imdbs_2d_patch_2D_AP/Y_test_coarse',Y_name),Y_test)
        
          
    #print(X_test.shape,'$$$$$$')

    #print(X_test.shape)
    plt.subplot(2,4,3)
    plt.imshow(X_test,'gray')
    
    plt.subplot(2,4,4)
    plt.imshow(Y_test,'gray')   
    
    img_test1, gt_test1,x,y,w,h=load_data(img_test[i,:,:,1],gt_test[i,:,:,1])
    plt.subplot(2,4,5)
    plt.imshow(img_test1,'gray')
    plt.subplot(2,4,6)
    plt.imshow(gt_test1,'gray')    
    
    if np.count_nonzero(X_test)>0:
        X_test=cv2.resize(X_test,(192, 192))
        #print(X_test.shape,'%%%%%')

        X_test=np.expand_dims(X_test,axis=0)
        #print(X_test.shape,'xxxx')
        X_test=np.expand_dims(X_test,axis=3)
        #print(X_test.shape,'wwwwwwwwww')
        #X_test=np.repeat(X_test, 3, axis = 3)
        y_pred_fine = fine_model.predict(X_test) #z,x,y
        
        y_pred_fine1=np.copy(y_pred_fine)
        y_pred_fine2=y_pred_fine[:,:,:,1]
        #print(y_pred_fine2.shape,'YYYYYYY')
      
        y_pred_fine2[y_pred_fine2>=0.52]=1
        y_pred_fine2[y_pred_fine2<0.52]=0 
         
        #y_pred_fine = np.argmax(y_pred_fine, axis=3)
        y_pred_fine=np.squeeze(y_pred_fine2)
       # print(y_pred_fine.shape,Y_test.shape,'999999999999')
        y_pred_fine=cv2.resize(y_pred_fine.astype('float32'),(Y_test.shape[1], Y_test.shape[0]))
        plt.subplot(2,4,7)
        plt.imshow(y_pred_fine,'gray')
        
        y_prediciton=np.zeros((512,512))
        y_prediciton[yi:yi+hi,xi:xi+wi]=y_pred_fine
        plt.subplot(2,4,8)
        plt.imshow(y_prediciton,'gray')
        plt.show()
        segmentation_fine[i,:,:]=y_prediciton#z,x,y

#segmentation_fine1=segmentation_fine.swapaxes(0,1).swapaxes(1,2) #x,y,z
segmentation_fine = np.float32(segmentation_fine)

#np.save(outputfile_fine,segmentation_fine )
end_time = time.time()
print("Elapsed time was %g seconds" % (end_time - start_time))
    
def computeDice(img_true,img_pre):
    intersection = np.sum(img_true * img_pre)
    dsc= (2. * intersection) / (np.sum(img_true) + np.sum(img_pre))
    return dsc

print('coarse_dice:',computeDice(gt_test[:,:,:,1],segmentation_coarse))
print('fine_dice:',computeDice(gt_test[:,:,:,1],segmentation_fine))

