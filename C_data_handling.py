
from __future__ import print_function

# import packages
import time, os#, random, cv2
import numpy as np
import nibabel as nib
from sklearn.feature_extraction.image import extract_patches
import matplotlib.pyplot as plt


# init configs
image_rows = 512
image_cols = 512
num_classes =2

# patch extraction parameters
patch_size = 512
extraction_step = 512

# create npy data
def create_npy_data(train_imgs_path,is_train):
    # empty matrix to hold patches
    patches_training_imgs_2d=np.empty(shape=[0, patch_size,patch_size], dtype='int16')
    patches_training_gtruth_2d=np.empty(shape=[0, patch_size,patch_size, num_classes], dtype='int16')
    
    #76 pancreastis plus 20 pacnreas data as training data ,19 as test data.
    images_train_dir = sorted(os.listdir(train_imgs_path))
    train_gts_dir=sorted(os.listdir(train_gts_path))
    
    #print(images_train_dir)
    #print(train_gts_dir)
    
    start_time = time.time()
  
    j=0
    print('-'*30)
    print('Creating training2d_patches...')
    print('-'*30)


    # for each volume do:
    for img_dir_name,gt_dir_name in zip(images_train_dir,train_gts_dir):
        patches_training_imgs_2d_temp = np.empty(shape=[0,patch_size,patch_size], dtype='int16')
        patches_training_gtruth_2d_temp = np.empty(shape=[0,patch_size,patch_size,num_classes], dtype='int16')
        print('Processing: volume {0} / {1} volume images'.format(j+1, len(images_train_dir)))
        
        # volume         
        img_name  = img_dir_name
        img_name = os.path.join(train_imgs_path, img_dir_name)
  
        # groundtruth
        img_seg_name = os.path.join(train_gts_path,gt_dir_name)
        
      
        # load volume, gt 
        img = nib.load(img_name).get_data()
        img_data = img
        img_data = np.squeeze(img_data)
        #0-1 normalizaion
        img_data=(img_data-np.min(img_data))/(np.max(img_data)-np.min(img_data))
    
        img_gtruth = nib.load(img_seg_name).get_data()
        img_gtruth_data = img_gtruth
        img_gtruth_data = np.squeeze(img_gtruth_data)

        print(img_data.shape,img_gtruth_data.shape,'8888888888888888')
        print(img_dir_name,gt_dir_name)
        # for each slice do
        for slice in range(img_gtruth_data.shape[2]):
            patches_training_imgs_2d_slice_temp = np.empty(shape=[0,patch_size,patch_size], dtype='int16')
            patches_training_gtruth_2d_slice_temp = np.empty(shape=[0,patch_size,patch_size,num_classes], dtype='int16')
            if np.count_nonzero(img_gtruth_data[:,:,slice])>900 : #pancreatitis over 100
        
                # extract patches of the jth volum image
                imgs_patches, gt_patches = extract_2d_patches(img_data[:,:,slice], \
                                                              img_gtruth_data[:,:,slice])
                
        
                # update database
                patches_training_imgs_2d_slice_temp  = np.append(patches_training_imgs_2d_slice_temp,imgs_patches, axis=0)
                patches_training_gtruth_2d_slice_temp  = np.append(patches_training_gtruth_2d_slice_temp,gt_patches, axis=0)
        
            patches_training_imgs_2d_temp  = np.append(patches_training_imgs_2d_temp,patches_training_imgs_2d_slice_temp, axis=0)
            patches_training_gtruth_2d_temp  = np.append(patches_training_gtruth_2d_temp,patches_training_gtruth_2d_slice_temp, axis=0)
               
        patches_training_imgs_2d  = np.append(patches_training_imgs_2d,patches_training_imgs_2d_temp, axis=0)
        patches_training_gtruth_2d  = np.append(patches_training_gtruth_2d,patches_training_gtruth_2d_temp, axis=0)
        j += 1
        X  = patches_training_imgs_2d.shape
        Y  = patches_training_gtruth_2d.shape
        print('shape im: [{0} , {1} , {2}]'.format(X[0], X[1], X[2])) 
        print('shape gt: [{0} , {1} , {2}, {3}]'.format(Y[0], Y[1], Y[2], Y[3]))

    #convert to single precission
    patches_training_imgs_2d = patches_training_imgs_2d.astype('float32')
    patches_training_imgs_2d = np.expand_dims(patches_training_imgs_2d, axis=3)
    
    end_time = time.time()
    print("Elapsed time was %g seconds" % (end_time - start_time))
    
    X  = patches_training_imgs_2d.shape
    Y  = patches_training_gtruth_2d.shape
    
    print('-'*30)
    print('Training set detail...')
    print('-'*30)
    print('shape im: [{0} , {1} , {2}, {3}]'.format(X[0], X[1], X[2], X[3]))
    print('shape gt: [{0} , {1} , {2}, {3}]'.format(Y[0], Y[1], Y[2], Y[3]))

    S  = patches_training_imgs_2d.shape
    print('Done: {0} 2d patches added from {1} volume images'.format(S[0], j))
    print('Loading done.')

    print('Saving to .npy files done.')

    # save train or validation
    if is_train:
        np.save('C_imdbs_2d_patch_AP/patches_training_imgs_2d.npy', patches_training_imgs_2d)
        np.save('C_imdbs_2d_patch_AP/patches_training_gtruth_2d.npy', patches_training_gtruth_2d)
    else:
        np.save('C_imdbs_2d_patch_AP/patches_val_imgs_2d.npy', patches_training_imgs_2d)
        np.save('C_imdbs_2d_patch_AP/patches_val_gtruth_2d.npy', patches_training_gtruth_2d)        
    print('Saving to .npy files done.')

# extract 2d patches
def extract_2d_patches(img_data, gt_data):
    patch_shape =(patch_size,patch_size)
    # empty matrix to hold patches
    imgs_patches_per_slice=np.empty(shape=[0,patch_size,patch_size], dtype='int16')
    gt_patches_per_slice=np.empty(shape=[0,patch_size,patch_size], dtype='int16')
    
      
    img_patches = extract_patches(img_data, patch_shape, extraction_step)
    gt_patches = extract_patches(gt_data, patch_shape, extraction_step)
  
 
    # extract patches which has center pixel lying inside mask    
    rows = []; cols = []
    for i in range(0,img_patches.shape[0]):        
        for j in range(0,img_patches.shape[1]):
            if np.count_nonzero(gt_patches[i,j,:,:])>=900:     #pancreas at  least 200 pixels
                rows.append(i)
                cols.append(j)
                plt.imshow(gt_patches[i,j,:,:],'gray')



    # number of n0m zero patches
    N = len(rows)

    M = len(rows)

    # select nonzeropatches index
    selected_img_patches = img_patches[rows,cols,:,:]
    selected_gt_patches  = gt_patches [rows,cols,:,:]

    # update database
    imgs_patches_per_slice  = np.append(imgs_patches_per_slice,selected_img_patches, axis=0)
    gt_patches_per_slice  = np.append(gt_patches_per_slice,selected_gt_patches, axis=0)

    gt_patches_per_slice = separate_labels(gt_patches_per_slice)
    return imgs_patches_per_slice, gt_patches_per_slice


# separate labels
def separate_labels(patch_3d_volume):
    result =np.empty(shape=[0,patch_size,patch_size,num_classes], dtype='int16')
    N = patch_3d_volume.shape[0]
    # for each class do:
    for V in range(N):
        V_patch = patch_3d_volume[V , :, :]
        U  = np.unique(V_patch)
        unique_values = list(U)
        result_v =np.empty(shape=[patch_size,patch_size,0], dtype='int16')
        if num_classes==3:
            start_point = 1
        else:
            start_point = 0
        for label in range(start_point,2):
            if label in unique_values:
                im_patch = V_patch == label
                im_patch = im_patch*1
            else:
                im_patch = np.zeros((V_patch.shape))
             
            im_patch = np.expand_dims(im_patch, axis=2) 
            result_v  = np.append(result_v,im_patch, axis=2)
        
   
        result_v = np.expand_dims(result_v, axis=0) 

        result  = np.append(result,result_v, axis=0)
    return result

# load train npy    
def load_train_data():
    imgs_train = np.load('C_imdbs_2d_patch_AP/patches_training_imgs_2d.npy')
    imgs_gtruth_train = np.load('C_imdbs_2d_patch_AP/patches_training_gtruth_2d.npy')
    return imgs_train, imgs_gtruth_train

# load validation npy
def load_validatation_data():
    imgs_validation = np.load('C_imdbs_2d_patch_AP/patches_val_imgs_2d.npy')
    gtruth_validation = np.load('C_imdbs_2d_patch_AP/patches_val_gtruth_2d.npy')
    return imgs_validation, gtruth_validation

# main
if __name__ == '__main__':
    if 'C_imdbs_2d_patch_AP' not in os.listdir(os.curdir):
        os.mkdir('C_imdbs_2d_patch_AP')
    train_imgs_path= '/data/ydeng1/pancreatitis/fcn_3Dunet/dataset/images/'
    train_gts_path='/data/ydeng1/pancreatitis/fcn_3Dunet/dataset/labels/'
    #train_imgs_path=  '/data/ydeng1/pancreatitis/fcn_3Dunet/data/val_data'
    #train_gts_path='/data/ydeng1/pancreatitis/fcn_3Dunet/data/val_label/'
 
    #print(sorted(os.listdir(train_imgs_path)) )
    create_npy_data(train_imgs_path, 1)
   
    #create_npy_data(train_imgs_path,0)
