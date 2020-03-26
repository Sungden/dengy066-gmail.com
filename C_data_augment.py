
#add some pancreas data to pancreatitis data and augment pancreatitis data by rot90 and flip

import numpy as np
import os
import nibabel as nib
import cv2

im_dir='/data/ydeng1/pancreas_data/data/images/'  #pancreas data
gt_dir='/data/ydeng1/pancreas_data/data/labels/'

im_dir_1='/data/ydeng1/pancreatitis/fcn_3Dunet/data/train_data/' #pancreatitis data
gt_dir_1='/data/ydeng1/pancreatitis/fcn_3Dunet/data/train_label/'

new_im_dir='/data/ydeng1/pancreatitis/fcn_3Dunet/dataset/images/' #data augmentate dir
new_gt_dir='/data/ydeng1/pancreatitis/fcn_3Dunet/dataset/labels/'
print(len(os.listdir(im_dir)),len(os.listdir(gt_dir)))

# data augment by 
print('----'*20)

for img,gt in zip (sorted(os.listdir(im_dir))[0:30],sorted(os.listdir(gt_dir))[0:30]):  #add 30 patients of pancreas
    directory1=np.load(os.path.join(im_dir,img))
    directory2=np.load(os.path.join(gt_dir,gt))    
   
    # truncated to [-100,240]  
    #directory1[directory1<=-100]=-100
    #directory1[directory1>=240]=240
    
    img_img = nib.Nifti1Image(directory1, np.eye(4))
    img_gt = nib.Nifti1Image(directory2, np.eye(4))
    #nib.save(img_img, os.path.join(new_im_dir,img[0:4]+'.nii.gz'))
    #nib.save(img_gt, os.path.join(new_gt_dir,gt[0:4]+'.nii.gz'))
    
  
for i,j in zip (sorted(os.listdir(im_dir_1)), sorted(os.listdir(gt_dir_1))):
  print(len(os.listdir(im_dir_1)),len(os.listdir(gt_dir_1)))
  im=nib.load(os.path.join(im_dir_1,i)).get_data()
  gt=nib.load(os.path.join(gt_dir_1,j)).get_data()
  #im=np.rot90(im,1)
  #gt=np.rot90(gt,1)
  im = cv2.flip(im, -1)
  gt = cv2.flip(gt, -1)
  
  # truncated to [800,1200]  
  #im[im<=800] =800
  #im[im>=1200] =1200
  
  img_img_p = nib.Nifti1Image(im, np.eye(4))
  img_gt_p = nib.Nifti1Image(gt, np.eye(4))
  nib.save(img_img_p, os.path.join(new_im_dir,i[0:18]+'_flip-1.nii.gz'))
  nib.save(img_gt_p, os.path.join(new_gt_dir,j[0:24]+'_flip-1.nii.gz'))
  
  #np.save(os.path.join(new_im_dir,i[0:18]+'_rote-90.npy'),im)  
  #np.save(os.path.join(new_gt_dir,j[0:24]+'_rote-90.npy'),gt)