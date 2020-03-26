import os
import numpy as np
from lib.model1 import RPN
from lib import utils as ut
from train import NucleiSequence ,NucleiConfig
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from keras.preprocessing.image import img_to_array, load_img
import statistics
import cv2
import imageio

VALIDATION_PATH = '/data/ydeng1/pancreatitis/fcn_3Dunet/C_result_png/validation/' # test data path
coarse_result_dir='/data/ydeng1/pancreatitis/fcn_3Dunet/result/coarse_result/' #coarse seg result
coarse_result_img=os.listdir(coarse_result_dir)
coarse_result_img.sort()
coarse_result_img.sort(key = lambda x: int(x[:-4])) 
coarse_img_dir='/data/ydeng1/pancreatitis/fcn_3Dunet/C_result_png/coarse_result/' # image corresponding to coarse seg result
RPN_result_dir='/data/ydeng1/pancreatitis/fcn_3Dunet/C_result_png/RPN_result2/'    #image corresponding to RPN localization
bbox_result_dir='/data/ydeng1/pancreatitis/fcn_3Dunet/C_result_png/bbox_result/'  #image corresponding to gt localization

VALIDATION_img=np.load("/data/ydeng1/pancreatitis/fcn_3Dunet/C_imdbs_2d_patch_AP/patches_val_imgs_2d.npy") #test data
figure_dir='/data/ydeng1/pancreatitis/fcn_3Dunet/fcn-rpn/keras-rpn-master/keras-rpn-master/figure3/' # path to save show_img

class NucleiInferenceConfig(NucleiConfig):

    # Data parameters
    IMAGE_SHAPE = (512, 512)
    ANCHOR_SCALES = (16, 32, 64, 128, 256)
    TRAIN_ANCHORS_PER_IMAGE = 256
    MEAN_PIXEL = np.array([15.53, 14.56, 13.22])

    LOGS = '/data/ydeng1/pancreatitis/fcn_3Dunet/fcn-rpn/keras-rpn-master/keras-rpn-master/logs/pancreatitis/'

    # Path to the weights file
    WEIGHTS_FILE ="/data/ydeng1/pancreatitis/fcn_3Dunet/fcn-rpn/keras-rpn-master/keras-rpn-master/logs/pancreatitis/03-05-20_12.30.22/cnn_weights/rpn_weights.58.hdf5"
   
    
#Calculate IoU (Intersection of Union)
def union(au, bu, area_intersection):
	area_a = (au[2] - au[0]) * (au[3] - au[1])
	area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
	area_union = area_a + area_b - area_intersection
	return area_union

def intersection(ai, bi):
	x = max(ai[0], bi[0])
	y = max(ai[1], bi[1])
	w = min(ai[2], bi[2]) - x
	h = min(ai[3], bi[3]) - y
	if w < 0 or h < 0:
		return 0
	return w*h

def iou(a, b):
	# a and b should be (x1,y1,x2,y2) m
	if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
		return 0.0
	area_i = intersection(a, b)
	area_u = union(a, b, area_i)
	return float(area_i) / float(area_u + 1e-6)        


def recall(a, b):
	# a and b should be (x1,y1,x2,y2)
	if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
		return 0.0
	area_i = intersection(a, b)
	area_u = (b[2]-b[0])*(b[3]-b[1]) # b is bbox
	return float(area_i) / float(area_u + 1e-6) 


def accuracy(a, b):
	# a and b should be (x1,y1,x2,y2)
	if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
		return 0.0
	area_i = intersection(a, b)
	area_u = (a[2]-a[0])*(a[3]-a[1]) # a is prediction
	return float(area_i) / float(area_u + 1e-6) 

def main():
    # Configuration
    config = NucleiInferenceConfig()
    # Nucleus dataset
    dataset = NucleiSequence(VALIDATION_PATH, config)   
    #calculte iou
    iou_RPN=[]
    iou_coarse=[]
    recall_RPN=[]
    recall_coarse=[]
    acc_RPN=[]
    acc_coarse=[]
        
    # Select a random sample from the validation set
    for idx in range(0,len(dataset)):
    #random_idx = np.random.randint(0, len(dataset))
        coarse_img=cv2.imread(os.path.join(coarse_result_dir,coarse_result_img[idx]))
        inputs, mask = dataset[idx]

        for random_idx in range(0,len(inputs[0])):
            #random_idx = np.random.randint(0, len(inputs[0]))
            # Select a random batch index
            image_gt = inputs[0][random_idx]
            rpn_match_gt = inputs[1][random_idx]
            rpn_bbox_gt = inputs[2][random_idx]
            gt=mask[random_idx]
            fig, axes = plt.subplots(ncols=3,nrows=2,sharex=False, sharey=True,figsize=(15,15))
            
            axes[0][0].imshow(VALIDATION_img[idx,:,:,0], cmap='gray')
            axes[0][0].set_title('original img',fontsize=10)
            axes[0][2].imshow(image_gt[:,:,0], cmap='gray')
            axes[0][2].set_title('feature map',fontsize=10)
            axes[0][1].imshow(gt[:,:,0], cmap='gray')
            axes[0][1].set_title('ground_truth',fontsize=10)  
            # Create the ground truth bounding boxes
            anchors = dataset.anchors       
            positive_anchors = np.where(rpn_match_gt == 1)[0]
            
            print(anchors.shape,positive_anchors.shape,'%%%%%%%%%%%%%%%%')
            bboxes_gt = ut.shift_bboxes(anchors[positive_anchors], rpn_bbox_gt[:positive_anchors.shape[0]] * config.RPN_BBOX_STD_DEV)
            # Visualize the ground truth bounding boxes
            
            axes[1][1].imshow(image_gt[:,:,0],cmap='gray')
            axes[1][1].set_title('localization results',fontsize=10)
            for a in bboxes_gt:
                rect = patches.Rectangle((a[1], a[0]), a[3]-a[1], a[2]-a[0], linewidth=1.3, edgecolor='r', facecolor='none', linestyle='-')
                rectt = patches.Rectangle((a[1]-30, a[0]-30), a[3]-a[1]+60, a[2]-a[0]+60, linewidth=1.3, edgecolor='r', facecolor='none', linestyle='-')
                axes[1][1].add_patch(rect)  
                

            # Visualize the RPN targets of positive and negative anchors
            #ut.visualize_training_anchors(anchors, rpn_match_gt, np.uint8(image_gt + config.MEAN_PIXEL))
            
            ##save bbox gt to calacute DSC, need to resize
            RPN_name=str(idx)+'.png'
            label=gt[:,:,0]
            #imageio.imsave(os.path.join(bbox_result_dir,RPN_name),label[int(a[0]-30):int(a[2]+30),int(a[1]-30):int(a[3]+30)])
            
                        
            # Create the Region Proposal Network and load the trained weights
            assert os.path.exists(config.WEIGHTS_FILE)
            rpn = RPN(config, 'inference')
            rpn.model.load_weights(config.WEIGHTS_FILE, by_name=True)

            # Predict the positive anchors
            rpn_match, rpn_bbox = rpn.model.predict(np.expand_dims(image_gt, 0))
            
            print(rpn_match.shape,'88888888888888888888')
            print(rpn_bbox.shape,'222222222222')
            rpn_match = np.squeeze(rpn_match)
            rpn_bbox = np.squeeze(rpn_bbox) * config.RPN_BBOX_STD_DEV
            
            print(len(rpn_match),'88888888888888888888')
            print(len(rpn_bbox),'222222222222')
            
            # Find where positive predictions took place
            positive_idxs = np.where(np.argmax(rpn_match, axis=1) == 1)[0]
            # Get the predicted anchors for the positive anchors
            #predicted_anchors = ut.shift_bboxes(anchors[positive_idxs], rpn_bbox[positive_idxs])
######################attention###############
            print(positive_idxs)
            print(len(anchors[positive_idxs]))
            predicted_anchors = ut.shift_bboxes(anchors[positive_idxs], rpn_bbox[positive_idxs])
            # Sort predicted class by strength of prediction
            top_n=1
            argsort = np.flip(np.argsort(rpn_match[positive_idxs, 1]), axis=0)
            sorted_anchors = predicted_anchors[argsort]
            sorted_anchors = sorted_anchors[:min(top_n, sorted_anchors.shape[0])]

           # axes[4].imshow(image_gt[:,:,0],cmap='gray')
            
            # Loop through predictions
            
            for a1 in sorted_anchors:
                rect1 = patches.Rectangle((a1[1], a1[0]), a1[3] - a1[1], a1[2] - a1[0], linewidth=1.5, edgecolor='c', facecolor='none',
                                         linestyle='-')
                                         
                rect2 = patches.Rectangle((a1[1]-30, a1[0]-30), a1[3] - a1[1]+60, a1[2] - a1[0]+60, linewidth=1.5, edgecolor='c', facecolor='none',
                                         linestyle='-')                
                axes[1][1].add_patch(rect1)
                
            RPN_name=str(idx)+'.png'
            
            #print(int(a1[1]-30),int(a1[0]-30),int(a1[3]+30),int(a1[2]+30),'999999')
            
            val_image=VALIDATION_img[idx,:,:,0]
            try:
              imageio.imsave(os.path.join(RPN_result_dir,RPN_name),val_image[int(a1[0]-30):int(a1[2]+30),int(a1[1]-30):int(a1[3]+30)])
            except:
              continue
            axes[1][0].imshow(coarse_img[:,:,0],'gray')
            axes[1][0].set_title("coarse_result",fontsize=10) 
            
            coarse_img1 = np.array(coarse_img[:,:,0],np.uint8)
            x, y, w, h = cv2.boundingRect(coarse_img1)
            xi=x-30
            yi=y-30
            wi=w+60
            hi=h+60 
            coarse_box=patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor='m', facecolor='none', linestyle='-')   
            coarse_box1=patches.Rectangle((x-30, y-30), w+60, h+60, linewidth=1.5, edgecolor='m', facecolor='none', linestyle='-') 
            #axes[4].imshow(np.uint8(image_gt + config.MEAN_PIXEL),cmap='gray')
            #axes[4].imshow(image_gt[:,:,0],'gray')  
            axes[1][1].add_patch(coarse_box)
            axes[1][1].legend([rect,rect1,coarse_box],['gt_bbox','RPN_bbox','coarse_bbox'])
            axes[1][2].imshow(val_image,'gray')
            axes[1][2].add_patch(rectt)
            try:
             axes[1][2].add_patch(rect2)
             axes[1][2].add_patch(coarse_box1)              
             axes[1][2].legend([rectt,rect2,coarse_box1],['gt_bbox','RPN bbox','coarse_bbox'])             
             axes[1][2].set_title("input of Unet",fontsize=10)
             
            except:
              print(rectt,rect2,coarse_box1,'9999999988888')
              print(idx,7777777)
              continue 
            coarse_image=VALIDATION_img[idx,:,:,0]
            #imageio.imsave(os.path.join(coarse_img_dir,RPN_name),coarse_image[int(yi):int(yi+hi),int(xi):int(xi+wi)])   
            iou_RPN.append(iou((a1[1]-30,a1[0]-30,a1[3]+30,a1[2]+30),(a[1]-30,a[0]-30,a[3]+30,a[2]+30)))
            iou_coarse.append(iou((xi,yi,xi+wi,yi+hi),(a[1]-30,a[0]-30,a[3]+30,a[2]+30)))
            
            recall_RPN.append(recall((a1[1]-30,a1[0]-30,a1[3]+30,a1[2]+30),(a[1]-30,a[0]-30,a[3]+30,a[2]+30)))
            recall_coarse.append(recall((xi,yi,xi+wi,yi+hi),(a[1]-30,a[0]-30,a[3]+30,a[2]+30)))        

            acc_RPN.append(accuracy((a1[1]-30,a1[0]-30,a1[3]+30,a1[2]+30),(a[1]-30,a[0]-30,a[3]+30,a[2]+30)))
            acc_coarse.append(accuracy((xi,yi,xi+wi,yi+hi),(a[1]-30,a[0]-30,a[3]+30,a[2]+30)))
                
            plt.savefig(os.path.join(figure_dir,RPN_name))
            #plt.pause(5) # show for 15s
            #plt.close()
            
    print('coarse_iou:',statistics.mean(iou_coarse)) 
    print('RPN_iou:',statistics.mean(iou_RPN))
    print('coarse_recall:',statistics.mean(recall_coarse)) 
    print('RPN_recall:',statistics.mean(recall_RPN))
    print('coarse_acc:',statistics.mean(acc_coarse)) 
    print('RPN_acc:',statistics.mean(acc_RPN))
    f = open("b.txt", 'w+')      
    print('RPN_iou:',statistics.mean(iou_RPN),'max_RPN_iou:',max(iou_RPN),'min_RPN_iou:',min(iou_RPN),'RPN_recall:',statistics.mean(recall_RPN),'max_RPN_recall:',max(recall_RPN),'min_RPN_recall:',min(recall_RPN),'RPN_acc:',statistics.mean(acc_RPN),'max_RPN_iou:',max(acc_RPN),'min_RPN_iou:',min(acc_RPN),file=f)
    print('coarse_iou:',statistics.mean(iou_coarse),'max_coarse_iou:',max(iou_coarse),'min_coarse_iou:',min(iou_coarse),'coarse_recall:',statistics.mean(recall_coarse),'max_coarse_recall:',max(recall_coarse),'min_coarse_recall:',min(recall_coarse),'coarse_acc:',statistics.mean(acc_coarse),'max_coarse_acc:',max(acc_coarse),'min_coarse_acc:',min(acc_coarse),file=f)    
        # Visualize the predictions
        #ut.visualize_rpn_predictions(np.uint8(image_gt + config.MEAN_PIXEL), rpn_match, rpn_bbox, anchors, top_n=3)


if __name__ == '__main__':
    
    main()
