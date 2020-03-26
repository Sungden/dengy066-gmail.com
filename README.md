# dengy066-gmail.com
pancreastitis segmentation
coarse-to-fine approach for pancreastitis segmentation
coarse segmentation is a FCN-8 model.fine segmentation is a 2D U-Net model. Using RPN to localize the pancrestitis region on feature map produced by coarse segmentation and send it to the U-Net.
