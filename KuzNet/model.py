import customblocks
import tflearn
import numpy as np
import tensorflow as tf


# Real-time data augmentation 
img_aug = tflearn.ImageAugmentation()
#add random left and right flips
img_aug.add_random_flip_leftright()


# Building Residual Network

# Specify the input shape to be [number of images, height, width, number of channels]
net = tflearn.input_data(shape=[None, 640, 480, 3],
                         data_augmentation=img_aug)

#Main model section 1
#first layer is a 2d convolution of size 7 and stride 2
net = tflearn.conv_2d(net, 7, 3, strides=2)

# second layer is a maxpool layer of size 3 and stride 2
#unsure if we need padding and what exactly is batch normalisation
net = tflearn.layers.conv.max_pool_2d (net, 3, strides=2)

#Main model section 2
net = res1(net,2) #type 1, stride 2     resblock1
net = res1(net,1) #type 1, stride 1
net = res1(net,1) #type 1, stride 1
resblock3 = net
net = res2(net,2) #type 2, stride 2     resblock 4

#Main model section 3
net = res1(net,1) #type 1, stride 1
net = res1(net,1) #type 1, stride 1
net = res1(net,1) #type 1, stride 1
resblock7 = net
net = res2(net,2) #type 2, stride 2       resblock 8


#Main model section 4
net = res1(net,1) #type 1, stride 1       resblock 9 
net = res1(net,1) #type 1, stride 1
net = res1(net,1) #type 1, stride 1
net = res1(net,1) #type 1, stride 1
net = res1(net,1) #type 1, stride 1       resblock 13
resblock13 = net
net = res2(net,2) #type 2, stride 2       resblock 14


#Main model section 5
net = res1(net,1) #type 1, stride 1       resblock 15 
net = res1(net,1) #type 1, stride 1


#Main model section 6
net = resup(net,1) #type 1, stride 1       upproject1 

