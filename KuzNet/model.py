import customblocks
import tflearn
import numpy as np
import tensorflow as tf


#Load data

from tflearn.datasets import cifar10
(X, Y), (testX, testY) = cifar10.load_data()
Y = tflearn.data_utils.to_categorical(Y)
testY = tflearn.data_utils.to_categorical(testY)

# Real-time data augmentation
img_aug = tflearn.ImageAugmentation()
#add random left and right flips
img_aug.add_random_flip_leftright()


# Building Residual Network

# Specify the input shape to be [number of images, height, width, number of channels]
net = tflearn.input_data(shape=[None, 640, 480, 3],
                         data_augmentation=img_aug)

#first layer is a 2d convolution of size 7 and stride 2
net = tflearn.conv_2d(net, 7, 3, strides=2)

# second layer is a maxpool layer of size 3 and stride 2
#unsure if we need padding and what exactly is batch normalisation
net = tflearn.layers.conv.max_pool_2d (net, 3, strides=2)


