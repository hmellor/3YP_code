import custom_blocks
import tflearn
import numpy as np
import tensorflow as tf

def model_network():
    # Real-time data augmentation
    img_aug = tflearn.ImageAugmentation()
    #add random left and right flips
    img_aug.add_random_flip_leftright()


    # Building Residual Network

    # Specify the input shape to be [number of images, height, width, number of channels]
    net = tflearn.input_data(shape=[None, 640, 480, 3],
                             data_augmentation=img_aug)

    #Main model section 1
    #first layer is a 2d convolution of size 7 and stride 2 and 3 channels
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
    net = res1(net,1) #type 1, stride 1       resblock 16
    #conv layer is a 2d convolution of size 1, stride 1 and 2048 channels
    net = tflearn.conv_2d(net, 1, 2048, strides=1)

    #Main model section 6
    net = resup(net,1) #type 1, stride 1       upproject1

    #Main model - remaining up projections
    net = resup(net + resblock13) #type 1, stride 1       upproject2
    net = resup(net + resblock7) #type 1, stride 1        upproject3
    net = resup(net + resblock3) #type 1, stride 1        upproject4


    #final conv layer is a 2d convolution of size 3, stride 1 and 64 channels
    net = tflearn.conv_2d(net, 64, 2048, strides=1)

    return net