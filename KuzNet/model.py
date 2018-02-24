from custom_blocks import res1,res2,resup,normalisation
import numpy as np
import tensorflow as tf
import tflearn

wd = 0.00004

def model_network():
    # real time data processing
    # img_prep = tflearn.ImagePreprocessing()
    # img_prep.add_featurewise_zero_center(per_channel=True)
    #
    # # Real-time data augmentation
    # img_aug = tflearn.ImageAugmentation()
    # #add random left and right flips
    # img_aug.add_random_flip_leftright()

    # Building Residual Network

    # Specify the input shape to be [number of images, height, width, number of channels]
    net = tflearn.input_data(shape=[None, 480, 640, 3])
    #    data_preprocessing=img_prep, data_augmentation=img_aug)

    #Main model section 1
    #first layer is a 2d convolution of size 7 and stride 2 and 3 channels
    net = tflearn.layers.conv_2d(net, 1, 7, strides=2, regularizer='L2',
    weight_decay=wd, name='conv1')
    net = normalisation(net)
    # second layer is a maxpool layer of size 3 and stride 2
    #unsure if we need padding and what exactly is batch normalisation
    net = tflearn.layers.conv.max_pool_2d (net, 3, strides=2, name='maxpool1')
    #net = normalisation(net)

    #Main model section 2
    net = res2(1,net,1) #type 2, stride 1     resblock1
    net = res1(2,net) #type 1, stride 1
    resblock3 = res1(3,net) #type 1, stride 1
    net = res2(4,resblock3,2) #type 2, stride 2     resblock 4

    #Main model section 3
    net = res1(5,net) #type 1, stride 1
    net = res1(6,net) #type 1, stride 1
    resblock7 = res1(7,net) #type 1, stride 1
    net = res2(8,resblock7,2) #type 2, stride 2       resblock 8


    #Main model section 4
    net = res1(9,net) #type 1, stride 1       resblock 9
    net = res1(10,net) #type 1, stride 1
    net = res1(11,net) #type 1, stride 1
    net = res1(12,net) #type 1, stride 1
    resblock13 = res1(13,net) #type 1, stride 1       resblock 13
    net = res2(14,resblock13,2) #type 2, stride 2       resblock 14


    #Main model section 5
    net = res1(15,net) #type 1, stride 1       resblock 15
    net = res1(16,net) #type 1, stride 1       resblock 16
    #conv layer is a 2d convolution of size 1, stride 1
    # conv2d syntax tflearn.layers.conv.conv_2d (incoming, nb_filter, filter_size, strides=1)
    net = tflearn.layers.conv_2d(net, 1, 1, strides=1, regularizer='L2',
    weight_decay=wd, name='conv2')
    net = normalisation(net)

    #Main model section 6
    net = resup(1,net) #                    upproject1

    #Main model - remaining up projections
    net = resup(2,net + resblock13)#        upproject2
    net = resup(3,net + resblock7) #        upproject3
    net = resup(4,net + resblock3) #        upproject4
    #net = tflearn.layers.core.dropout(net, 0.5, noise_shape=None, name='Dropout')


    #final conv layer is a 2d convolution of size 3, stride 1
    net = tflearn.layers.conv_2d(net, 1, 3, strides=1, regularizer='L2',
    weight_decay=wd, name='conv3')
    # No normalisation

    #Regression
    adam = tflearn.optimizers.Adam (learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, use_locking=False, name='Adam')
    r2 = tflearn.metrics.R2()
    net = tflearn.layers.estimator.regression (
        net, metric=r2, optimizer=adam, loss='mean_square')

    return net
