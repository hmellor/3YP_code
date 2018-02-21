import tflearn

def res1(incoming, stride):
    net = tflearn.layers.conv.conv_2d (
        incoming, nb_filter=1, filter_size=1, stride, padding='same',
        activation='relu', bias=True, weights_init='uniform_scaling', bias_init='zeros', regularizer=None, weight_decay=0.001, trainable=True, restore=True, reuse=False, scope=None, name='Conv2D')

    net =tflearn.layers.conv.conv_2d (
        net1, nb_filter=1, filter_size=3, strides=1, padding='same',
        activation='relu', bias=True, weights_init='uniform_scaling', bias_init='zeros', regularizer=None, weight_decay=0.001, trainable=True, restore=True, reuse=False, scope=None, name='Conv2D')

    net =tflearn.layers.conv.conv_2d (
        net1, nb_filter=1, filter_size=1, strides=1, padding='same',
        activation='linear', bias=True, weights_init='uniform_scaling', bias_init='zeros', regularizer=None, weight_decay=0.001, trainable=True, restore=True, reuse=False, scope=None, name='Conv2D')

    net += incoming

    net = tflearn.activations.relu (net)

    return net

def res2(net, s):

    return net

def resup(net, s):

    return net
