import tensorflow as tf
import tflearn

# Define type 1 residual block
def res1(block_number, incoming):
    block = 'res_block%d_' % block_number
    # First convolution that uses res1 inputs
    net = tflearn.layers.conv.conv_2d (
        incoming, nb_filter=1, filter_size=1, strides=1,padding='same',
        activation='relu', bias=True, weights_init='truncated_normal',
        bias_init='zeros', regularizer=None, weight_decay=0.001, trainable=True,
        restore=True, reuse=False, scope=None, name='%sConv2D_1' % block)
    net = normalisation(net)
    # Second convolution
    net = tflearn.layers.conv.conv_2d (
        net, nb_filter=1, filter_size=3, strides=1, padding='same',
        activation='relu', bias=True, weights_init='truncated_normal',
        bias_init='zeros', regularizer=None, weight_decay=0.001, trainable=True,
        restore=True, reuse=False, scope=None, name='%sConv2D_2' % block)
    net = normalisation(net)
    # Third connvolution
    net = tflearn.layers.conv.conv_2d (
        net, nb_filter=1, filter_size=1, strides=1, padding='same',
        activation='linear', bias=True, weights_init='truncated_normal',
        bias_init='zeros', regularizer=None, weight_decay=0.001, trainable=True,
        restore=True, reuse=False, scope=None, name='%sConv2D_3' % block)
    net = normalisation(net)
    # Add the raw input and the third convolution output
    net += incoming
    # Pass net through a ReLU activation function
    net = tflearn.activations.relu (net)

    return net

# Define type 2 residual block
def res2(block_number, incoming, stride_size):
    block = 'res_block%d' % block_number
    # First convolution that uses res2 inputs
    net = tflearn.layers.conv.conv_2d (
        incoming, nb_filter=1, filter_size=1, strides=stride_size, padding='same',
        activation='relu', bias=True, weights_init='truncated_normal',
        bias_init='zeros', regularizer=None, weight_decay=0.001, trainable=True,
        restore=True, reuse=False, scope=None, name='%sConv2D_1' % block)
    net = normalisation(net)
    # Second convolution
    net = tflearn.layers.conv.conv_2d (
        net, nb_filter=1, filter_size=3, strides=1, padding='same',
        activation='relu', bias=True, weights_init='truncated_normal',
        bias_init='zeros', regularizer=None, weight_decay=0.001, trainable=True,
        restore=True, reuse=False, scope=None, name='%sConv2D_2' % block)
    net = normalisation(net)
    # Third convolution
    net = tflearn.layers.conv.conv_2d (
        net, nb_filter=1, filter_size=1, strides=1, padding='same',
        activation='linear', bias=True, weights_init='truncated_normal',
        bias_init='zeros', regularizer=None, weight_decay=0.001, trainable=True,
        restore=True, reuse=False, scope=None, name='%sConv2D_3' % block)
    net = normalisation(net)
    # Residual convolution that uses res2 unputs
    res = tflearn.layers.conv.conv_2d (
        incoming=incoming, nb_filter=1, filter_size=1, strides=stride_size, padding='same',
        activation='linear', bias=True, weights_init='truncated_normal',
        bias_init='zeros', regularizer=None, weight_decay=0.001, trainable=True,
        restore=True, reuse=False, scope=None, name='%sConv2D_Residual' % block)
    res = normalisation(res)
    # Add the residual convolution and the third convolution outputs
    net += res
    # Pass net through a ReLU activation function
    net = tflearn.activations.relu (net)

    return net

# Define up project residual block
def resup(block_number, incoming):
    block = 'upproject%d' % block_number
    # Upsample by a factor of 2
    net = tflearn.layers.conv.upsample_2d (
        incoming, kernel_size=2, scope=None, name='%sUpSample2D' % block)
    # Resudual convolution using upsample as input
    res = tflearn.layers.conv.conv_2d (
        net, nb_filter=1, filter_size=5, strides=1, padding='same',
        activation='linear', bias=True, weights_init='truncated_normal',
        bias_init='zeros', regularizer=None, weight_decay=0.001, trainable=True,
        restore=True, reuse=False, scope=None, name='%sConv2D_Residual' % block)
    res = normalisation(res)
    # First convolution using upsample as input
    net = tflearn.layers.conv.conv_2d (
        net, nb_filter=1, filter_size=5, strides=1, padding='same',
        activation='relu', bias=True, weights_init='truncated_normal',
        bias_init='zeros', regularizer=None, weight_decay=0.001, trainable=True,
        restore=True, reuse=False, scope=None, name='%sConv2D_1' % block)
    net = normalisation(net)
    # Second convolution
    net = tflearn.layers.conv.conv_2d (
        net, nb_filter=1, filter_size=3, strides=1, padding='same',
        activation='linear', bias=True, weights_init='truncated_normal',
        bias_init='zeros', regularizer=None, weight_decay=0.001, trainable=True,
        restore=True, reuse=False, scope=None, name='%sConv2D_2' % block)
    net = normalisation(res)
    # Add the residual convolution and the second convolution outputs
    net += res
    # Pass net through a ReLU activation function
    net = tflearn.activations.relu (net)

    return net

def normalisation(incoming):
    net = tflearn.layers.normalization.batch_normalization (incoming,
                                                            beta=0.0,
                                                            gamma=1.0,
                                                            epsilon=1e-05,
                                                            decay=0.9,
                                                            stddev=0.002,
                                                            trainable=True,
                                                            restore=True,
                                                            reuse=False,
                                                            scope=None,
                                                            name='BatchNormalization')
    return net
