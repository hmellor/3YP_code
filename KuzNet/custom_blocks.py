import tensorflow as tf
import tflearn

# Define type 1 residual block
def res1(incoming):
    # First convolution that uses res1 inputs
    net = tflearn.layers.conv.conv_2d (
        incoming, nb_filter=1, filter_size=1, strides=1,padding='same',
        activation='relu', bias=True, weights_init='truncated_normal',
        bias_init='zeros', regularizer=None, weight_decay=0.001, trainable=True,
        restore=True, reuse=False, scope=None, name='Type1_Conv2D_1')
    # Second convolution
    net = tflearn.layers.conv.conv_2d (
        net, nb_filter=1, filter_size=3, strides=1, padding='same',
        activation='relu', bias=True, weights_init='truncated_normal',
        bias_init='zeros', regularizer=None, weight_decay=0.001, trainable=True,
        restore=True, reuse=False, scope=None, name='Type1_Conv2D_2')
    # Third connvolution
    net = tflearn.layers.conv.conv_2d (
        net, nb_filter=1, filter_size=1, strides=1, padding='same',
        activation='linear', bias=True, weights_init='truncated_normal',
        bias_init='zeros', regularizer=None, weight_decay=0.001, trainable=True,
        restore=True, reuse=False, scope=None, name='Type1_Conv2D_3')
    # Add the raw input and the third convolution output
    net += incoming
    # Pass net through a ReLU activation function
    net = tflearn.activations.relu (net)

    return net

# Define type 2 residual block
def res2(incoming, stride_size):
    # First convolution that uses res2 inputs
    net = tflearn.layers.conv.conv_2d (
        incoming, nb_filter=1, filter_size=1, strides=stride_size, padding='same',
        activation='relu', bias=True, weights_init='truncated_normal',
        bias_init='zeros', regularizer=None, weight_decay=0.001, trainable=True,
        restore=True, reuse=False, scope=None, name='Type2_Conv2D_1')
    # Second convolution
    net = tflearn.layers.conv.conv_2d (
        net, nb_filter=1, filter_size=3, strides=1, padding='same',
        activation='relu', bias=True, weights_init='truncated_normal',
        bias_init='zeros', regularizer=None, weight_decay=0.001, trainable=True,
        restore=True, reuse=False, scope=None, name='Type2_Conv2D_2')
    # Third convolution
    net = tflearn.layers.conv.conv_2d (
        net, nb_filter=1, filter_size=1, strides=1, padding='same',
        activation='linear', bias=True, weights_init='truncated_normal',
        bias_init='zeros', regularizer=None, weight_decay=0.001, trainable=True,
        restore=True, reuse=False, scope=None, name='Type2_Conv2D_3')
    # Residual convolution that uses res2 unputs
    res = tflearn.layers.conv.conv_2d (
        incoming=incoming, nb_filter=1, filter_size=1, strides=stride_size, padding='same',
        activation='linar', bias=True, weights_init='truncated_normal',
        bias_init='zeros', regularizer=None, weight_decay=0.001, trainable=True,
        restore=True, reuse=False, scope=None, name='Type2_Conv2D_res')
    # Add the residual convolution and the third convolution outputs
    net += res
    # Pass net through a ReLU activation function
    net = tflearn.activations.relu (net)

    return net

# Define up project residual block
def resup(incoming):
    # Upsample by a factor of 2
    net = tflearn.layers.conv.upsample_2d (
        incoming, kernel_size=2, name='resup UpSample2D')
    # Resudual convolution using upsample as input
    res = tflearn.layers.conv.conv_2d (
        net, nb_filter=1, filter_size=5, strides=1, padding='same',
        activation='linear', bias=True, weights_init='truncated_normal',
        bias_init='zeros', regularizer=None, weight_decay=0.001, trainable=True,
        restore=True, reuse=False, scope=None, name='resup_Conv2D_res')
    # First convolution using upsample as input
    net = tflearn.layers.conv.conv_2d (
        net, nb_filter=1, filter_size=5, strides=1, padding='same',
        activation='relu', bias=True, weights_init='truncated_normal',
        bias_init='zeros', regularizer=None, weight_decay=0.001, trainable=True,
        restore=True, reuse=False, scope=None, name='resup_Conv2D_11')
    # Second convolution
    net = tflearn.layers.conv.conv_2d (
        net, nb_filter=1, filter_size=3, strides=1, padding='same',
        activation='linear', bias=True, weights_init='truncated_normal',
        bias_init='zeros', regularizer=None, weight_decay=0.001, trainable=True,
        restore=True, reuse=False, scope=None, name='resup_Conv2D_2')
    # Add the residual convolution and the second convolution outputs
    net += res
    # Pass net through a ReLU activation function
    net = tflearn.activations.relu (net)

    return net
