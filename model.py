import tensorflow as tf


def encode(images, reuse=False, trainable=True)
    conv1 = conv2d('conv1',images,[7,7,3,64],[64],[1,2,2,1], padding='VALID', reuse=reuse, trainable=trainable)
    max_pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1],paddinf='VALID', name='pool1')
    resblock1 = 
    return conv2_output

def decode(conv2_output, res_block13_output, res_block7_output, res_block3_output):
    upproject1 = 