#encoding: utf-8

from datetime import datetime
from tensorflow.python.platform import gfile
import numpy as np
import tensorflow as tf
import model
import train_operation as op
import sys, os

# if there are 1000 training examples and batch size is 10, it will
# take 100 iterations to complete 1 epoch
ITERATIONS = 100
MAX_EPOCHS = 100
BATCH_SIZE = 0
LOG_DEVICE_PLACEMENT = False
CUDA_VISIBLE_DEVICES=0

REFINE_TRAIN = False
FINE_TUNE = True

IMAGE_HEIGHT = 228
IMAGE_WIDTH = 304
TARGET_HEIGHT = 55
TARGET_WIDTH = 74

def eigen(data_path):
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        if sys.argv[1] == 'train':
            BATCH_SIZE = 10
        else:
            BATCH_SIZE = 1
        # Load the data from the .npz file
        images, depths, invalid_depths, NUMBER_OF_IMAGES = load_data(data_path)

        print('\nImage array shape: %s' % str(images.shape))
        print('\nDepth array shape: %s' % str(depths.shape))
        print('\nNumber of images: %d' % NUMBER_OF_IMAGES)
        print('\nBatch size: %d' % BATCH_SIZE)
        print('\n')

        keep_conv = tf.placeholder(tf.float32)
        keep_hidden = tf.placeholder(tf.float32)

        if REFINE_TRAIN:
            print("refine train.")
            coarse = model.inference(images, keep_conv, trainable=False)
            logits = model.inference_refine(images, coarse, keep_conv, keep_hidden)
        else:
            print("coarse train.")
            logits = model.inference(images, keep_conv, keep_hidden)

        loss = model.loss(logits, depths, invalid_depths)
        train_op = op.train(loss, global_step, BATCH_SIZE)
        init_op = tf.global_variables_initializer()

        # Session
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=LOG_DEVICE_PLACEMENT))
        sess.run(init_op)

        # parameters
        coarse_params = {}
        refine_params = {}

        if REFINE_TRAIN:
            for variable in tf.all_variables():
                variable_name = variable.name
                print("parameter: %s" % (variable_name))
                if variable_name.find("/") < 0 or variable_name.count("/") != 1:
                    continue
                if variable_name.find('coarse') >= 0:
                    coarse_params[variable_name] = variable
                print("parameter: %s" %(variable_name))
                if variable_name.find('fine') >= 0:
                    refine_params[variable_name] = variable
        else:
            for variable in tf.trainable_variables():
                variable_name = variable.name
                print("parameter: %s" %(variable_name))
                if variable_name.find("/") < 0 or variable_name.count("/") != 1:
                    continue
                if variable_name.find('coarse') >= 0:
                    coarse_params[variable_name] = variable
                if variable_name.find('fine') >= 0:
                    refine_params[variable_name] = variable

        # define saver
        print coarse_params
        saver_coarse = tf.train.Saver(coarse_params)
        print refine_params
        saver_refine = tf.train.Saver(refine_params)
        # fine tune
        if FINE_TUNE:
            coarse_ckpt = tf.train.get_checkpoint_state('checkpoints_c')
            if coarse_ckpt and coarse_ckpt.model_checkpoint_path:
                print("Pretrained coarse Model Loading.")
                saver_coarse.restore(sess, coarse_ckpt.model_checkpoint_path)
                print("Pretrained coarse Model Restored.")
            else:
                print("No Pretrained coarse Model.")
            if REFINE_TRAIN:
                refine_ckpt = tf.train.get_checkpoint_state('checkpoints_r')
                if refine_ckpt and refine_ckpt.model_checkpoint_path:
                    print("Pretrained refine Model Loading.")
                    saver_refine.restore(sess, refine_ckpt.model_checkpoint_path)
                    print("Pretrained refine Model Restored.")
                else:
                    print("No Pretrained refine Model.")

        # Train the network
        if sys.argv[1] == 'train':
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            for epoch in xrange(MAX_EPOCHS):
                iteration = 0
                for i in xrange(ITERATIONS):
                    _, loss_value, logits_val, images_val = sess.run([train_op, loss, logits, images], feed_dict={keep_conv: 0.8, keep_hidden: 0.5})
                    if iteration % 10 == 0:
                        print("%s: %d[epoch]: %d[iteration]: train loss %f" % (datetime.now(), epoch, iteration, loss_value))
                        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                    iteration += 1

                if (epoch+1) % 5 == 0:
                    if REFINE_TRAIN:
                        refine_checkpoint_path = 'checkpoints_r/refine_model_parameters.ckpt'
                        saver_refine.save(sess, refine_checkpoint_path, global_step=epoch)
                    else:
                        coarse_checkpoint_path = 'checkpoints_c/coarse_model_parameters.ckpt'
                        saver_coarse.save(sess, coarse_checkpoint_path, global_step=epoch)
            coord.request_stop()
            coord.join(threads)
            sess.close()
        # Test or validate the network
        else:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            for i in xrange(NUMBER_OF_IMAGES):
                _, loss_value, logits_val, images_val = sess.run([train_op, loss, logits, images], feed_dict={keep_conv: 0.8, keep_hidden: 0.5})
                print(logit_val.shape)
                prediction = tf.expand_dims(logits_val, 2)
                prediction = tf.image.resize_images(prediction,[480,640])
                # If this is the 1st image, initialise predictions
                if i == 0:
                    predictions = prediction
                    print(predictions.shape)
                # Else append the current image to the predictions array
                else:
                    predictions = tf.concat([predictions, prediction], 2)
                    print(predictions.shape)

            # Convert tensor to numpy array
            predictions = tf.image.resize_images(predictions,[480,640])
            predictions = predictions.eval()
            # Save the entire array as a .npz
            np.savez('predictions_%s.npz' % sys.argv[1], depths=predictions)
            print('images saved to predictions_%s.npz' % sys.argv[1])

            coord.request_stop()
            coord.join(threads)
            sess.close()

def load_data(data_path):
    print("load dataset: %s" % (data_path))
    # Load image and depth data
    data = np.load(data_path)
    # Extract and manipulate images
    images = data['images']
    images = np.transpose(images, [3, 0, 1, 2])
    images =tf.image.resize_images(images, [IMAGE_HEIGHT, IMAGE_WIDTH])
    images = tf.cast(images, tf.float32)
    # Extract and manipulate depths
    depths = data['depths']
    depths = np.transpose(depths, [2, 0, 1])
    depths = tf.expand_dims(depths, 3)
    depths =tf.image.resize_images(depths, [TARGET_HEIGHT, TARGET_WIDTH])
    depths = tf.cast(depths, tf.float32)
    invalid_depths = tf.sign(depths)
    NUMBER_OF_IMAGES = images.shape[0]
    if sys.argv[1] == 'train':
        NETWORK_MODE = 'True'
    else:
        NETWORK_MODE = 'False'

    queue_input_data   = tf.placeholder(tf.float32, shape=[20, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    queue_input_target = tf.placeholder(tf.float32, shape=[20, TARGET_HEIGHT, TARGET_WIDTH, 1])
    queue_negtv_target = tf.placeholder(tf.float32, shape=[20, TARGET_HEIGHT, TARGET_WIDTH, 1])

    queue = tf.FIFOQueue(
        capacity=50,
        dtypes=[tf.float32, tf.float32, tf.float32],
        shapes=[[MAGE_HEIGHT, IMAGE_WIDTH, 3], [TARGET_HEIGHT, TARGET_WIDTH, 1], [TARGET_HEIGHT, TARGET_WIDTH, 1]]
        )

    enqueue_op = queue.enqueue_many([queue_input_data, queue_input_target, queue_negtv_target])
    dequeue_op = queue.dequeue()

    images, depths , invalid_depths = tf.train.batch(
        dequeue_op,
        batch_size=BATCH_SIZE,
        capacity=50 + 3 * BATCH_SIZE
        )

    # images, depths, invalid_depths = tf.train.batch(
    #     [images, depths, invalid_depths],
    #     batch_size=BATCH_SIZE,
    #     num_threads=4,
    #     capacity= 50 + 3 * BATCH_SIZE,
    #     enqueue_many = True,
    #     )
    return images, depths, invalid_depths, NUMBER_OF_IMAGES

def main(argv=None):
    if len(sys.argv) != 2:
        print("Please run:\n\tpython eigen.py <train/val/test>")
        exit()
    data_path = '%s.npz' % (sys.argv[1])
    if not os.path.isfile(data_path):
    	print('%s file does not exist' % data_path)
    	exit()
    eigen(data_path)


if __name__ == '__main__':
    tf.app.run()
