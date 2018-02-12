import sys
import numpy as np
import tensorflow as tf
from PIL import Image
import glob

def predict(model_path, input_directory, output_directory):

    # Default input size
    height = 228
    width = 304
    channels = 3
    batch_size = 1

    # Read image
    images = []
    for filename in glob.glob('%s/*.jpg' % input_directory): #assuming gif
        im = tf.image.decode_jpeg(filename, channels=3)
        im = tf.cast(im, tf.float32)
        im = tf.image.resize_images(im, (height, width))
        images.append(im)

    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(height, width, channels))

    with tf.Session() as sess:

        # Load the converted parameters
        print('Loading the model')

        # Use to load from ckpt file
        saver = tf.train.import_meta_graph('%s.meta' % model_path)
        saver.restore(sess, model_path)

        # Evalute the network for the given image

        print("output predict into %s" % output_directory)
        for i, (image) in enumerate(zip(images)):
            depth = sess.run(logits, feed_dict={input_node: image})
            depth = depth.transpose(2, 0, 1)
            if np.max(depth) != 0:
                ra_depth = (depth/np.max(depth))*255.0
            else:
                ra_depth = depth*255.0
            depth_pil = Image.fromarray(np.uint8(ra_depth[0]), mode="L")
            depth_name = "%s/%05d.png" % (output_directory, i)
            depth_pil.save(depth_name)


def main():
    # Check for correct number of input arguments
    if len(sys.argv) != 4:
    	print("Please run:\n\tpython validate.py <model_path> <input_directory> <output_directory>")
    	exit()
    model_path = sys.argv[1]
    input_directory = sys.argv[2]
    output_directory = sys.argv[3]
    # Predict the image
    predict(model_path, input_directory, output_directory)

if __name__ == '__main__':
    main()
