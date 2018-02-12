import sys
import numpy as np
import tensorflow as tf
from PIL import Image
import glob
import model

def predict(model_path, input_directory, output_directory):
    
    # Default input size
    height = 228
    width = 304
    channels = 3
    batch_size = 1

    with tf.Session() as sess:
        # Use to load from ckpt file
        saver = tf.train.import_meta_graph('%s.meta' % model_path)
        saver.restore(sess, model_path)

        images = []
        for filename in glob.glob('%s/*.jpg' % input_directory): #assuming gif
            im = tf.image.decode_jpeg(filename, channels=3)
            im = tf.cast(im, tf.float32)
            im = tf.image.resize_images(im, (height, width))
            images.append(im)
        for i, (image) in enumerate(zip(images)):
            keep_conv = tf.placeholder(tf.float32)
            keep_hidden = tf.placeholder(tf.float32)
            coarse = model.inference(image, keep_conv, trainable=False)
            depth = model.inference_refine(image, coarse, keep_conv, keep_hidden)
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
