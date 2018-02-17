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

    # Read image
    images = np.empty([height,width,channels], dtype=int, order='C')
    print('\n ** initial size of images ' + str(images.get_shape()) + ' ** \n')  # Output image input size
    for filename in glob.glob('%s/*.jpg' % input_directory): #assuming gif, jpgs are RGB pictures
        im = tf.image.decode_jpeg(filename, channels=3) # convert jpg into uint8 tensor
        im = tf.cast(im, tf.float32)
        im = tf.image.resize_images(im, (height, width))
        print('\n ** size of image tensor in ' + str(im.get_shape()) + ' ** \n')  # Output image size
        images = tf.concat([images, im], 3)
        #images.append(im)


    print('\n ** size of whole image tensor in ' + str(images.get_shape()) + ' ** \n')  # Output image input size

    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32)
    keep_conv = input_node
    keep_hidden = input_node
    with tf.Session() as sess:

        # Load the converted parameters
        print('\n** Loading the model **\n')

        # Use to load from ckpt file
        saver = tf.train.import_meta_graph('%s.meta' % model_path)
        saver.restore(sess, model_path)


        # Evalute the network for the given image

        print("\n ** output predict into %s **\n" % output_directory)
        # run image through coarse and refine models
        coarse = model.inference(images, keep_conv, trainable=False)
        depth = model.inference_refine(images, coarse, keep_conv,keep_hidden)
            
        # see size of tensor
        print('\n ** size of image out ' + str(depth.get_shape()) + ' ** \n')

        depth = np.transpose(depth, [2, 0, 1] )
        if np.max(depth) != 0:
            ra_depth = (depth/np.max(depth))*255.0
        else:
            ra_depth = depth*255.0
        depth_pil = Image.fromarray(np.uint8(ra_depth[0]), mode="L")
        
        #depth_name = "%s/%05d.png" % (output_directory, i)
        #print(depth_name)
        #depth_pil.save(depth_name)
        print('\n ** Got Through  ** \n')

def main():
    # Check for correct number of input arguments
    # Model_path is a .cpkt file
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
