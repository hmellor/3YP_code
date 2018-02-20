import sys,  os
import numpy as np
import tensorflow as tf
from PIL import Image
import glob
import model

def predict(model_path, input_directory, output_directory):

    sess = tf.Session()
    with sess.as_default():

        # Default input size
        height = 228
        width = 304
        channels = 3
        batch_size = 1

        print("load dataset: %s" % (input_directory))
        f = np.load(input_directory)
        images_np = f['images']
        print(images_np.dtype)
        f32_images_np = images_np.astype('float32')
        print(f32_images_np.dtype)


        f32_images_np = tf.transpose(f32_images_np, [3,0,1,2] ) # sort image stack (tensor) into proper dimensions, height, wdith, channels, image_id
        
        f32_images_np = f32_images_np[0]
        
        f32_images_np = tf.image.resize_images(f32_images_np, (height, width))
        print('\n** Loaded ' + str(f32_images_np.shape) + ' images. ** \n')
        images = tf.convert_to_tensor(f32_images_np, dtype=tf.float16)
        images = tf.expand_dims(images,0)

        print('\n ** ' + str(tf.shape(images))+' ** \n')

        # Create a placeholder for the input image
        input_node = tf.placeholder(dtype=tf.float32)
        keep_conv = input_node
        keep_hidden = input_node

        # Load the converted parameters
        print('\n** Loading the model **\n')

        # Use to load from ckpt file
        saver = tf.train.import_meta_graph('%s.meta' % model_path)
        saver.restore(sess, model_path)


        # run image through coarse and refine models
        coarse = model.inference(images, keep_conv, trainable =False )
        depth = model.inference_refine(images, coarse, keep_conv,keep_hidden)
        print('\n ** size of ra_depth tensor out ' + str(depth.shape) + ' **')
        print('\n ** size of depth tensor out ' + str(tf.shape(depth)) + ' ** \n')

        #depth = np.transpose(depth, [2, 0, 1] )
        if np.max(depth) != 0:
            ra_depth = (depth/np.max(depth))*255.0
        else:
            ra_depth = depth*255.0


        tf.to_int32(ra_depth)
        # see size of tensor
        print('\n ** size of ra_depth tensor out ' + str(type(ra_depth)) + ' **')
        print('\n ** size of ra_depth tensor out ' + str(ra_depth.get_shape()) + ' ** \n')
        
        
        sess = tf.Session()
        with sess.as_default():
            print(ra_depth.eval())
        #print(type(tf.sess.run(ra_depth)))
        
        
        #depth_numpy = tf.Session().run(ra_depth) # convert tensor to numpy array to loop through
        for i,depth_image in enumerate(ra_depth[0]):
            # using output_depth_images method
            output_directory = '..data/val_datasets/val_output/' # TEMPORARY - until i find way to pass through
            print('\n** saving ' + str(depth_image.get_shape()) + ' size image. ** \n')

            depth_name = os.path.join("data","datasets_%s" % (sys.argv[1]), "%05d.png" % (i))
            print(depth_name)
            #depth_pil.save(depth_name)


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
    exit()

if __name__ == '__main__':
    main()
