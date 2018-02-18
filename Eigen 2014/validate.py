import sys,  os
import numpy as np
import tensorflow as tf
from PIL import Image
import glob
import model 
        
def predict(model_path, input_directory, output_directory):

    with tf.Session() as sess:
    # Default input size
        height = 228
        width = 304
        channels = 3
        batch_size = 1
    
        filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(input_directory+"/*.jpg"))
        print(filename_queue)       
        image_reader = tf.WholeFileReader()
        _, image_file = image_reader.read(filename_queue)
        images = tf.image.decode_jpeg(image_file)
    
    
    
    
        # Read image
        #imageslist = []
        #for filename in glob.glob('%s/*.jpg' % input_directory): #assuming gif, jpgs are RGB pictures
        #    print(filename)
        #    
        #    jpg = tf.read_file(filename)        
        #    im = tf.image.decode_jpeg(jpg, channels=3) # convert jpg into uint8 
        #    im = tf.cast(im, tf.float32)
        #    print('\n** Loaded ' + str(im.shape()) + ' image. ** \n')  
        #    im = tf.image.resize_images(im, (height, width))
        #    imageslist.append(im)
    
        #images = tf.stack(imageslist)
        
        #images = tf.transpose(images, [1,2,3,0] ) # sort image stack (tensor) into proper dimensions, height, wdith, channels, image_id
        print('\n** Loaded ' + str(images.shape()) + ' images. ** \n')
    
    
        # Create a placeholder for the input image
        input_node = tf.placeholder(tf.float32)
        keep_conv = input_node
        keep_hidden = input_node
           
        # Load the converted parameters
        print('\n** Loading the model **\n')
    
            # Use to load from ckpt file
        saver = tf.train.import_meta_graph('%s.meta' % model_path)
        saver.restore(sess, model_path)
    
        # Evalute the network for the given image
        print('\n ** shape of iages inputted ' + str(images.shape()) + ' ** \n')
            
    
    # run image through coarse and refine models
        coarse = model.inference(images, keep_conv, trainable =False )
        print('\n ** size of depth tensor out ' + str(coarse.shape()) + ' ** \n')
        depth = model.inference_refine(images, coarse, keep_conv,keep_hidden)
            
            
        print('\n ** size of depth tensor out ' + str(depth.shape()) + ' ** \n')
            
        #depth = np.transpose(depth, [2, 0, 1] )
        if np.max(depth) != 0:  
            ra_depth = (depth/np.max(depth))*255.0
        else:
            ra_depth = depth*255.0
                
        # see size of tensor
        print('\n ** size of ra_depth tensor out ' + str(ra_depth.shape()) + ' ** \n')
            
        depth_numpy = ra_depth.eval() # convert tensor to numpy array to loop through
        for i,depth_image in enumerate(depth_numpy[0]):
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
