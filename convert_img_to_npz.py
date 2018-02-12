#encoding: utf-8
import os, random, sys
import numpy as np
from PIL import Image
from tempfile import TemporaryFile
import glob
import tensorflow as tf
#define image to npz function

def img_to_npz(input_depth_directory, output_depth_directory):
    
    #Read depths
    depths = []
    
    depths0 = []
    depths1 = []
    depths2 = []
    
    
    
    print("load depth dataset: %s" % (input_depth_directory))
    
    #assuming all png images here are predicted depths
    for filename in glob.glob('%s/*.png' % input_directory): 
        im = tf.image.decode_jpeg(filename, channels=1)
        predictions = tf.image.resize_images(im,[55,74])
        predictions = tf.transpose(predictions, [1,2,0])
        depths0 = np.append(depths,predictions, axis=0)
        depths1 = np.append(depths,predictions, axis=1)
        depths2 = np.append(depths,predictions, axis=2)
        
        
  
    #print(depths   
    
    depths0 = np.asarray(depths0)
    print(depths0.shape)
    
    depths1 = np.asarray(depths1)
    print(depths1.shape)
    
    depths2 = np.asarray(depths2)
    print(depths2.shape)
    
    #np.savez('predictions', depths=depths)
    np.savez('predictions', depths=depths)


if __name__ == '__main__':
    if len(sys.argv) != 3:
    	print("Please run:\n\tpython convert_img_to_npz <input_depth_directory> <output_depth_directory>")
    	exit()
    current_directory = os.getcwd()
    input_directory = os.path.join((sys.argv[1]))
    output_directory = os.path.join((sys.argv[2]))
    img_to_npz(input_directory,output_directory)
