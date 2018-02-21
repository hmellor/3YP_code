import tflearn

import numpy as np
import models



def main():
    # Check for correct number of input arguments
    
    if len(sys.argv) != 2:
    	print("Please run:\n\tpython kuznet.py <mode> <input_directory>")
    	exit()
    
    mode = sys.argv[1]
    input_directory= sys.argv[2]
    
    #load images from input npz file
    f = np.load(input_directory)
    images = f['images']
    
    #rearrange into proper columns
    images = np.transpose(images, [3,0,1,2])
    images = tf.convert_to_tensor(images, dtype=tf.float32)
    
    print('\n images loaded successfully \n')
    
    if mode == 'train':
        train = true
      
    
    
    
    
    # Predict the image
    #predict(model, input_directory)
    exit()
