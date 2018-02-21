import sys,os
import tflearn
import tensorflow as tf
import numpy as np
import model

def main():
    # Check for correct number of input arguments

    if len(sys.argv) != 2:
    	print("Please run:\n\tpython kuznet.py <mode> <input_directory>")
    	exit()

    mode = sys.argv[1]
    input_directory= sys.argv[2]

    print('\n ** Loading Images from %s ** \n',input_directory)

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


if __name__ == "__main__":
    main()
