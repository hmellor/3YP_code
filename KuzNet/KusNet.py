import sys,os
import tflearn
import tensorflow as tf
import numpy as np
import model

def train():


def main():
    # Check for correct number of input arguments
    if sys.argv[1] == 'train' and len(sys.argv) != 3:
        print("\n ** If training please run:\n\tpython kuznet.py train <npz_input_directory> ** \n")
    	exit()
    elif sys.argv[1] == 'val' and len(sys.argv) != 4:
    	print("\n ** If validating please run: \n\tpython kuznet.py val <npz_input_directory> <model_input_directory>** \n")

    mode = sys.argv[1]
    input_directory= sys.argv[2]
    model_directory= sys.argv[3]

    print('\n ** Loading Images from %s ** \n',input_directory)

    #load images from input npz file
    f = np.load(input_directory)
    images = f['images']

    #rearrange into proper columns
    images = np.transpose(images, [3,0,1,2])
    images = tf.convert_to_tensor(images, dtype=tf.float32)

    print('\n **Images loaded successfully ** \n')

    # If train mode, generate weights
    if mode == 'train':
        #
        train()
    else
        # load model values






    # Predict the image
    #predict(model, input_directory)
    exit()


if __name__ == "__main__":
    main()
