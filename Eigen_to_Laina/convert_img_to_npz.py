import tensorflow as tf
import sys, glob, imageio
import numpy as np

def convert_predictions():

    sess = tf.Session()
    with sess.as_default():
        # For all png images in arg 1 folder
        for i, image in enumerate(sorted(glob.glob('%s/*[0-9].png' % sys.argv[1]))):
            print(image)
            # Read and resize image (currently in tensor form)
            prediction = imageio.imread(image)
            prediction = tf.expand_dims(prediction, 2)
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
        predictions = predictions.eval()
        # Save the entire array as a .npz
        np.savez('predictions.npz', depths=predictions)
        print('images saved to predictions.npz')

if __name__ == '__main__':
    # Check that argument has been provided
    if len(sys.argv) != 2:
    	print("Please run:\n\tpython convert_npz_to_img.py <predictions_path>")
    	exit()
    convert_predictions()
