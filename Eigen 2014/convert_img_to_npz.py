import tensorflow as tf
import sys, glob, imageio
import numpy as np

def convert_predictions():

    # For all png images in arg 1 folder
    for i, image in enumerate(glob.glob('%s/*.png' % sys.argv[1])):
        print(image)
        # Read and resize image
        prediction = imageio.imread(image)
        prediction = tf.expand_dims(prediction, 2)
        print(prediction.shape)
        prediction = tf.image.resize_images(prediction,[480,640])
        print(prediction.shape)

        # If this is the 1st image, create predictions
        # and add dimension for image number
        if i == 0:
            predictions = prediction
            print(predictions.shape)
        # Else append the current image to the predictions array
        else:
            predictions = np.append(predictions, prediction, axis = 2)
            print(predictions.shape)

    # Save the entire array as a .npz
    np.savez('predictions.npz', depths=predictions)

if __name__ == '__main__':
    # Check that argument has been provided
    if len(sys.argv) != 2:
    	print("Please run:\n\tpython convert_npz_to_img.py <predictions_path>")
    	exit()
    convert_predictions()
