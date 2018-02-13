import tensorflow as tf
import sys, glob, imageio
import numpy as np

def convert_predictions():
    path = sys.argv[1]
    predictions = []

    for i, image in enumerate(glob.glob('%s/*.png' % path)):
        print(image)
        prediction = imageio.imread(image)
        prediction = tf.expand_dims(prediction, 2)
        print(prediction.shape)
        prediction = tf.image.resize_images(prediction,[480,640])
        print(prediction.shape)
        prediction = tf.squeeze(prediction,2)
        print(prediction.shape)
        predictions[:,:,i] = prediction
        print(predictions.shape)
    np.savez('predictions.npz', depths=predictions)

if __name__ == '__main__':
    if len(sys.argv) != 2:
    	print("Please run:\n\tpython convert_npz_to_img.py <predictions_path>")
    	exit()
    path = sys.argv[1]
    convert_predictions()
