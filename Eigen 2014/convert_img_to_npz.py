import tensorflow as tf
import sys, glob, imageio
import numpy as np

def convert_predictions():
    path = sys.argv[1]
    predictions = []

    for i, image in enumerate(glob.glob('%s/*.png' % path)):
        print(image)
        depth = imageio.imread(image)
        depth = tf.expand_dims(depth, 2)
        print(depth.shape)
        prediction = tf.image.resize_images(depth,[480,640])
        print(depth.shape)
        prediction = tf.squeeze(prediction,3)
        print(depth.shape)
        prediction = tf.transpose(prediction, [1,2,0])
        print(depth.shape)
        predictions[:,:,i] = prediction
        print(depth.shape)
    np.savez('predictions.npz', depths=predictions)

if __name__ == '__main__':
    if len(sys.argv) != 2:
    	print("Please run:\n\tpython convert_npz_to_img.py <predictions_path>")
    	exit()
    path = sys.argv[1]
    convert_predictions()
