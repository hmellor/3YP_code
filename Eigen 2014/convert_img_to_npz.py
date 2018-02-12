import tensorflow as tf
import os, sys
import numpy as np

def convert_predictions():
    path = sys.argv[1]
    images = os.listdir(path)

    predictions = []
    for i, (image) in enumerate(zip(images)):
        depth_png = tf.read_file(image)
        depth = tf.image.decode_png(depth_png, channels=1)
        print(shape.depth)
        prediction = tf.image.resize_images(depth,[480,640])
        print(shape.depth)
        prediction = tf.squeeze(prediction,3)
        print(shape.depth)
        prediction = tf.transpose(prediction, [1,2,0])
        print(shape.depth)
        predictions[:,:,i] = prediction
        print(shape.depth)
    np.savez('predictions.npz', depths=predictions)

if __name__ == '__main__':
    if len(sys.argv) != 2:
    	print("Please run:\n\tpython convert_npz_to_img.py <predictions_path>")
    	exit()
    path = sys.argv[1]
    convert_train()
