import tensorflow as tf
import os
import numpy as np

def convert_predictions()
    path = sys.argv[1]
    images = os.listdir(path)

    for i, image in enumerate(zip(images))
        depth_png = tf.read_file(image)
        depth = tf.image.decode_png(depth_png, channels=1)
        prediction = tf.image.resize_images(depth,[480,640])
        prediction = tf.squeeze(predictions,3)
        prediction = tf.transpose(predictions, [1,2,0])
    np.savez('%s.npz' % (output_dir), depths=predictions)

if __name__ == '__main__':
    if len(sys.argv) != 2:
    	print("Please run:\n\tpython convert_npz_to_img.py <predictions_path>")
    	exit()
    path = sys.argv[1]
    convert_train()
