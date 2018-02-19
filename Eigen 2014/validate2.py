#encoding: utf-8
import os, random, sys
import numpy as np
from PIL import Image
from tensorflow.python.platform import gfile
import tensorflow as tf

def convert_train(path):
    print("load dataset: %s" % (path))
    f = np.load(path)
    images = f['images']
    images = np.transpose(images, [3, 2, 1, 0])
    depths = f['depths']
    depths = np.transpose(depths, [2, 1, 0])


    for i, (image, depth) in enumerate(zip(images, depths)):
        ra_image = image.transpose(2, 1, 0)

        images = tf.convert_to_tensor(ra_image, dtype=tf.float32)


        print('\n**')
        print(images.get_shape())
        print('**\n')
        #image_pil = Image.fromarray(np.uint8(ra_image))
        #depth_pil = Image.fromarray(np.uint8(re_depth))
        #image_name = os.path.join("data","datasets_%s" % (sys.argv[1]), "%05d.jpg" % (i))
        #image_pil.save(image_name)
        #depth_name = os.path.join("data","datasets_%s" % (sys.argv[1]), "%05d.png" % (i))
        #depth_pil.save(depth_name)

        #trains.append((image_name, depth_name))

        #random.shuffle(trains)

if __name__ == '__main__':
    if len(sys.argv) != 2:
    	print("Please run:\n\tpython convert_npz_to_img.py <train/val/test>")
    	exit()
    train_path = os.path.join("data/", "%s.npz" % (sys.argv[1]))
    convert_train(train_path)
