#encoding: utf-8
import os, random, sys
import numpy as np
from PIL import Image


def convert_train(path):
    print("load dataset: %s" % (path))
    f = np.load(path)
    depths = f['depths']
    depths = np.transpose(depths, [2, 1, 0])

    for i, (image, depth) in enumerate(zip(images, depths)):
        ra_depth = depth.transpose(1, 0)
        re_depth = (ra_depth/np.max(ra_depth))*255.0
        depth_pil = Image.fromarray(np.uint8(re_depth))
        depth_name = os.path.join("data","%s" % (sys.argv[1]), "%05d.png" % (i))
        depth_pil.save(depth_name)

if __name__ == '__main__':
    if len(sys.argv) != 2:
    	print("Please run:\n\tpython convert_npz_to_img.py <data_path>")
    	exit()
    data_path = sys.argv[1]
    convert_train(data_path)
