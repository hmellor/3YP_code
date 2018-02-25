#encoding: utf-8
import os, random, sys, datetime
import numpy as np
from PIL import Image

def convert_train(data_path):
    print("load dataset: %s" % (data_path))

    #manipulate argument for file naming
    file_name, ext = path.splitext(data_path)
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

    f = np.load(data_path)
    depths = f['depths']
    depths = np.transpose(depths, [2, 1, 0])

    for i, depth in enumerate(depths):
        ra_depth = depth.transpose(1, 0)
        re_depth = (ra_depth/np.max(ra_depth))*255.0
        depth_pil = Image.fromarray(np.uint8(re_depth))
        depth_name = os.path.join("data", "%s_%s" % (file_name, time_str), "%05d.png" % (i))
        depth_pil.save(depth_name)

if __name__ == '__main__':
    if len(sys.argv) != 2:
    	print("Please run:\n\tpython convert_npz_to_img.py <data_path>")
    	exit()
    data_path = sys.argv[1]
    convert_train(data_path)
