#encoding: utf-8
import os, random, sys
import numpy as np
from PIL import Image
from tensorflow.python.platform import gfile

def convert_train(path):
    print("load dataset: %s" % (path))
    f = np.load(path)
    
    # get 
    images = f['images']
    images_flip = np.fliplr(images)
    images = np.append(images,images_flip,)
    images = np.transpose(images, [3, 2, 1, 0])
    
    
    
    depths = f['depths']
    depths_flip = np.fliplr(depths)
    depths = np.append(depths,depths_flip)
    depths = np.transpose(depths, [2, 1, 0])
    
    counter = 0
    trains = []
    for i, (image, depth) in enumerate(zip(images, depths)):
        ra_image = image.transpose(2, 1, 0)
        ra_depth = depth.transpose(1, 0)
        re_depth = (ra_depth/np.max(ra_depth))*255.0
        image_pil = Image.fromarray(np.uint8(ra_image))
        depth_pil = Image.fromarray(np.uint8(re_depth))
        image_name = os.path.join("data","datasets_%s" % (sys.argv[1]), "%05d.jpg" % (i))
        image_pil.save(image_name)
        depth_name = os.path.join("data","datasets_%s" % (sys.argv[1]), "%05d.png" % (i))
        depth_pil.save(depth_name)

        trains.append((image_name, depth_name))
        counter=counter+1
        
    print("number of jpgs ")
    random.shuffle(trains)

    with open('%s.csv' % (sys.argv[1]), 'w') as output:
        for (image_name, depth_name) in trains:
            output.write("%s,%s" % (image_name, depth_name))
            output.write("\n")

if __name__ == '__main__':
    if len(sys.argv) != 2:
    	print("Please run:\n\tpython convert_npz_to_img.py <train/val/test>")
    	exit()
    
    #train_path = os.path.join("../data", "%s.npz" % (sys.argv[1]))
    #if not gfile.Exists('data/datasets_%s' % (sys.argv[1])):
        #gfile.MakeDirs('data/datasets_%s' % (sys.argv[1]))
        
    train_path = os.path.join("../data", "%s.npz" % (sys.argv[1]))
    if not gfile.Exists('data/MalikaGeorgedatasets_%s' % (sys.argv[1])):
        gfile.MakeDirs('data/MalikaGeorgedatasets_%s' % (sys.argv[1]))

    convert_train(train_path)
