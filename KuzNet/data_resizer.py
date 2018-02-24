import numpy as np
from scipy.misc import imresize
from os import path
import sys

#define desired resolutions of input and target data
image_height = 480
image_width = 640
target_height = 240
target_width = 320

#check that an argument has been provided
if len(sys.argv) == 1:
    print("\n ** No data selected, please run:\n\tpython data_resizer.py <data_path> ** \n")
    exit()

#manipulate argument for file naming
data_path = sys.argv[1]
file_name, ext = path.splitext(data_path)
file_name = '%s_resized' % file_name

# load data
data = np.load(data_path)
depths = data['depths']
images = data['images']

#rearrange into proper columns
images = np.transpose(images, [3,0,1,2])
depths = np.transpose(depths, [2, 0, 1])

#initialise resized arrays
depths_resized = np.zeros([0, target_height, target_width], dtype=np.float32)
images_resized = np.zeros([0, image_height, image_width, 3], dtype=np.float32)
for i in range(depths.shape[0]):
    #resize depth image to the desired resolution
    de_temp = imresize(depths[i], [target_height, target_width], 'lanczos')
    #convert to float32
    de_temp = np.float32(de_temp)
    #normalise so that all images have the same maximum brightness
    # de_temp = (de_temp/np.max(de_temp))*255.0
    #append the processed image to the output array
    depths_resized = np.append(depths_resized, np.expand_dims(de_temp, axis=0), axis=0)
    #do exactly the same for the images
    im_temp = imresize(images[i], [image_height, image_width], 'lanczos')
    im_temp = np.float32(im_temp)
    # im_temp = (im_temp/np.max(im_temp))*255.0
    images_resized = np.append(images_resized, np.expand_dims(im_temp, axis=0), axis=0)
    #print every 25 sets of images so you can see it's working
    if (i+1) % 25 == 0:
        print('%d image/depth pairs resized' % (i+1))

#expand depths_resized to have a single colour channel
depths_resized = np.expand_dims(depths_resized, 3)

#check data shapes and maximum values
print(np.amax(depths_resized))
print(np.amax(images_resized))
print(images_resized.shape)
print(depths_resized.shape)

#save to .npz
np.savez(file_name, depths=depths_resized, images=images_resized)
