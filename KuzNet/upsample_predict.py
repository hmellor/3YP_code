import numpy as np
from scipy.misc import imresize
from os import path
import sys

#define desired resolutions of input and target data
height = 480
width = 640

#check that an argument has been provided
if len(sys.argv) == 1:
    print("\n ** No data selected, please run:\n\tpython upsample_predict.py <data_path> ** \n")
    exit()

#manipulate argument for file naming
data_path = sys.argv[1]
file_name, ext = path.splitext(data_path)
file_name = '%s_resized' % file_name

# load data
data = np.load(data_path)
depths = data['depths']

#initialise resized arrays
depths_resized = np.zeros([0, height, width], dtype=np.float32)
for i in range(depths.shape[0]):
    #resize depth image to the desired resolution
    de_temp = imresize(depths[i], [height, width], 'bicubic')
    #convert to float32
    de_temp = np.float32(de_temp)
    #append the processed image to the output array
    depths_resized = np.append(depths_resized, np.expand_dims(de_temp, axis=0), axis=0)
    #print every 25 sets of images so you can see it's working
    if (i+1) % 25 == 0:
        print('%d depth images resized' % (i+1))

#rearrange into proper columns
depths_resized = np.transpose(depths_resized, [1, 2, 0])

#check data shapes
print(depths_resized.shape)

#save to .npz
np.savez(file_name, depths=depths_resized)
