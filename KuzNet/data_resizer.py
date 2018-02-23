import numpy as np
from scipy.misc import imresize
import sys

if len(sys.argv) == 1:
    print("\n ** No data selected, please run:\n\tpython data_resizer.py <data_path> ** \n")
    exit()

data_path = sys.argv[1]

# load data
data = np.load(data_path)
depths = data['depths']
images = data['images']

#resize depths to 240x320
depths_resized = np.zeros([0, 240, 320], dtype=np.float32)
for depth in range(depths_np.shape[0]):
    temp = imresize(depths[depth], [240, 320], 'lanczos')
    temp = np.float32(temp)
    temp = (temp/np.max(temp))*255.0
    depths_resized = np.append(depths_resized, np.expand_dims(temp, axis=0), axis=0)
    if (depth+1) % 25 == 0:
        print('%d depth images resized' % (depth+1))

#expand depths_np to have a single colour channel
depths = np.expand_dims(depths, 3)

#save to .npz
numpy.savez('%s_resized' % data_path, depths=depths_resized, images=images)
