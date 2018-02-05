import numpy
import copy

# load data and print matrices
data = numpy.load('data/val.npz')
print('Images: ', data['images'].shape)
print('Depths: ', data['depths'].shape)

print('Show sample image')
imageIdx = 0
image = data['images'][:,:,:,imageIdx]
depth = data['depths'][:,:,imageIdx]

#from PIL import Image
#image2 = Image.fromarray(image, 'RGB')
#image2.show()

#import matplotlib.pyplot as plt
#plt.imshow(image)
#plt.show()
#plt.imshow(depth)
#plt.show()

print('Add a noise to depths and save them as predictions')
predictions = copy.deepcopy(data['depths'])
noise = numpy.random.normal(0, 0.5, data['depths'].shape)
predictions += noise
numpy.savez('predictions', depths=predictions)
