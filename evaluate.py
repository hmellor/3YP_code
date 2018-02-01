import sys
import os.path
import numpy

if len(sys.argv) != 3:
	print("Please run:\n\tpython evaluate.py <predictions>.npz <ground truth>.npz")
	exit()

if not os.path.isfile(sys.argv[1]):
	print(sys.argv[1] + " is not a file")
	exit()

if not os.path.isfile(sys.argv[2]):
	print(sys.argv[2] + " is not a file")
	exit()

predictions = numpy.load(sys.argv[1])
depths = numpy.load(sys.argv[2])

predictions = predictions['depths']
depths = depths['depths']

if not depths.shape == predictions.shape:
	print("Error: depths.shape != predictions.shape")
	exit()

n_pxls = depths.size

# Mean Absolute Relative Error
rel = numpy.divide( abs(depths - predictions), depths )
rel = rel.sum() / n_pxls
print('Mean Absolute Relative Error: ' + str(rel))

# Root Mean Squared Error
rms = numpy.square( depths - predictions )
rms = numpy.sqrt(rms.sum() / n_pxls)
print('Root Mean Squared Error: ' + str(rms))

# LOG10 Error
predictionsPos = predictions
predictionsPos[ predictionsPos < 0 ] = 10**19
lg10 = abs(numpy.log10(depths) - numpy.log10(predictionsPos))
lg10 = lg10.sum() / n_pxls
print('Mean Log10 Error: ' + str(lg10))

