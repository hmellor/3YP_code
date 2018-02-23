import sys
import tensorflow as tf
import tflearn
import numpy as np
from model import model_network
from PIL import Image

def develop_model(net):
    model = tflearn.DNN(net,
                        clip_gradients=5.0,
                        tensorboard_verbose=2,
                        tensorboard_dir='tflearn_logs',
                        checkpoint_path='Checkpoints',
                        best_checkpoint_path=None,
                        max_checkpoints=None,
                        session=None,
                        best_val_accuracy=0.0)
    print('\n ** Model Developed ** \n')
    return model

def train(net,images,depths):
    # Build model
    model = develop_model(net)
    print('\n ** Training ** \n')
    # Train Weights
    input1 = tf.placeholder(tf.float32, shape=(None,480,640,3))
    target1 = tf.placeholder(tf.float32, shape=(None,240,320,1))
    model.fit(
        feed_dict={input1: images, target1: depths},
        n_epoch=20,
        snapshot_epoch=True,
        snapshot_step=500,
        show_metric=True,
        batch_size=10,
        shuffle=True,
        run_id='KusNet')

    return model

def validate(net,images):
    model = develop_model(net)
    print('\n ** Predicting ** \n')
    outlist = model.predict(images)
    outarray = np.asarray(outlist)
    np.savez('predictions.npz', depths=outarray)
    print('\n ** Done, saved in this directory ** \n' % (filename))

def main():

    # Check for correct number of input arguments
    if len(sys.argv) == 1:
        print("\n ** No mode selected, please run:\n\tpython KuzNet.py <train/val> ** \n")
    	exit()
    elif sys.argv[1] == 'train' and len(sys.argv) != 3:
        print("\n ** No dataset selected, please run:\n\tpython KuzNet.py train <npz_input_directory> ** \n")
    	exit()
    elif sys.argv[1] == 'val' and len(sys.argv) != 4:
    	print("\n ** No dataset or model checkpoint selected, please run:\n\tpython KuzNet.py val <npz_input_directory> <model_input_directory>** \n")
        exit()

    mode = sys.argv[1]
    input_directory= sys.argv[2]
    if sys.argv[1]=='val':
        model_directory= sys.argv[3]
    print('\n ** Loading Images from %s ** \n'% (input_directory))

    #load images from input npz file
    data = np.load(input_directory)
    images_np = data['images']
    depths_np = data['depths']

    #rearrange into proper columns
    images_np = np.transpose(images_np, [3,0,1,2])
    depths_np = np.transpose(depths_np, [2, 0, 1])
    #expand depths_np to have a single colour channel
    depths_np = np.expand_dims(depths_np, 3)
    #resize depths to 240x320 by halfing resolution using max pool
    depths_tf = tflearn.layers.conv.max_pool_2d (depths_np, 2, strides=2)

    print(images_np.shape)
    print('\n ** %s images loaded successfully** \n' % (images_np.shape[0]))

    # Build model
    net = model_network()

    print('\n ** Net Built ** \n')

    # train or validate
    if mode == 'train':
        model = train(net,images_np,depths_tf)        # load model values
    if mode == 'val':
        model = validate(net,images_tf)

    exit()

if __name__ == "__main__":
    main()
