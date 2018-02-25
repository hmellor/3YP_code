import sys
import tensorflow as tf
import tflearn
import numpy as np
from model import model_network
import datetime
from tensorflow.python import debug as tf_debug

fine_tune = 1

def develop_model(net):
    # if fine_tune:
    #     checkpoint =
    #     model.load('checkpoints/%s' % )

    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

    model = tflearn.DNN(net,
                        clip_gradients=5.0,
                        tensorboard_verbose=3,
                        tensorboard_dir='tflearn_logs',
                        checkpoint_path='checkpoints/%s/ckpt' % (time_str),
                        best_checkpoint_path='checkpoints/%s/best' % (time_str),
                        max_checkpoints=None,
                        session=None,
                        best_val_accuracy=0.9)
    print('\n ** Model Developed ** \n')
    return model

def train(net,images,depths,val_images,val_depths):
    # Build model
    model = develop_model(net)
    print('\n ** Training ** \n')

    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

    # Train Weights
    hook = tf_debug.TensorBoardDebugHook("anubis:6007")
    model.fit(
        images, depths,
        n_epoch=100,
        validation_set=(val_images, val_depths),
        snapshot_epoch=True,
        show_metric=True,
        batch_size=10,
        shuffle=True,
        run_id='KusNet_'+ time_str,)

    return model

def validate(net,val_images,variables_path):
    # Build model
    model = develop_model(net)
    print('\n ** Predicting ** \n')
    model.load(variables_path) # write which ckeckpoint you want to use here
    outlist = model.predict(val_images)
    outarray = np.asarray(outlist)
    print(outarray.shape)
    np.savez('predictions.npz', depths=outarray)
    print('\n ** Done, saved in this directory ** \n')

def main():

    # Check for correct number of input arguments
    if len(sys.argv) == 1:
        print("\n ** No mode selected, please run:\n\tpython KuzNet.py <train/val> ** \n")
    	exit()
    elif sys.argv[1] == 'train' and len(sys.argv) != 4:
        print("\n ** No dataset selected, please run:\n\tpython KuzNet.py train <npz_train_path> <npv_val_path>** \n")
    	exit()
    elif sys.argv[1] == 'val' and len(sys.argv) != 4:
    	print("\n ** No dataset or model checkpoint selected, please run:\n\tpython KuzNet.py val <npz_val_path> <variables_path>** \n")
        exit()

    mode = sys.argv[1]
    train_path = sys.argv[2]
    val_path = sys.argv[3]
    if sys.argv[1]=='val':
        val_path = sys.argv[2]
        variables_path = sys.argv[3]

    print('\n ** Loading Images from %s ** \n'% (train_path))

    if sys.argv[1]=='train':
        #load images from input npz file
        data = np.load(train_path)
        images = data['images']
        depths = data['depths']
        depths = np.float32(depths)
        images = np.float32(images)

    #load validate data from file
    val_data = np.load(val_path)
    val_images = val_data['images']
    val_depths = val_data['depths']

    #make sure we are using float32

    val_depths = np.float32(val_depths)
    val_images = np.float32(val_images)

    if sys.argv[1]=='train':
        print(depths.shape)
        print('\n ** %s images loaded successfully** \n' % (images.shape[0]))
    else:
        print(val_depths.shape)
        print('\n ** %s images loaded successfully** \n' % (val_images.shape[0]))

    # Build model
    net = model_network()

    print('\n ** Net Built ** \n')

    # train or validate
    if mode == 'train':
        model = train(net,images,depths,val_images,val_depths)  # load model values
    if mode == 'val':
        model = validate(net,val_images,variables_path)

    exit()

if __name__ == "__main__":
    main()
