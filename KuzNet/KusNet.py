import sys
import tensorflow as tf
import tflearn
import numpy as np
from model import model_network

fine_tune = 1

def develop_model(net):
    # if fine_tune:
    #     checkpoint =
    #     model.load('checkpoints/%s' % )
    model = tflearn.DNN(net,
                        clip_gradients=5.0,
                        tensorboard_verbose=2,
                        tensorboard_dir='tflearn_logs',
                        checkpoint_path='checkpoints/ckpt',
                        best_checkpoint_path='checkpoints/best',
                        max_checkpoints=None,
                        session=None,
                        best_val_accuracy=0.0)
    print('\n ** Model Developed ** \n')
    return model

def train(net,images,depths,val_images,val_depths):
    # Build model
    model = develop_model(net)
    print('\n ** Training ** \n')
    # Train Weights
    model.fit(
        images, depths,
        n_epoch=100,
        val_feed_dicts=(val_images, val_depths),
        snapshot_epoch=False,
        snapshot_step=500,
        show_metric=True,
        batch_size=10,
        shuffle=True,
        run_id='KusNet')

    return model

def validate(net,images):
    # Build model
    model = develop_model(net)
    print('\n ** Predicting ** \n')
    model.load('checkpoints/ckpt-') # write which ckeckpoint you want to use here
    outlist = model.predict(images)
    outarray = np.asarray(outlist)
    np.savez('predictions.npz', depths=outarray)
    print('\n ** Done, saved in this directory ** \n' % (filename))

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

    #load validate data from file
    val_data = np.load(val_path)
    val_images = val_data['images']
    val_depths = val_data['depths']

    #make sure we are using float32
    depths = np.float32(depths)
    images = np.float32(images)
    val_depths = np.float32(val_depths)
    val_images = np.float32(val_images)

    print(np.amax(depths))
    print(np.amax(images))

    print(depths.shape)
    print('\n ** %s images loaded successfully** \n' % (images.shape[0]))

    # Build model
    net = model_network()

    print('\n ** Net Built ** \n')

    # train or validate
    if mode == 'train':
        model = train(net,images,depths,val_images,val_depths)  # load model values
    if mode == 'val':
        model = validate(net,images)

    exit()

if __name__ == "__main__":
    main()
