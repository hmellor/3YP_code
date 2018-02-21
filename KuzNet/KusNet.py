import sys
import tensorflow as tf
import tflearn
import numpy as np
import model_network

def develop_model(net):
    model = tflearn.DNN(net,
                        clip_gradients=5.0,
                        tensorboard_verbose=2,
                        tensorboard_dir='/tmp/tflearn_logs/',
                        checkpoint_path='Checkpoints/',
                        best_checkpoint_path=None,
                        max_checkpoints=None,
                        session=None,
                        best_val_accuracy=0.0)
    return model

def train(net,images,depths):
    model = develop_model(net)
    model.fit(images,
          depths,
          n_epoch=20,
          snapshot_epoch=False,
          snapshot_step=500,
          show_metric=True,
          batch_size=10,
          shuffle=True,
          run_id='KusNet')

    return model

def validate(net,images):
    model = develop_model(net)
    model.predit(images)

def main():

    # Check for correct number of input arguments
    if sys.argv[1] == 'train' and len(sys.argv) != 3:
        print("\n ** If training please run:\n\tpython kuznet.py train <npz_input_directory> ** \n")
    	#exit()
    elif sys.argv[1] == 'val' and len(sys.argv) != 4:
    	print("\n ** If validating please run: \n\tpython kuznet.py val <npz_input_directory> <model_input_directory>** \n")
        #exit()

    mode = sys.argv[1]
    input_directory= sys.argv[2]
    if sys.argv[1]=='val':
        model_directory= sys.argv[3]
    print('\n ** Loading Images from %s ** \n'% (input_directory))

    #load images from input npz file
    f = np.load(input_directory)
    images_np = f['images']
    depths_np = f['depths']

    #rearrange into proper columns
    images_np = np.transpose(images_np, [3,0,1,2])
    depths_np = np.transpose(depths_np, [2, 1, 0])

    #images_tf = tf.convert_to_tensor(images_np, dtype=tf.float32)
    #images_tf = tf.convert_to_tensor(depths_np, dtype=tf.float32)

    images_placeholder = tf.placeholder(tf.float32, shape=images_np.shape)
    v = tf.Variable(images_placeholder)
    sess.run(v.initializer, feed_dict={images_placeholder: images_np})

    print('\n **Images loaded successfully ** \n')

    # Build model
    net = model_network()

    # If train mode, generate weights
    if mode == 'train':
        model = train(net,images,depths)        # load model values
    if mode == 'val':
        model = validate(net,images)

        # Run Images through models






    # Predict the image
    #predict(model, input_directory)
    exit()


if __name__ == "__main__":
    main()
