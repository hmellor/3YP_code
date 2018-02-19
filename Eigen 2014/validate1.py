import sys, model
from tensorflow.python.platform import gfile
import tensorflow as tf
import numpy as np
from PIL import Image
import train_operation as op

if len(sys.argv) != 2:
    print("Please run:\n\tpython validate.py <val/test>")
    exit()

VALIDATE_FILE = '%s.csv' % (sys.argv[1])
MODEL_DIR = "refine_train"

def csv_inputs(csv_file_path):
   
    IMAGE_HEIGHT = 228
    IMAGE_WIDTH = 304
    filename_queue = tf.train.string_input_producer([csv_file_path], shuffle=False)
    
    print(csv_file_path)
    
    reader = tf.TextLineReader()
    _, serialized_example = reader.read(filename_queue)
    
    print('\n**')
    print(serialized_example)
    print('** \n')
    
    filename, depth_filename = tf.decode_csv(serialized_example, [["path"], ["annotation"]])
    
    print('\n**')
    print(tf.shape(filename))
    print('** \n')

    # input
    jpg = tf.read_file(filename)
    image = tf.image.decode_jpeg(jpg, channels=3)
    image = tf.cast(image, tf.float32)
    # resize
    image = tf.image.resize_images(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
    # generate batch
    images = tf.train.batch(
        [image],
        batch_size=10000,
        num_threads=4,
        capacity= 10000,
        allow_smaller_final_batch=True
    )
    return images

def output_predict(depths, output_dir):
    print("output predict into %s" % output_dir)
    if not gfile.Exists(output_dir):
        gfile.MakeDirs(output_dir)
    for i, (depth) in enumerate(depths):
        depth = depth.transpose(2, 0, 1)
        if np.max(depth) != 0:
            ra_depth = (depth/np.max(depth))*255.0
        else:
            ra_depth = depth*255.0
        depth_pil = Image.fromarray(np.uint8(ra_depth[0]), mode="L")
        depth_name = "%s/%05d.png" % (output_dir, i)
        depth_pil.save(depth_name)

def val():
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        images = csv_inputs(VALIDATE_FILE)
        keep_conv = tf.placeholder(tf.float32)
        keep_hidden = tf.placeholder(tf.float32)

        print("refine validate.")
        coarse = model.inference(images, keep_conv, trainable=False)
        logits = model.inference_refine(images, coarse, keep_conv, keep_hidden)

        init_op = tf.initialize_all_variables()

        # Session
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        sess.run(init_op)

        # parameters
        coarse_params = {}
        refine_params = {}

        for variable in tf.all_variables():
            variable_name = variable.name
            print("parameter: %s" % (variable_name))
            if variable_name.find("/") < 0 or variable_name.count("/") != 1:
                continue
            if variable_name.find('coarse') >= 0:
                coarse_params[variable_name] = variable
            print("parameter: %s" %(variable_name))
            if variable_name.find('fine') >= 0:
                refine_params[variable_name] = variable

        # define saver
        saver_refine = tf.train.Saver(refine_params)


        # import pretrained model

        refine_ckpt = tf.train.get_checkpoint_state(MODEL_DIR)
        if refine_ckpt and refine_ckpt.model_checkpoint_path:
            print("Pretrained refine Model Loading.")
            saver_refine.restore(sess, refine_ckpt.model_checkpoint_path)
            print("Pretrained refine Model Restored.")
        else:
            print("No Pretrained refine Model.")

        # train
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        logits_val= sess.run([logits], feed_dict={keep_conv: 0.8, keep_hidden: 0.5})
        output_predict(logits_val, "data/%s_predict" % (sys.argv[1] ) )

        coord.request_stop()
        coord.join(threads)
        sess.close()

def main(argv=None):
    val()

if __name__ == '__main__':
    tf.app.run()
