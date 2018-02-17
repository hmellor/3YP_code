import gfile, sys, model
import tensorflow as tf
import numpy as np
from PIL import Image

if len(sys.argv) != 2:
    print("Please run:\n\tpython validate.py <val/test>")
    exit()

VALIDATE_FILE = '%s.csv' % (sys.argv[1])
MODEL_DIR = "refine_train"

with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)
    images = csv_inputs(VALIDATE_FILE)
    keep_conv = tf.placeholder(tf.float32)
    keep_hidden = tf.placeholder(tf.float32)

    print("refine validate.")
    coarse = model.inference(images, keep_conv, trainable=False)
    logits = model.inference_refine(images, coarse, keep_conv, keep_hidden)

    loss = model.loss(logits, depths, invalid_depths)
    train_op = op.train(loss, global_step, BATCH_SIZE)
    init_op = tf.initialize_all_variables()

    # Session
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=LOG_DEVICE_PLACEMENT))
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

    # fine tune
    if FINE_TUNE:
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

    _, loss_value, logits_val, images_val = sess.run([train_op, loss, logits, images], feed_dict={keep_conv: 0.8, keep_hidden: 0.5})
    output_predict(logits_val, "data/%s_predict" % (sys.argv[2])

    coord.request_stop()
    coord.join(threads)
    sess.close()

def csv_inputs(csv_file_path):
    IMAGE_HEIGHT = 228
    IMAGE_WIDTH = 304
    filename_queue = tf.train.string_input_producer([csv_file_path], shuffle=False)
    reader = tf.TextLineReader()
    _, serialized_example = reader.read(filename_queue)
    filename = tf.decode_csv(serialized_example, [["path"], ["annotation"]])
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
