


    predictions = tf.image.resize_images(depths,[480,640])
    predictions = tf.squeeze(predictions,3)
    predictions = tf.transpose(predictions, [1,2,0])
    np.savez('%s.npz' % (output_dir), depths=predictions)
