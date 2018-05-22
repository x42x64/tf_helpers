import tensorflow as tf
from tensorflow.python.summary import summary

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    new_saver = tf.train.import_meta_graph('/home/jan/Downloads/squeezeDet/model.ckpt-87000.meta')
    new_saver.restore(sess, '/home/jan/Downloads/squeezeDet/model.ckpt-87000')

    log_dir = '/home/jan/Downloads/squeezeDet/tensorboard'

    pb_visual_writer = summary.FileWriter(log_dir)
    pb_visual_writer.add_graph(sess.graph)
    print("Model Imported. Visualize by running: "
          "tensorboard --logdir={}".format(log_dir))

    print("done")