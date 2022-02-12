import tensorflow as tf

tf.train.import_meta_graph("./model.ckpt.meta")
with open('checkpoints/all_layers.txt', 'w') as f:
    for n in tf.get_default_graph().as_graph_def().node:
        print(n, file=f)
   
with tf.Session() as sess:
  writer = tf.summary.FileWriter("./output/", sess.graph)
  writer.close()