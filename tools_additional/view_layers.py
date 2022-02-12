import tensorflow as tf

model_path = "latest_model_FC-DenseNet56_CamVid.ckpt"

sess = tf.Session()

saver = tf.train.import_meta_graph(model_path + ".meta")
saver.restore(sess, model_path)
graph = tf.get_default_graph()

with open('all_layers_1.txt', 'w') as f:
    for op in graph.get_operations():
        print(op.name, file=f)