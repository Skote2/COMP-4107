#Here we try using 4 convolutional layers, 3x3x3x32, 3x3x32x64, 5x5x64x128, and 5x5x128x256; along with four fully connected layers with 625, 512, 256, and 128 neurons

import tensorflow as tf
import numpy as np
import pickle
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 100
test_size = 256

#Begin data processing section

def normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x

def oneHotEncode(x):
    encoded = np.zeros((len(x), 10))
    for idx, val in enumerate(x):
        encoded[idx][val] = 1
    return encoded

def loadBatch(folder, id):

    if(id == 0): 
        with open(folder + '/test_batch', mode='rb') as file:
            batch = pickle.load(file, encoding='latin1')

    else:
        with open(folder + '/data_batch_' + str(id), mode='rb') as file:
            batch = pickle.load(file, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']

    return normalize(features), oneHotEncode(labels)

def loadLabels():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#End data processing section

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w, w2, w3, w4, w_fc, w_fc2, w_fc3, w_fc4, w_o, p_keep_conv, p_keep_hidden):
    l1a = tf.nn.relu(tf.nn.conv2d(X, w,                       # l1a shape=(?, 32, 32, 32)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],              # l1 shape=(?, 16, 16, 32)
                        strides=[1, 2, 2, 1], padding='SAME')
    #l1 = tf.nn.dropout(l1, p_keep_conv)

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2,                       # l1a shape=(?, 16, 16, 64)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],              # l1 shape=(?, 8, 8, 64)
                        strides=[1, 2, 2, 1], padding='SAME')
    #l2 = tf.nn.dropout(l2, p_keep_conv)

    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3,                       # l1a shape=(?, 16, 16, 64)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],              # l1 shape=(?, 8, 8, 64)
                        strides=[1, 2, 2, 1], padding='SAME')
    #l3 = tf.nn.dropout(l3, p_keep_conv)

    l4a = tf.nn.relu(tf.nn.conv2d(l3, w4,                       # l1a shape=(?, 16, 16, 64)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l4 = tf.nn.max_pool(l4a, ksize=[1, 2, 2, 1],              # l1 shape=(?, 8, 8, 64)
                        strides=[1, 2, 2, 1], padding='SAME')
    l4 = tf.nn.dropout(l4, p_keep_conv)

    l5 = tf.reshape(l4, [-1, w_fc.get_shape().as_list()[0]])    # reshape to (?, 4x4x128)

    l6 = tf.nn.relu(tf.matmul(l5, w_fc))
    l6 = tf.nn.dropout(l6, p_keep_hidden)

    l7 = tf.nn.relu(tf.matmul(l6, w_fc2))
    l7 = tf.nn.dropout(l7, p_keep_hidden)

    l8 = tf.nn.relu(tf.matmul(l7, w_fc3))
    l8 = tf.nn.dropout(l8, p_keep_hidden)

    l9 = tf.nn.relu(tf.matmul(l8, w_fc4))
    l9 = tf.nn.dropout(l9, p_keep_hidden)

    pyx = tf.matmul(l9, w_o)
    return pyx

TrF, TrL = loadBatch("CIFARdata", 1)
TeF, TeL = loadBatch("CIFARdata", 0)

#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
#trX = trX.reshape(-1, 32, 32, 3)  # 32x32x3 input img
#teX = teX.reshape(-1, 32, 32, 3)  # 32x32x3 input img

trX, trY, teX, teY = TrF, TrL, TeF, TeL

X = tf.placeholder("float", [None, 32, 32, 3])
Y = tf.placeholder("float", [None, 10])

w = init_weights([3, 3, 3, 32])       # 3x3x3 conv, 32 outputs
w2 = init_weights([3, 3, 32, 64])       # 3x3x32 conv, 64 outputs
w3 = init_weights([5, 5, 64, 128])       # 5x5x64 conv, 128 outputs
w4 = init_weights([5, 5, 128, 256])       # 5x5x128 conv, 256 outputs
w_fc = init_weights([256 * 2 * 2, 625]) # FC 32 * 16 * 16 inputs, 625 outputs
w_fc2 = init_weights([625, 512]) # FC 32 * 16 * 16 inputs, 625 outputs
w_fc3 = init_weights([512, 256]) # FC 32 * 16 * 16 inputs, 625 outputs
w_fc4 = init_weights([256, 128])
w_o = init_weights([128, 10])         # FC 625 inputs, 10 outputs (labels)z

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w, w2, w3, w4, w_fc, w_fc2, w_fc3, w_fc4, w_o, p_keep_conv, p_keep_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    for j in range(1, 6):
        trX, trY = loadBatch("CIFARdata", j)
        for i in range(100):
            training_batch = zip(range(0, len(trX), batch_size),
                                range(batch_size, len(trX)+1, batch_size))
            for start, end in training_batch:
                sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                            p_keep_conv: 0.8, p_keep_hidden: 0.5})

            test_indices = np.arange(len(teX)) # Get A Test Batch
            np.random.shuffle(test_indices)
            test_indices = test_indices[0:test_size]

            print(i, np.mean(np.argmax(teY[test_indices], axis=1) ==
                            sess.run(predict_op, feed_dict={X: teX[test_indices],
                                                            p_keep_conv: 1.0,
                                                            p_keep_hidden: 1.0})))
