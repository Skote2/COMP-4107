import tensorflow as tf
import numpy as np
import pickle
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 100
test_size = 256

#Begin data processing section

#this funciton is copied form tensorflow.org
def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

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

def model(X, w, w_fc, w_o, p_keep_conv, p_keep_hidden):
    l1a = tf.nn.relu(tf.nn.conv2d(X, w,                       # l1a shape=(?, 28, 28, 32)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],              # l1 shape=(?, 14, 14, 32)
                        strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)


    l3 = tf.reshape(l1, [-1, w_fc.get_shape().as_list()[0]])    # reshape to (?, 14x14x32)

    l4 = tf.nn.relu(tf.matmul(l3, w_fc))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o)
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

with tf.name_scope("Weights"):
    w = init_weights([3, 3, 3, 32])       # 3x3x1 conv, 32 outputs
    w_fc = init_weights([32 * 16 * 16, 625]) # FC 32 * 14 * 14 inputs, 625 outputs
    w_o = init_weights([625, 10])         # FC 625 inputs, 10 outputs (labels)
    variable_summaries(w)
    variable_summaries(w_fc)
    variable_summaries(w_o)

with tf.name_scope("pVariables"):
    p_keep_conv = tf.placeholder("float")
    p_keep_hidden = tf.placeholder("float")
with tf.name_scope("model"):
    py_x = model(X, w, w_fc, w_o, p_keep_conv, p_keep_hidden)

with tf.name_scope("Loss"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
with tf.name_scope("GradientDescent"):
    train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)

with tf.name_scope('Accuracy'):
    # Accuracy
    predict_op = tf.argmax(py_x, 1)
    acc = tf.equal(tf.argmax(py_x, 1), tf.argmax(Y, 1))
    acc = tf.reduce_mean(tf.cast(acc, tf.float32))

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()
    # Create a summary to monitor cost tensor
    tf.summary.scalar("loss", cost)
    # Create a summary to monitor accuracy tensor
    tf.summary.scalar("accuracy", acc)
    
    summaries = tf.summary.merge_all()
    
    writer = tf.summary.FileWriter('./board/question1part5/model1/', sess.graph)
    
    for j in range(1, 6):
        trX, trY = loadBatch("CIFARdata", j)
        for i in range(15):
            training_batch = zip(range(0, len(trX), batch_size),
                                range(batch_size, len(trX)+1, batch_size))
            for start, end in training_batch:
                summary, _ = sess.run([summaries, train_op], feed_dict={X: trX[start:end], Y: trY[start:end],
                                            p_keep_conv: 0.8, p_keep_hidden: 0.5})
                writer.add_summary(summary, i)

            test_indices = np.arange(len(teX)) # Get A Test Batch
            np.random.shuffle(test_indices)
            test_indices = test_indices[0:test_size]

            accuracy = sess.run(predict_op, feed_dict={X: teX[test_indices],
                                                            p_keep_conv: 1.0,
                                                            p_keep_hidden: 1.0})
            tf.summary.scalar("accuracy", accuracy)
            print(i, np.mean(np.argmax(teY[test_indices], axis=1) == accuracy))
