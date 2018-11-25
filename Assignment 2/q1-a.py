import tensorflow as tf
import numpy as np
import random

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def network(**kwargs):
    args = {
        'learningRate': 0.02,
        'hiddenLayerN': 8,
        'stddev': 2,
        'hiddenType': tf.nn.sigmoid,
        'optimizer': tf.train.GradientDescentOptimizer
    }

    print(kwargs)

    for key in kwargs:
        args[key] = kwargs[key]

    learningRate = args['learningRate']
    hiddenLayerN = args['hiddenLayerN']
    stddev = args['stddev']
    hiddenType = args['hiddenType']
    optimizer = args['optimizer']

    x = tf.placeholder(tf.float32, [None, 2])
    y = tf.placeholder(tf.float32, [None, 1])

    w1 = tf.Variable(tf.random_normal([2, hiddenLayerN], stddev = stddev))
    b1 = tf.Variable(tf.zeros([1, hiddenLayerN], dtype=tf.float32))
    wo = tf.Variable(tf.random_normal([hiddenLayerN, 1], stddev = stddev))

    h1 = hiddenType(tf.matmul(x, w1)+b1)
    logits = tf.matmul(h1, wo)

    cost = tf.losses.mean_squared_error(labels=y, predictions=logits)
    trainOp = optimizer(learningRate).minimize(cost)

    return x, y, cost, trainOp, logits

trainingDim = 10
testingDim = 9

def f(x, y):
    return (np.cos(x+(6*(0.35*y))) + 2*(0.35*x*y))

trainingData = []
testingData = []

#Generate training data table
for i in range(0, trainingDim+2):
    for j in range(0, trainingDim+2):
        if(j > 0 and j < trainingDim+1 and i > 0 and i < trainingDim+1): trainingData.append([(i*(2/(trainingDim+1))-1), (j*(2/(trainingDim+1))-1)])

#Generate testing data table
for i in range(0, testingDim+2):
    for j in range(0, testingDim+2):
        if(j > 0 and j < 10 and i > 0 and i < 10): testingData.append([(i*(2/10)-1), (j*(2/10)-1)])

#shuffle tables
random.shuffle(trainingData)

testingLabels = []
trainingLabels = []

#Generate training data labels
for i in range(0, trainingDim**2):
    trainingLabels.append([f(trainingData[i][0], trainingData[i][1])])

#Generate testing data labels
for i in range(0, testingDim**2):
    testingLabels.append([f(testingData[i][0], testingData[i][1])])

colours = ['r', 'g', 'b']
numN = [2, 8, 50]

epochs = []
MSE = []

matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
fig, ax = plt.subplots()

for iter in range(0, 3):
    x, y, cost, trainOp, logits = network(hiddenLayerN = numN[iter])

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        step = 0
        prevCost = 0.0
        while(sess.run(cost, feed_dict={x: trainingData, y: trainingLabels}) > 0.0002):

            if(np.abs(sess.run(cost, feed_dict={x: trainingData, y: trainingLabels}) - prevCost) < 0.0000005): break

            step += 1
            prevCost = sess.run(cost, feed_dict={x: trainingData, y: trainingLabels})

            if step%25 == 0:
                print("Step: ", step, " Cost: ", sess.run(cost, feed_dict={x: trainingData, y: trainingLabels}))

            batches = trainingDim**2
            batchSize = int(len(trainingData)/batches)

            for i in range(batches):
                xBatch = trainingData[i*batchSize:(i+1)*batchSize]
                yBatch = trainingLabels[i*batchSize:(i+1)*batchSize]
                sess.run(trainOp, feed_dict={x: xBatch, y: yBatch})

        print("Step: ", step, " Cost: ", sess.run(cost, feed_dict={x: trainingData, y: trainingLabels}))
        z = sess.run(logits, feed_dict={x: testingData, y: testingLabels}).reshape(testingDim, testingDim)

        epochs.append(step)
        MSE.append(sess.run(cost, feed_dict={x: trainingData, y: trainingLabels}))

        X = []
        Y = []
        for i in range(0, testingDim+2):
            toAddX = []
            toAddY = []
            for j in range(0, testingDim+2):
                if(j > 0 and j < 10 and i > 0 and i < 10):
                    toAddY.append((j*(2/10)-1))
                    toAddX.append((i*(2/10)-1))
            if(len(toAddX) > 0): X.append(toAddX)
            if(len(toAddY) > 0): Y.append(toAddY)

        X = np.array(X)
        Y = np.array(Y)
        Z = np.array(testingLabels).reshape(testingDim, testingDim)

        CS = ax.contour(X, Y, z, 6, colors=colours[iter])
        if(iter == 2): CS = ax.contour(X, Y, Z, 6, colors='k')

for i in range(0, 3):
    print(numN[i]," Neurons, Epochs: ", epochs[i], " Final MSE: ", MSE[i])

ax.set_title('Neuron Comparison')
plt.show()