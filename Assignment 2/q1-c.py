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
validationData = []

#Generate training data table
for i in range(0, trainingDim+2):
    for j in range(0, trainingDim+2):
        if(j > 0 and j < trainingDim+1 and i > 0 and i < trainingDim+1): trainingData.append([(i*(2/(trainingDim+1))-1), (j*(2/(trainingDim+1))-1)])

#Generate testing data table
for i in range(0, testingDim+2):
    for j in range(0, testingDim+2):
        if(j > 0 and j < 10 and i > 0 and i < 10): testingData.append([(i*(2/10)-1), (j*(2/10)-1)])

#Generate validation data table
for i in range(0, trainingDim+2):
    for j in range(0, trainingDim+2):
        if(j > 0 and j < trainingDim+1 and i > 0 and i < trainingDim+1): validationData.append([random.uniform(-1, 1), random.uniform(-1, 1)])

#shuffle tables
random.shuffle(trainingData)

testingLabels = []
trainingLabels = []
validationLabels = []

#Generate training data labels
for i in range(0, trainingDim**2):
    trainingLabels.append([f(trainingData[i][0], trainingData[i][1])])

#Generate testing data labels
for i in range(0, testingDim**2):
    testingLabels.append([f(testingData[i][0], testingData[i][1])])

#Generate testing data labels
for i in range(0, trainingDim**2):
    validationLabels.append([f(validationData[i][0], validationData[i][1])])

matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
fig, ax = plt.subplots()

#Run with early stopping
x, y, cost, trainOp, logits = network(optimizer=tf.train.RMSPropOptimizer)

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    z1 = None
    z2 = None

    step = 0
    prevCost = 0.0
    numErr = 0
    prevErr = 0.0
    while(step < 100):

        if(sess.run(cost, feed_dict={x: validationData, y: validationLabels}) > prevErr): numErr += 1
        else: numErr = 0

        if(numErr == 10): z1 = sess.run(logits, feed_dict={x: testingData, y: testingLabels}).reshape(testingDim, testingDim)

        prevErr = sess.run(cost, feed_dict={x: validationData, y: validationLabels})

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
    z2 = sess.run(logits, feed_dict={x: testingData, y: testingLabels}).reshape(testingDim, testingDim)
    if(type(z1) is None): z1 = z2

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

    CS = ax.contour(X, Y, z1, 6, colors='r')
    CS = ax.contour(X, Y, z2, 6, colors='b')
    CS = ax.contour(X, Y, Z, 6, colors='k')

ax.set_title('Early stopping comparison')
plt.show()

MSE = []

for iter in range(1, 50):
    x, y, cost, trainOp, logits = network(hiddenLayerN = (iter+1), optimizer=tf.train.RMSPropOptimizer)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        step = 0
        prevCost = 0.0
        while(sess.run(cost, feed_dict={x: trainingData, y: trainingLabels}) > 0.02):

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

        MSE.append(sess.run(cost, feed_dict={x: trainingData, y: trainingLabels}))

X = np.arange(2, 51)

plt.plot(X, MSE)
plt.ylabel("MSE value at convergence")
plt.xlabel("# of Neurons")
plt.show()