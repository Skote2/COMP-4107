import tensorflow as tf
import numpy as np
import random

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import time

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
    if(optimizer == tf.train.MomentumOptimizer): trainOp = optimizer(learningRate, 0.001).minimize(cost)
    else: trainOp = optimizer(learningRate).minimize(cost)

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

optimizers = [tf.train.GradientDescentOptimizer, tf.train.MomentumOptimizer, tf.train.RMSPropOptimizer]
mode = ["Gradient Descent Optimizer", "Momentum Optimizer", "RMS Optimizer"]

epochs = []
MSE = []
Times = []

for iter in range(0, 3):
    x, y, cost, trainOp, logits = network(optimizer = optimizers[iter])
    MSEVals = []
    times = []

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        step = 0
        prevCost = 0.0
        startT = int(round(time.time() * 1000))

        while(sess.run(cost, feed_dict={x: trainingData, y: trainingLabels}) > 0.02 and step < 100):
            step += 1
            prevCost = sess.run(cost, feed_dict={x: trainingData, y: trainingLabels})
            MSEVals.append(sess.run(cost, feed_dict={x: trainingData, y: trainingLabels}))

            if step%25 == 0:
                print("Step: ", step, " Cost: ", sess.run(cost, feed_dict={x: trainingData, y: trainingLabels}))

            batches = trainingDim**2
            batchSize = int(len(trainingData)/batches)

            for i in range(batches):
                xBatch = trainingData[i*batchSize:(i+1)*batchSize]
                yBatch = trainingLabels[i*batchSize:(i+1)*batchSize]
                sess.run(trainOp, feed_dict={x: xBatch, y: yBatch})

            times.append(int(round(time.time() * 1000)) - startT)
            startT = int(round(time.time() * 1000))

        plt.plot(MSEVals, label=mode[iter])

        print("Step: ", step, " Cost: ", sess.run(cost, feed_dict={x: trainingData, y: trainingLabels}))

        epochs.append(step)
        MSE.append(sess.run(cost, feed_dict={x: trainingData, y: trainingLabels}))

    Times.append(times)

acc = 100.0
result = 0

for iter in range(0, 3):
    x, y, cost, trainOp, logits = network(optimizer = optimizers[iter])

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        step = 0
        prevCost = 0.0
        startT = int(round(time.time() * 1000))

        while(step < 100):
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

        if(sess.run(cost, feed_dict={x: trainingData, y: trainingLabels}) < acc):
            acc = sess.run(cost, feed_dict={x: trainingData, y: trainingLabels})
            result = iter

print("Best accuracy after 100 epochs is: ", acc, " from ", mode[iter])

for i in range(0, 3):
    print(mode[i]," Epochs: ", epochs[i], " Final MSE: ", MSE[i])

plt.ylabel('MSE Value')
plt.xlabel('Epoch')
plt.legend()

fig, ax = plt.subplots()
plt.bar([0, 1, 2], [np.mean(Times[0]), np.mean(Times[1]), np.mean(Times[2])])
plt.xticks([0, 1, 2], mode)
plt.ylabel('Time(ms)')
plt.xlabel('Optimizer')
plt.show()