import random
import os
import numpy as np
import cv2
import tensorflow as tf
import Network
from os import listdir
from os.path import isfile, join


def rebin(a, shape):
  sh = shape[0], a.shape[0] // shape[0], shape[1], a.shape[1] // shape[1]
  return a.reshape(sh).mean(-1).mean(1)


# Parameters
BatchLength = 10  # Number of images in a minibatch
X = np.load('Data2700.npy')
X = rebin(X, [X.shape[0], 270])
SizeX = X.shape[1]
SizeY = 1
SizeZ = 1  # Input img will be resized to this size
NumEpochs = 100  # Number of epochs to run
learning_rate = 1e-4
n_classes = 2  # number of output classes
droput = 0.1  # droupout parameters in the FNN layer - currently not used
EvalFreq = 10  # evaluate on every 100th iteration

index = 0
first = True
n_input = SizeX * SizeY * SizeZ

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])  # network input
y = tf.placeholder(tf.float32, [None, n_classes])  # network output
# dropout (keep probability -currently not used)
keep_prob = tf.placeholder(tf.float32)

weights, biases = Network.GetRandomWeightsAndBiases(n_classes)


# Construct model
pred = Network.conv_net(x, weights, biases, [SizeX, SizeY, SizeZ])


# Define loss and optimizer
with tf.name_scope('loss'):
  loss = tf.reduce_mean(tf.contrib.losses.softmax_cross_entropy(pred, y))
  #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
with tf.name_scope('accuracy'):
  correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.name_scope('optimizer'):
  #grads_and_vars = optimizer.compute_gradients(cnn.loss)
  #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)
  # Use ADAM optimizer this is currently the best performing training
  # algorithm in most cases
  optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
  #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# Evaluate model - currently not used, evaluation goal could also be defined in TF
#correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()


TrainIds = []
TestIds = []
TrainLabels = []
TestLabels = []

# generate !!RANDOM!! labels
for ind in range(X.shape[0]):
  if np.random.rand() > 0.1:
    # add to train data
    TrainIds.append(ind)
    # random label
    if np.random.rand() > 0.5:
      TrainLabels.append([1, 0])
    else:
      TrainLabels.append([0, 1])
  else:
    # add to test data
    TestIds.append(ind)
    # random label
    if np.random.rand() > 0.5:
      TestLabels.append([1, 0])
    else:
      TestLabels.append([0, 1])

print "Number of images for training: " + str(len(TrainIds))
print "Number of images for test: " + str(len(TestIds))

# save log here
logs_path = '/tmp/tensorflow_logs'

tf.summary.scalar("loss", loss)
tf.summary.scalar("accuracy", accuracy)

# Launch the session with default graph
with tf.Session() as sess:
  sess.run(init)
  summary_writer = tf.summary.FileWriter(logs_path, tf.get_default_graph())
  saver = tf.train.Saver()

  step = 1
  # Keep training until reach max iterations - other stopping criterion
  # could be added
  while step < NumEpochs:
    print(step)
    # create minibatch here
    data = np.zeros((BatchLength, SizeX * SizeY * SizeZ))
    label = np.zeros((BatchLength, 2))
    Images = np.zeros((BatchLength, SizeX, SizeY, SizeZ))
    # select random images from the samples
    Indices = random.sample(range(1, len(TrainLabels)), BatchLength)
    for ind in range(BatchLength):
      data[ind] = X[Indices[ind]]
      label[ind] = TrainLabels[Indices[ind]]
    if step == 1:
      # save example images at first run for report
      tf.summary.image('images', Images, max_outputs=50)
    # add data to summaries
    summary_op = tf.summary.merge_all()
    summary = sess.run([summary_op, optimizer],
               feed_dict={x: data, y: label})
    summary_writer.add_summary(summary[0], step)
    # evaluate on every 100th Step
    if (step % EvalFreq == 1):
      NumOk = 0
      # create test data
      NumberOfTestSamples = len(TestIds)
      data = np.zeros((1, SizeX * SizeY * SizeZ))
      for ind in range(NumberOfTestSamples):
        data[0] = X[TestIds[ind], :]
        TestLabels[ind]
        Res = sess.run(pred, feed_dict={x: data})
        if (Res[0][0] > Res[0][1]) and (TestLabels[ind][0] == 1):
          NumOk += 1
        elif (Res[0][1] > Res[0][0]) and (TestLabels[ind][1] == 1):
          NumOk += 1
      print "Independent test accuracy: " + str(float(NumOk) / float(NumberOfTestSamples))
      # save checkpoint
      saver.save(sess, 'model' + str(step))
    step += 1
  print("Optimization Finished!")
  print("Run the command line:\n"
      "--> tensorboard --logdir=/tmp/tensorflow_logs "
      "\nThen open http://127.0.1.1:6006/ into your web browser")
