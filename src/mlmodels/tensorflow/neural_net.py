import tensorflow as tf
from src.settings import data_path
import src.loader as loader

def entry(configs, modelname):
  config = None
  for cfg in configs:
    if modelname == cfg.name:
      config = cfg
  if config is None:
    raise Exception('Model not found in config')
  task(config)

def task(config):
  dwi_data = loader.loadData("{}/{}/".format(data_path, config.name))
  ground_truth = loader.loadParams("{}/{}/{}.params".format(data_path, config.name, config.name))

  n = len(dwi_data)
  nparams = len(ground_truth[0])

  inputs = tf.placeholder(tf.float32, [1, n], name="X")
  outputs = tf.placeholder(tf.float32, [None, nparams], name="Yhat")

  W = tf.Variable(tf.zeros([n, 1]), dtype=tf.float32, name="W")
  y = tf.matmul(inputs, W)

  learning_rate = tf.Variable(0.5, trainable=False)
  cost_op = tf.reduce_mean(tf.pow(y - outputs, 2))
  train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_op)

  tolerance = 1e-3
  last_cost = 0
  alpha = 0.4
  max_epochs = 50000

  sess = tf.Session()
  print "Beginning Training"

  with sess.as_default():
    init = tf.initialize_all_variables()
    sess.run(init)
    sess.run(tf.assign(learning_rate, alpha))
    while True:
      sess.run(train_op, feed_dict={inputs: dwi_data, outputs: ground_truth})

      if epochs % 100 == 0:
        cost = sess.run(cost_op, feed_dict={inputs: dwi_data, outputs: ground_truth})
        print "Epoch: %d - Error: %.4f" %(epochs, cost)

        if abs(last_cost - cost) < tolerance or epochs > max_epochs:
          print "converged"
          break
        last_cost = cost

      epochs += 1
    w = W.eval()
    print "w= ", w
    print "Test Cost =", sess.run(cost_op, feed_dict={inputs: dwi_data, outputs: ground_truth})
