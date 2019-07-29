"""
License: Apache-2.0 
Code by Oliver Aurelius Ellison and Michael Edward Cruz of Boston University (2019) 
Adapted from code by Huadong Liao of Stanford University (2017)
E-mail: aurelius@bu.edu, mecruz@bu.edu
"""

import tensorflow as tf

flags = tf.app.flags


############################
#    hyper parameters      #
############################

# For separate margin loss
flags.DEFINE_float('m_plus', 0.9, 'the parameter of m plus')
flags.DEFINE_float('m_minus', 0.1, 'the parameter of m minus')
flags.DEFINE_float('lambda_val', 0.5, 'down weight of the loss for absent digit classes')

# for training
flags.DEFINE_integer('batch_size', 64, 'batch size')
flags.DEFINE_integer('epoch', 30, 'epoch')
flags.DEFINE_integer('iter_routing', 2, 'number of iterations in routing algorithm')
flags.DEFINE_boolean('mask_with_y', False, 'use the true label to mask out target capsule or not') 
flags.DEFINE_float('stddev', 0.01, 'stddev for W initializer')
flags.DEFINE_float('regularization_scale', 7.2, 'regularization coefficient for reconstruction loss, default to 7.2')


############################
#   environment setting    #
############################
flags.DEFINE_string('dataset', 'mnist', 'The name of dataset [mnist, fashion-mnist')
flags.DEFINE_boolean('is_training', True, 'train or predict phase')
flags.DEFINE_integer('num_threads', 2, 'number of threads of enqueueing examples')
flags.DEFINE_string('logdir', 'logdir', 'logs directory')
flags.DEFINE_integer('train_sum_freq', 200, 'the frequency of saving train summary(step)')
flags.DEFINE_integer('val_sum_freq', 500, 'the frequency of saving valuation summary(step)')
flags.DEFINE_integer('save_freq', 3, 'the frequency of saving model(epoch)')
flags.DEFINE_string('results', 'results', 'path for saving results')

############################
#   distributed setting    #
############################
#flags.DEFINE_integer('num_gpu', 1, 'number of gpus for distributed training')
#flags.DEFINE_integer('batch_size_per_gpu', 64, 'batch size on 1 gpu')
#flags.DEFINE_integer('thread_per_gpu', 2, 'Number of preprocessing threads per tower.')

cfg = tf.app.flags.FLAGS
# tf.logging.set_verbosity(tf.logging.INFO)
