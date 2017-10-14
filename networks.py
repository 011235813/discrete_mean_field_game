import tensorflow as tf
from layers import *

def hidden2(vec_input, n_hidden1, n_hidden2, n_outputs, nonlinearity1, nonlinearity2):

    h1 = linear_layer(vec_input, n_hidden1, nonlinearity1, scope='fc1')
    h2 = linear_layer(h1, n_hidden2, nonlinearity2, scope='fc2')
    out = linear_layer(h2, n_outputs, nonlinearity=None, scope='out')

    return out


def r_net(state_input, action_input, f1=2, k1=5, f2=4, k2=3, d=15):
    """
    state_input - batch of states, assumed to be [batch_size, d]
    action_input - batch of transition matrices, assumed to be [batch_size, d, d]
    f1 - number of filters for first conv layer
    k1 - kernel size for first conv layer (height=width)
    f2 - number of filters for second conv layer
    k2 - kernel size for second conv layer (height=width)
    d - number of topics
    """
    action_input = tf.reshape(action_input, [-1,d,d,1])
    state_input = tf.reshape(state_input, [-1,d])
    # first convolutional layer with f1 filters, kernel k1, stride 1
    # e.g. with k1=5, padding=2, output size = (d + 2*2 - 5)/1 + 1 = d
    conv1 = tf.contrib.layers.conv2d(inputs=action_input, num_outputs=f1, kernel_size=k1, stride=1, padding="SAME", activation_fn=tf.nn.relu, scope='conv1')
    # second convolutional layer with f2 filters, kernel k2, stride 1, padding 1
    # e.g. with k2=3, padding=1, output_size = (d + 2*1 - 3)/1 + 1 = d
    conv2 = tf.contrib.layers.conv2d(inputs=conv1, num_outputs=f2, kernel_size=k2, stride=1, padding="SAME", activation_fn=tf.nn.relu, scope='conv2')
    # flatten to vector
    conv2_flat = tf.reshape(conv2, [-1, f2*d*d]) # batch_size x (f2*d*d)
    # feed conv output to linear layer before combining with state vector
    fc3 = tf.contrib.layers.fully_connected(inputs=conv2_flat, num_outputs=64, activation_fn=tf.nn.relu, scope='fc3') # batch_size x 64
    # for each sample in batch, concatenate conved action with state
    fc3_action = tf.concat([fc3, state_input], 1)
    # one more fc layer
    fc4 = tf.contrib.layers.fully_connected(inputs=fc3_action, num_outputs=64, activation_fn=tf.nn.relu, scope='fc4')
    fc5 = tf.contrib.layers.fully_connected(inputs=fc4, num_outputs=16, activation_fn=tf.nn.relu, scope='fc5')
    # final output layer without nonlinearity
    out = tf.contrib.layers.fully_connected(inputs=fc5, num_outputs=1, activation_fn=tf.nn.tanh, scope='out')

    return out

    
