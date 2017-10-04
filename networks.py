import tensorflow as tf
from layers import *

def hidden2(vec_input, n_hidden1, n_hidden2, n_outputs, nonlinearity1, nonlinearity2):

    h1 = linear_layer(vec_input, n_hidden1, nonlinearity1, scope='fc1')
    h2 = linear_layer(h1, n_hidden2, nonlinearity2, scope='fc2')
    out = linear_layer(h2, n_outputs, nonlinearity=None, scope='out')

    return out


def r_net(state_input, action_input):
    """
    Assumes action_input is 20x20
    """
    action_input = tf.reshape(action_input, [-1,20,20,1])
    # reduce to 8 x 10 x 10
    conv1 = tf.contrib.layers.conv2d(inputs=action_input, num_outputs=8, kernel_size=2, stride=2, padding="VALID", activation_fn=tf.nn.relu, scope='conv1')
    # reduce to 16 x 5 x 5
    conv2 = tf.contrib.layers.conv2d(inputs=conv1, num_outputs=16, kernel_size=2, stride=2, padding="VALID", activation_fn=tf.nn.relu, scope='conv2')
    conv2_flat = tf.reshape(conv2, [-1, 5*5*16])
    fc3 = linear_layer(conv2_flat, 200, tf.nn.relu, 'fc3')
    fc3_action = tf.concat([fc3, state_input], 1)
    fc4 = linear_layer(fc3_action, 200, tf.nn.relu, 'fc4')
    out = linear_layer(fc4, 1, None, 'out')

    return out

    
