import tensorflow as tf
import numpy as np

def linear_layer(vec_input, num_nodes, nonlinearity, scope):
    if nonlinearity == None:
        nonlinearity = tf.identity

    with tf.variable_scope(scope):
        h = tf.contrib.layers.fully_connected(inputs=vec_input, num_outputs=num_nodes, activation_fn=nonlinearity)

    return h
