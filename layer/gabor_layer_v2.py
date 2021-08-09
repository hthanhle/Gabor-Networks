"""
Created on Fri Nov 30 22:48:33 2018
Gabor layer
@author: Thanh Le
"""
from keras import backend as K
from keras import activations, regularizers, initializers, constraints
from keras.engine.topology import Layer
from keras.utils import conv_utils
import tensorflow as tf
import numpy as np


class Gabor2D(Layer):
    def __init__(self, no_filters,
                 kernel_size,  # must be a scalar integer 
                 strides=(1, 1),
                 padding='valid',
                 data_format='channels_last',
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=False,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 kernel_regularizer=None,
                 **kwargs):

        super(Gabor2D, self).__init__(**kwargs)
        self.no_filters = no_filters
        self.kernel_size = kernel_size
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = K.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        self.input_channels = input_shape[channel_axis]
        self.sigma = self.add_weight(name='sigma',
                                     shape=(self.input_channels, self.no_filters),
                                     initializer=initializers.TruncatedNormal(mean=5.0, stddev=1.5),
                                     trainable=True)
        self.theta = self.add_weight(name='theta',
                                     shape=(self.input_channels, self.no_filters),
                                     initializer=initializers.RandomUniform(minval=0.0, maxval=1.0),
                                     trainable=True)
        self.lamda = self.add_weight(name='lambda',
                                     shape=(self.input_channels, self.no_filters),
                                     initializer=initializers.TruncatedNormal(mean=5.0, stddev=1.5),
                                     trainable=True)
        self.gamma = self.add_weight(name='gamma',
                                     shape=(self.input_channels, self.no_filters),
                                     initializer=initializers.TruncatedNormal(mean=1.5, stddev=0.4),
                                     trainable=True)
        self.psi = self.add_weight(name='psi',
                                   shape=(self.input_channels, self.no_filters),
                                   initializer=initializers.RandomUniform(minval=0.0, maxval=1.0),
                                   trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.no_filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        super(Gabor2D, self).build(input_shape)

    def call(self, inputs):
        gabor_kernels = tf.zeros([self.kernel_size, self.kernel_size,
                                  self.input_channels, 0], tf.float32)

        for filters in range(self.no_filters):
            # Take a column of 2D weights
            sigma = self.sigma[:, filters]
            theta = self.theta[:, filters]
            lamda = self.lamda[:, filters]
            gamma = self.gamma[:, filters]
            psi = self.psi[:, filters]

            sigma_x = sigma
            sigma_y = sigma / gamma

            half_size = np.floor(self.kernel_size / 2)
            y, x = np.mgrid[-half_size: half_size + 1, -half_size: half_size + 1]

            x_kernel = np.empty((self.kernel_size, self.kernel_size, 0))
            y_kernel = np.empty((self.kernel_size, self.kernel_size, 0))

            for channels in range(self.input_channels):
                x_kernel = np.dstack((x_kernel, x))  # 3D Numpy arrays
                y_kernel = np.dstack((y_kernel, y))

            rot_x_kernel = x_kernel * tf.cos(theta - np.pi) + y_kernel * tf.sin(theta - np.pi)
            rot_y_kernel = -x_kernel * tf.sin(theta - np.pi) + y_kernel * tf.cos(theta - np.pi)

            gabor = tf.zeros(y_kernel.shape, dtype=np.float32)
            gabor = tf.exp(-0.5 * (rot_x_kernel ** 2 / sigma_x ** 2 + rot_y_kernel ** 2 / sigma_y ** 2))
            gabor /= 2 * np.pi * sigma_x * sigma_y
            gabor *= tf.cos(2 * np.pi / lamda * rot_x_kernel + psi)

            gabor = tf.expand_dims(gabor, axis=3)

            gabor_kernels = tf.concat([gabor_kernels, gabor], axis=3)

        self.gabor_kernels = gabor_kernels
        outputs = K.conv2d(inputs,
                           self.gabor_kernels,
                           strides=self.strides,
                           padding=self.padding,
                           data_format=self.data_format,
                           dilation_rate=self.dilation_rate)

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):

        kernel_size_tuple = conv_utils.normalize_tuple(self.kernel_size, 2,
                                                       'kernel_size_tuple')
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []

            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    kernel_size_tuple[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)

            return (input_shape[0],) + tuple(new_space) + (self.no_filters,)

        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    kernel_size_tuple[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)

            return (input_shape[0], self.no_filters) + tuple(new_space)
