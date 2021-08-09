"""
Created on Fri Nov 30 22:48:33 2018
Bayesian-optimized Gabor Networks
@author: Thanh Le
"""

import math
from keras.layers import BatchNormalization, MaxPool2D, Flatten, Dense, Activation
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import hdf5storage
from layer.gabor_layer_v4 import Gabor2D
from keras.utils import np_utils


class BayesOptimizedGNN:
    def __init__(self, num_block=3, 
                 learning_rate=0.001,
                 beta_1=0.9, 
                 beta_2=0.999):

        self.num_block = num_block
        self.init_num_filter = round(16 / math.sqrt(self.num_block))
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.train_patches, self.train_labels, self.test_patches, self.test_labels = self.load_data()
        self.model = self.build_gabor_network()

    def load_data(self):
        train_data = hdf5storage.loadmat('./dataset/smethod/fold1/patches/train.mat')
        train_patches = train_data["train_patches"]
        train_labels = train_data["train_labels"]

        test_data = hdf5storage.loadmat('./dataset/smethod/fold1/patches/test.mat')
        test_patches = test_data["test_patches"]
        test_labels = test_data["test_labels"]

        train_patches = np.transpose(train_patches, (2, 0, 1))
        train_patches = np.expand_dims(train_patches, axis=3)
        test_patches = np.transpose(test_patches, (2, 0, 1))
        test_patches = np.expand_dims(test_patches, axis=3)

        # Convert the labels to one-hot vectors
        train_labels = np.squeeze(train_labels, axis=0)
        train_labels = np_utils.to_categorical(train_labels, 3)
        test_labels = np.squeeze(test_labels, axis=0)
        test_labels = np_utils.to_categorical(test_labels, 3)

        return train_patches, train_labels, test_patches, test_labels

    def build_gabor_network(self):
        model = Sequential()

        # Stage 1
        model.add(Gabor2D(1 * self.init_num_filter, kernel_size=15, padding='same',
                          input_shape=(128, 134, 1), data_format='channels_last'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        for x in range(self.num_block - 1):
            model.add(Gabor2D(1 * self.init_num_filter, kernel_size=15, padding='same'))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))

        # Stage 2
        for x in range(self.num_block):
            model.add(Gabor2D(2 * self.init_num_filter, kernel_size=7, padding='same'))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))

        # Stage 3 
        for x in range(self.num_block):
            model.add(Gabor2D(4 * self.init_num_filter, kernel_size=3, padding='same'))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))

        # Add a dense layer
        model.add(Flatten())
        model.add(Dense(3, activation='softmax'))

        # Compile the model
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2),
                      metrics=['accuracy'])
        return model

    def train(self):
        early_stopping = EarlyStopping(monitor='val_acc', patience=10, mode='max')
        # checkpoint = ModelCheckpoint('./checkpoints/model.h5', monitor='val_acc', verbose=1,
        #                              save_best_only=True, mode='max')

        self.model.fit(self.train_patches, self.train_labels,
                       batch_size=64,
                       epochs=80,
                       verbose=2,
                       validation_data=(self.test_patches, self.test_labels),
                       callbacks=[early_stopping])

    def evaluate(self):
        self.train()
        result = self.model.evaluate(self.test_patches, self.test_labels,
                                     batch_size=64,
                                     verbose=1)
        return result  # return the accuracy and the loss
