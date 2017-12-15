#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

from keras.models import Sequential, Model
from keras.models import model_from_yaml

import keras.backend as K
from keras.callbacks import Callback

from ..utility.utils import path


def custom_uniform(shape, range=(-1, 1), name=None):
    """
    Example of custom function for keras.
    """
    min_, max_ = range
    return K.variable(
        np.random.uniform(low=min_, high=max_, size=shape), name=name)

    # Example usage:
    # net.add(Dense(10, input_dim=5, init=lambda shape,
    # name: custom_uniform(shape, (-10, 5), name)))


class TestCallback(Callback):
    """
    Example callback class for keras.
    """

    def __init__(self, generator):
        self.data_generator = generator

    def on_epoch_end(self, epoch, logs={}):
        x, y = next(self.data_generator)
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))


class Network(object):
    """
    Base class for the various neural networks.
    """

    def __init__(self):
        self.metrics = ()
        self.model = Sequential()

    def first_layer_output(self, x):
        weights = self.get_layer_weights(1)
        W = weights[0]
        b = weights[1]

        return np.dot(x, W) + b

    def predict_on_batch(self, x):
        return self.model.predict_on_batch(x)

    def get_weights(self, layer=None):
        if layer is None:
            return self.model.get_weights()
        return self.model.layers[layer].get_weights()

    def weight_shapes(self):
        return self.get_weights()[0].shape, self.get_weights()[1].shape

    def set_layer_weights(self, layer, weights):
        self.model.layers[layer].set_weights(
            [weights, self.get_weights(layer)[1]])

    def set_layer_bias(self, layer, bias):
        self.model.layers[layer].set_weights(
            [self.get_weights(layer)[0], bias])

    def set_layer_parameters(self, layer, weights, bias):
        self.model.layers[layer].set_weights([weights, bias])

    def get_layer_weights(self, layer):
        return self.model.get_layer(index=layer).get_weights()

    def train_once(self, data, batch_size):
        self.model.fit(data[0], data[1], epochs=1, batch_size=batch_size)

    def train_on_generator(self, training_set_generator, batches_per_epoch,
                           epochs, verbose):
        h = self.model.fit_generator(
            training_set_generator, batches_per_epoch, epochs, verbose=verbose)
        loss = h.history['loss'][epochs - 1]
        acc = h.history['categorical_accuracy'][epochs - 1]
        self.metrics = '{0:.3g}'.format(loss), '{0:.3g}'.format(acc)

    def save(self, relative_path, filename=None):
        if filename is None:
            filename = 'model'

        absolute_path = ''.join([path(), relative_path, filename])
        network_out = ''.join([absolute_path, '.yaml'])
        weight_out = ''.join([absolute_path, '.h5'])

        model_yaml = self.model.to_yaml()
        with open(network_out, 'w') as yaml_file:
            yaml_file.write(model_yaml)
        self.model.save_weights(weight_out)

    def load(self, relative_path, filename):
        absolute_path = ''.join([path(), relative_path, filename])
        network = ''.join([absolute_path, '.yaml'])
        weights = ''.join([absolute_path, '.h5'])

        with open(network, 'r') as yaml_file:
            loaded_model_yaml = yaml_file.read()

        self.model = model_from_yaml(loaded_model_yaml)
        self.model.load_weights(weights)
