#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

from keras.layers import Dense
from keras import activations, initializers, optimizers
from keras import losses, metrics, regularizers

from .network import TestCallback, Network

# Input size L: 30
# Cost: Cross-entropy
# Regularization: L2
# Optimizer: Adam
# Initialization: Normal distribution (zero mean unit variance)


class Feedforward(Network):
    """

    A fixed neural network.
    Machine learning phases of matter, Nature Physics
    https://www.nature.com/nphys/journal/v13/n5/pdf/nphys4035.pdf

    Constructor: ---
    @param input_length : input vector length

    """

    def __init__(self, input_length):
        super().__init__()
        self.model.add(
            Dense(
                2,
                activation=activations.sigmoid,
                kernel_initializer=initializers.RandomNormal(
                    mean=0.0, stddev=1.0),
                kernel_regularizer=regularizers.l2(0.001),
                input_dim=input_length))
        self.model.add(
            Dense(
                2,
                activation=activations.softmax,
                kernel_initializer=initializers.RandomNormal(
                    mean=0.0, stddev=1.0)))
        self.model.compile(
            optimizer=optimizers.Adam(),
            loss=losses.categorical_crossentropy,
            metrics=[metrics.categorical_accuracy])

    def recompile(self):
        self.model.compile(
            optimizer=optimizers.Adam(),
            loss=losses.categorical_crossentropy,
            metrics=[metrics.categorical_accuracy])
