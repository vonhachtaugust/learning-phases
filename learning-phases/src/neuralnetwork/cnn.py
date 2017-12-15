#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

from keras.layers import Conv2D, Dense, Dropout, Flatten
from keras import activations, initializers, optimizers
from keras import losses, metrics, regularizers

from .network import Network

# Input: 64 2 x 2 filters, no padding, unit stride, periodic boundary
# Activation: ReLu
# Hidden: fully, dropout
# Activation: ReLu
# Output: 2 neurons softmax


class ConvNet(Network):
    """

    A convolutional neural network.
    Machine learning phases of matter, Nature Physics
    https://www.nature.com/nphys/journal/v13/n5/pdf/nphys4035.pdf

    Constructor: ---
    @param input_shape   : 4D tensor

    """

    def __init__(self, L):
        super().__init__()
        self.model.add(
            Conv2D(
                input_shape=(2, L + 1, L + 1),
                filters=64,
                kernel_size=(2, 2),
                padding='valid',
                data_format='channels_first',
                activation=activations.relu,
                kernel_regularizer=regularizers.l2(0.01)))
        self.model.add(Flatten())
        self.model.add(
            Dense(
                units=64,
                activation=activations.relu,
                kernel_regularizer=regularizers.l2(0.01),
                input_dim=(L * L * 2)))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(
            2,
            activation=activations.softmax,
        ))
        self.model.compile(
            optimizer=optimizers.Adam(),
            loss=losses.categorical_crossentropy,
            metrics=[metrics.categorical_accuracy])
        print(
            "------------------ :Constructing convolutional neural network: ------------------"
        )
        print(
            "Input shape (2, %s, %s), 64 filters, (2, 2) kernels, (1,1) stride and no padding"
            % (L + 1, L + 1))
        print(
            "Architecture: Conv2D(Input shape) -> Dense(64) -> Dense(2) -> Softmax"
        )
        print(
            "Do not forget to transpose the conv2D filters when inspecting them."
        )
        print(
            "---------------------------------------------------------------------------------"
        )

    def recompile(self):
        self.model.compile(
            optimizer=optimizers.Adam(),
            loss=losses.categorical_crossentropy,
            metrics=[metrics.categorical_accuracy])
