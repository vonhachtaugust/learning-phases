#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
import numpy as np
import json

from src.physic import ising, models, ising_lgt
from src.neuralnetwork import feedforward, cnn
from src.visualization import visualization as v, animations as animate
from src.serialization import serialization as serial
from src.simulation.monte_carlo import MC2D, MC2D_recorded, RecordTemperatureStrategy, SamplingStrategy


def example1(model, batch_size):
    g = model.data(batch_size)
    net = feedforward.Feedforward(L)
    net.train_on_generator(g, 100, 40, 1)
    x, y = next(g)
    return y, net.predict_on_batch(x)

def example2(model):
    MC2D(model, model.init_random, 1, 10000)
    v.quiver(model)

def example3():
    model = ising_lgt.Ising_LGT((4, 4), 1)
    network = cnn.ConvNet(L)
    data = model.cnn_data(32)
    network.train_on_generator(data, 100, 100, 1)
    x, y = next(data)
    return y, network.predict_on_batch(x)


def predict_and_plot(model, net, T, mcs):
    r = serial.ArrayRecording()
    MC2D(model, model.init_ordered, T, mcs, r)

    y_pred = net.predict_on_batch(r.recording())
    v.scatter(y_pred, r.recording())

def main(*args):
    pass


if __name__ == '__main__':
    main(*sys.argv[1:])
