#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
import itertools

from ..utility.utils import get_filenames

marks = ['.', 'o', '*', 's']
colors = ['b', 'g', 'r', 'k']
markers = []

for mark in marks:
    for color in colors:
        markers.append(''.join([color, mark]))


def plot_magnetization(recording, path_to_files):
    fig = plt.figure()
    ax = fig.gca()
    ax.set_xlim(0, 20000)
    ax.set_ylim(-1.1, 1.1)
    filenames = get_filenames(path_to_files)
    plots = [plt.plot([], [], '.')[0] for _ in range(len(filenames))]

    for i, filename in enumerate(filenames):
        recording.load(['magnetization'], filename)
        dataset = recording.recording['magnetization']
        x = np.arange(0, dataset.shape[0])
        plots[i].set_data(x, dataset)


def temperature_vs_magnetization(recording, path_to_files):
    fig = plt.figure()
    ax = fig.gca()
    ax.set_xlim(0.9, 3.5)
    ax.set_ylim(-1.1, 1.1)
    plot = plt.plot([], [], '.')[0]

    filenames = get_filenames(path_to_files)

    for i, filename in enumerate(filenames):
        recording.load(['state'], filename)
        dataset = recording.recording['state']
        T = recording.attributes['Temperature']
        J = recording.attributes['J']

        m = np.array([np.mean(item) for item in dataset])
        t = np.array([T / J for _ in range(dataset.shape[0])])
        if i == 0:
            plot.set_data(t, m)
        else:
            plot.set_data(
                np.vstack([plot.get_xdata(), t]),
                np.vstack([plot.get_ydata(), m]))


output_modes = {
    'first_layer': lambda net, data: net.first_layer_output(data),
    'output_layer': lambda net, data: net.predict_on_batch(data)
}


def plot_output(model,
                net,
                recording,
                path_to_files,
                outputs,
                mode='first_layer',
                figure=1):
    fig = plt.figure(figure)
    plots = [
        plt.plot([], [], ''.join([colors[i], '.']))[0] for i in range(outputs)
    ]
    filenames = get_filenames(path_to_files)

    for i, filename in enumerate(filenames):
        recording.load(['state'], filename)
        dataset = recording.recording['state']
        m = [np.mean(state) for state in dataset]
        y_pred = output_modes[mode](net, dataset)
        for j, plot in enumerate(plots):
            if i == 0:
                plot.set_data(m, y_pred[:, j])
            else:
                plot.set_data(
                    np.vstack([plot.get_xdata(), m]),
                    np.vstack([plot.get_ydata(), y_pred[:, j]]))

    recording.clear()


def show(xlabel='x', ylabel='y', xlim=[-1.02, 1.02], ylim=[-30, 30], figure=1):
    fig = plt.figure(figure)
    ax = fig.gca()
    ax.grid(True)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()


def plot(data):
    plt.plot(data)
    plt.ylim([-1, 1])
    plt.show()


def scatter(neuron_data, states):
    m = [np.mean(state) for state in states]
    for i in range(neuron_data.shape[1]):
        plt.plot(m, neuron_data[:, i], markers[i])


def quiver(model):
    X, Y = np.meshgrid(
        np.arange(0, model.shape[0]), np.arange(0, model.shape[1]))
    U = 0 * model.as_grid('spin')
    V = model.as_grid('spin')

    P = plt.quiver(X, Y, U, V, pivot='mid')
    plt.tight_layout()
    plt.show()
