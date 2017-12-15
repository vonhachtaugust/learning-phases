#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from ..utility.utils import get_filenames

marks = ['.', 'o', '*', 's']
colors = ['b', 'g', 'r', 'k']
markers = []

for mark in marks:
    for color in colors:
        markers.append(''.join([color, mark]))


def animate_learned_critical_temperature(model, net, recording, path_to_files,
                                         outputs):
    batch_generator = model.data(32)

    fig = plt.figure()
    ax = fig.gca()
    ax.set_xlim(1, 3.5)
    ax.set_ylim(0, 1)
    plots = [plt.plot([], [], '.')[0] for _ in range(outputs)]
    filenames = get_filenames(path_to_files)

    def animate():
        for i, filename in enumerate(filenames):
            g = recording.load(filename)
            dataset = next(g)
            T = recording.attributes['Temperature']
            J = recording.attributes['J']
            #x = np.array([T / J for _ in range(dataset.shape[0])])
            y_pred = net.predict_on_batch(dataset)
            for j, plot in enumerate(plots):
                if i == 0:
                    plot.set_data(T / J, np.mean(y_pred[:, j]))
                else:
                    plot.set_data(
                        np.vstack([plot.get_xdata(), T / J]),
                        np.vstack([plot.get_ydata(),
                                   np.mean(y_pred[:, j])]))
        ax.relim()
        ax.autoscale_view(True, True, True)
        return plots

    def updatefig(*args):
        net.train_on_generator(batch_generator, 20, 1)
        return animate()

    animate()
    ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)
    plt.show()


def animate_training(model,
                     net,
                     recording,
                     path_to_files,
                     outputs,
                     output_file=None,
                     plot=False):
    batch_generator = model.data(32)

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    fig = plt.figure()
    ax = fig.gca()
    ax.grid(True)
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-30, 30)
    ax.set_xlabel('m(x)')
    ax.set_ylabel('Wx + b')
    plots = [plt.plot([], [], '.')[0] for _ in range(outputs)]
    filenames = get_filenames(path_to_files)

    sub = fig.add_subplot(111)
    fig.subplots_adjust(top=0.85)

    def animate(net, recording):
        for i, filename in enumerate(filenames):
            recording.load(['state'], filename)
            dataset = recording.recording['state']
            #y_pred = net.predict_on_batch(dataset)
            m = [np.mean(state) for state in dataset]
            y_pred = net.first_layer_output(dataset)
            for j, plot in enumerate(plots):
                if i == 0:
                    plot.set_data(m, y_pred[:, j])
                else:
                    plot.set_data(
                        np.vstack([plot.get_xdata(), m]),
                        np.vstack([plot.get_ydata(), y_pred[:, j]]))

    def animate_batch(net, batch_generator):
        dataset = next(batch_generator)[0]
        m = [np.mean(state) for state in dataset]
        y_pred = net.first_layer_output(dataset)
        for i, plot in enumerate(plots):
            plot.set_data(m, y_pred[:, i])

    def updatefig(*args):
        net.train_on_generator(batch_generator, 100, 1, 1)
        sub.set_title('Loss: {},  Accuracy: {}'.format(net.metrics[0],
                                                       net.metrics[1]))
        #animate_batch(net, batch_generator)
        animate(net, recording)
        return plots

    if output_file is None:
        output_file = 'output'
    ani = animation.FuncAnimation(
        fig, updatefig, interval=50, frames=500, blit=False, repeat=True)
    output = ''.join([output_file, '.mp4'])
    ani.save(output, writer=writer)
