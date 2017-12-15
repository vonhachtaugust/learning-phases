#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

default_transient = 10000.0


class RecordTemperatureStrategy(object):
    """
    Monte carlo recording strategy for temperature intervals and steps.

    Constructor: ---
    @param Tmin       : Start temperature
    @param Tmax       : End temperature
    @param minorStep  : Smaller step size can be used when simulating around the critical temperature
    @param majorStep  : Otherwise, the step size can be quite rough
    @param limit      : +/- interval in which the minor step is used
    @param Tc         : Analytical expression of Tc (if exists) otherwise scalar where estimated Tc is.
    """

    def __init__(self, Tmin, Tmax, TminorStep, TmajorStep, limit, f):
        self.Tmin = Tmin
        self.Tmax = Tmax
        self.minorStep = TminorStep
        self.majorStep = TmajorStep
        self.limit = limit
        self.Tc = f
        print(
            "---------------- : Record Temperature Strategy : ----------------"
        )
        print("Start at %s and ending at %s" % (Tmin, Tmax))
        print(
            "Withing [%s, %s] the stepsize %s is used, otherwise the stepsize %s is used."
            % (f - limit, f + limit, TminorStep, TmajorStep))
        print("Others: %s, %s, %s" % (self.sample_start,
                                      self.sample_after_ratio, self.mod))
        print(" ---------------- : ---------------- : ---------------- ")

        def continues(self, T):
            return T < self.Tmax

        def minorStep(self, T):
            return T + self.minorStep

        def majorStep(self, T):
            return T + self.majorStep


class SamplingStrategy(object):
    """
    Monte carlo sampling strategy for selecting suitable samples.

    Constructor: ---
    @param transient   : Transient treshold to avoid non-equilibirium states
    @param mcs         : (monte-carlo) steps
    @param samples     : Number of samples to collect

    """

    def __init__(self, transient, mcs, samples, verbose=False):
        self.mcs = mcs
        self.transient = transient
        self.sample_after_ratio = 1.0 - (transient / mcs)
        self.sample_start = int(mcs * self.sample_after_ratio)
        self.samples = samples
        self.mod = np.round(mcs *
                            (1.0 - self.sample_after_ratio) / self.samples)
        if verbose:
            print("---------------- : Sampling Strategy : ----------------")
            print("Total number of steps to simulate: %s" % mcs)
            print(
                "Sampling starts after %s steps and %s samples are collected\nSelected uniformly distributed (default) at %s"
                % (self.sample_start, samples,
                   [i for i in range(self.sample_start, mcs) if self.this(i)]))
            print(" ---------------- : ---------------- : ---------------- ")

    def start(self, iteration):
        return iteration > self.sample_start

    def this(self, iteration):
        return iteration % self.mod == 0


def MC2D_recorded(model, init_as, file_id, recorder, rts, ss):
    T = rts.Tmin
    ID = file_id
    while rts.continues(T):
        print("Monte-carlo simulating at temperature %s" % T)
        MC2D(model, init_as, T, mcs=ss.mcs, recorder=recorder, ss=ss)

        filename = ''.join([model.name(), '_data_', str(ID)])
        recorder.set_filename(filename)
        recorder.add_attributes(model.parameters())
        print("Storing %s samples in file %s" % (ss.samples, filename))
        recorder.save()

        if (T > rts.Tc() - rts.limit) and (T < rts.Tc() + rts.limit):
            T = rts.minorStep(T)
        else:
            T = rts.majorStep(T)

        ID += 1
        recorder.clear()


def keep_flip(dE, T):
    return (dE < 0) or (np.random.rand() < np.exp(-dE / T))


def MC2D(model, init_as, T, mcs=50000, recorder=None, ss=None):
    """
    Monte-carlo simulation on a two-dimensional lattice.
    The simulation requires a physics model (ex. Ising model) having overridden the Model base class.

    @param model   : Object derived from the Model base class
    @param T       : Temperature of heat bath
    @param mcs     : (monte-carlo) steps
    @param recored : Object derived from serialization.ArrayRecording used to save snapshots of lattice

    The monte-carlo method modifies the internal lattice of model, thus the result is model.lattice
    """
    if mcs < default_transient:
        raise ValueError(
            "Monte carlo simulation should be larger than %s for reliable results"
            % mcs)

    np.random.seed()
    model.initialize(init_as)
    record = False

    if recorder is not None:
        if ss is None:
            raise ValueError("Recorder requires a sample strategy.")
        recorder.add_attribute('Temperature', T)
        recorder.add_attribute('Shape', model.shape)
        record = True

    # Metropolis algorithm
    for i in range(1, mcs + 1):
        x = np.random.randint(0, model.shape[0])
        y = np.random.randint(0, model.shape[1])

        # Previous energy
        previous_energy = model.energy_at_position(x, y)

        # New energy due to flip
        model.flip(x, y)
        new_energy = model.energy_at_position(x, y)

        # Delta E
        dE = new_energy - previous_energy
        if not keep_flip(dE, T):
            model.undo_flip(x, y)

        if record:
            if ss.start(i) and ss.this(i):
                recorder.add_record(
                    model.as_array('spin', 'float32').reshape(1, model.size),
                    'state')
