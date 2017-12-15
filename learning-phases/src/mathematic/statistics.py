#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import scipy
from scipy import constants


def Boltzmann_numerator(energy, temperature):
    return np.exp(-energy / (temperature * constants.k))


def Boltzmann_denominator(energy_of_states, temperature):
    return np.sum([
        Boltzmann_numerator(energy, temperature) for energy in energy_of_states
    ])


def Boltzmann_distribution(energy_of_states, temperature):
    return [
        Boltzmann_numerator(energy, temperature) / Boltzmann_denominator(
            energy_of_states, temperature) for energy in energy_of_states
    ]
