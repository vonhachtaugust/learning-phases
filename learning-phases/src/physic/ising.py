#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

from .models import Model, ScalarLattice2D, bond_type, State

init_type = {
    'Random': lambda: 1 if np.random.rand() > 0.5 else -1,
    'OrderedUp': lambda: 1,
    'OrderedDown': lambda: -1
}


class Ising(ScalarLattice2D):
    """
    Ising model on the 2D lattice.

    Constructor: ---
    @param shape2D      : Lattice width and height
    @param magnetic_t   : Bond energy (etc. ferro or anti-ferro)

    This is a model that can be feed to the Monte carlo function to simulate it in a temperature bath.

    """

    def __init__(self,
                 shape2D,
                 bond_strength,
                 bond_type=bond_type['Ferromagnetic']):
        self.__observables = {'Energy': 0, 'Magnetization': 0}
        self.__bond_strength = bond_strength
        self.__bond_type = bond_type
        self.__parameters = {'J', self.J}
        super().__init__(shape2D, State)

    @property
    def J(self):
        return self.__bond_strength

    @property
    def observables(self):
        return self.__observables

    @property
    def magnetization(self):
        return np.mean(self.lattice)

    @property
    def init_ordered(self):
        return init_type['OrderedDown'] if np.random.rand(
        ) > 0.5 else init_type['OrderedUp']

    @property
    def init_random(self):
        return init_type['Random']

    @property
    def energy(self):
        return self.operate_forall(lambda a, b: a + b,
                                   lambda x, y: self.energy_at_position(x, y))

    """
    Overrides name
    """

    def name(self):
        return 'Ising'

    """
    Overrides initialize
    """

    def initialize(self, init_as):
        if init_as == 'Random':
            self.set_parameter_forall('spin', init_type['Random'])
        elif init_as == 'Ordered':
            if np.random.rand() >= 0.5:
                self.set_parameter_forall('spin', init_type['OrderedUp'])
            else:
                self.set_parameter_forall('spin', init_type['OrderedDown'])
        else:
            raise ValueError("Unknown initialization type")

    """
    Overrides energy_at_position
    """

    def energy_at_position(self, x, y):
        # Periodic boundary conditions
        up = self.getIndexAbove(x)
        down = self.getIndexBelow(x)
        right = self.getIndexRightOf(y)
        left = self.getIndexLeftOf(y)

        return self.__Ising_two_site_interaction_energy(
            x, y, up, down, right, left)

    """
    Overrides flip
    """

    def flip(self, x, y):
        self.set(x, y, 'spin', -1 * self.lattice[x][y].get('spin'))

    """
    Overrides undo_flip
    """

    def undo_flip(self, x, y):
        self.flip(x, y)

    """
    Overrides parameters
    """

    def parameters(self):
        return self.__parameters

    """
    Overrides compute_observables
    """

    def compute_observables(self):
        self.observables['Magnetization'] = self.magnetization

    def data(self, batch_size):
        """

        'The generator is expected to loop over its data indefinitely', Keras

        """
        while True:
            x_train = np.full([batch_size, self.size], 0)
            y_train = np.full([batch_size, 2], 0)
            for i in range(batch_size):
                r = np.random.random()
                if r > 0.5:
                    x_train[i] = self.__a_ground_state()
                    y_train[i] = [1, 0]
                else:
                    x_train[i] = self.__an_infinity_state()
                    y_train[i] = [0, 1]
            yield x_train, y_train

    def __Ising_two_site_interaction_energy(self, x, y, up, down, right, left):
        return self.__bond_type * self.J * self.lattice[x][y].get('spin') * (
            self.lattice[up][y].get('spin') + self.lattice[down][y]
            .get('spin') + self.lattice[x][right].get('spin') +
            self.lattice[x][left].get('spin'))

    def __a_ground_state(self):
        return np.ones(
            self.size) if np.random.random() > 0.5 else -1 * np.ones(
                self.size)

    def __an_infinity_state(self):
        return np.array(
            [1 if np.random.random() > 0.5 else -1 for i in range(self.size)])
