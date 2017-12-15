#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

from .models import Model, VectorLattice2D, bond_type, State
from ..simulation.monte_carlo import SamplingStrategy

init_type = {
    'Random': lambda: 1 if np.random.rand() > 0.5 else -1,
    'OrderedUp': lambda: 1,
    'OrderedDown': lambda: -1
}


class Ising_LGT(VectorLattice2D):
    """
    Ising lattice gauge theory on the 2D lattice

    Constructor: ---
    @param shapeD      : Lattice width and height
    @param magnetic_t  : Bond energy (etc. ferro or anti-ferro)

    This is a model that can be feed to the Monte carlo function to simulate it in a temperature bath.

    """

    def __init__(self,
                 shape2D,
                 bond_strength,
                 bond_type=bond_type['Ferromagnetic']):
        super().__init__(shape2D, 2, State)
        self.__observables = {'Energy': 0}
        self.__bond_strength = bond_strength
        self.__bond_type = bond_type
        self.__parameters = {'J', self.J}

    @property
    def J(self):
        return self.__bond_strength

    @property
    def observables(self):
        return self.__observables

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
        return 'Ising_LGT'

    """
    Overrides initialize
    """

    def initialize(self, init_as, array=None):
        if init_as == 'Random':
            self.set_forall('spin', init_type['Random'])
        elif init_as == 'Ordered':
            if np.random.rand() > 0.5:
                self.set_forall('spin', init_type['OrderedUp'])
            else:
                self.set_forall('spin', init_type['OrderedDown'])
        elif init_as == 'Array':
            self.set_forall('spin', lambda d, i, j: array[d][i][j])
        else:
            raise ValueError("Unknown initialization type")

    """
    Overrides energy_at_position
    """

    def energy_at_position(self, x, y):
        # Periodic boundary conditions, x is row with positive down, y is column with positive right
        below = self.getIndexBelow(x)
        rightOf = self.getIndexRightOf(y)

        return self.__Ising_LGT_plaquette_energy(x, y, below, rightOf)

    """
    Overrides flip
    """

    def flip(self, x, y):
        which = 0 if np.random.rand() > 0.5 else 1
        self.get(which, x, y).set('spin', -1 * self.get(which, x, y))

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
        pass

    def __Ising_LGT_plaquette_energy(self, x, y, below, rightOf):
        above = self.get(0, x, y).get('spin')
        under = self.get(0, below, y).get('spin')
        left = self.get(1, x, y).get('spin')
        right = self.get(1, x, rightOf).get('spin')

        return self.__bond_type * self.J * above * under * left * right

    def cnn_gs_data(self, batch_size):
        while True:
            x_train = np.full((batch_size, self.dim, self.shape2D[0] + 1,
                               self.shape2D[1] + 1), 0)
            y_train = np.full((batch_size, 2), 0)

            for i in range(batch_size):
                if np.random.rand() > 0.5:
                    x_train[i][:self.dim, :-1, :-1] = self.__a_ground_state()
                    y_train[i] = [1, 0]
                else:
                    x_train[i][:self.dim, :-1, :
                               -1] = self.__an_infinity_state()
                    y_train[i] = [0, 1]

            self.__apply_periodic_boundary(x_train)

            yield x_train, y_train

    def cnn_data(self, batch_size):
        """
        Data for cnn model, data shape = 4D tensor (batch_size, channels, rows, cols)

        Abbreviations,
        gs : ground states

        """
        while True:
            x_train = np.full((batch_size, self.dim, self.shape2D[0] + 1,
                               self.shape2D[1] + 1), 0)
            y_train = np.full((batch_size, 2), 0)

            # Another way to randomise the batch, shuffle!
            #p = np.random.permutation(batch_size)
            gs_indices = []
            for i in range(batch_size):
                if np.random.rand() > 0.5:
                    # Delay ground states, they are obtained all at once
                    gs_indices.append(i)
                    y_train[i] = [1, 0]
                else:
                    x_train[i][:self.dim, :-1, :
                               -1] = self.__an_infinity_state()
                    y_train[i] = [0, 1]

            if gs_indices != []:
                self.__fill_gs(x_train, gs_indices)
            self.__apply_periodic_boundary(x_train)

            yield x_train, y_train

    def __fill_gs(self, x_train, gs_indices):
        iterations = len(gs_indices) * self.size2D
        ss = SamplingStrategy(int(iterations / 2), iterations, len(gs_indices))
        gs = self.__a_ground_state()
        j = 0

        for i in range(1, iterations + 1):
            x = np.random.randint(0, self.shape2D[0])
            y = np.random.randint(0, self.shape2D[1])

            above = self.getIndexAbove(x)
            leftOf = self.getIndexLeftOf(y)

            gs[0][x][y] = -1 * gs[0][x][y]
            gs[1][x][y] = -1 * gs[1][x][y]
            gs[0][x][leftOf] = -1 * gs[0][x][leftOf]
            gs[1][above][y] = -1 * gs[1][above][y]

            if ss.start(i) and ss.this(i):
                x_train[gs_indices[j]][:self.dim, :-1, :-1] = gs
                j += 1

    def __apply_periodic_boundary(self, x_train):
        for i in range(len(x_train)):
            x_train[i][:self.dim, :, -1] = x_train[i][:self.dim, :, 0]
            x_train[i][:self.dim, -1, :] = x_train[i][:self.dim, 0, :]

    def __a_ground_state(self):
        arr = self.__an_ordered_array()
        return np.array([arr for _ in range(self.dim)])

    def __an_infinity_state(self):
        return np.array([self.__random_array() for _ in range(self.dim)])

    def __random_array(self, dtype=None):
        """
        Scale and shift: [0, 1] -> [-1, 1]

        """
        return (np.random.randint(2, size=self.shape2D) * 2) - 1

    def __an_ordered_array(self, dtype=None):
        return -1 * np.ones(
            shape=self.shape2D,
            dtype=dtype) if np.random.rand() > 0.5 else np.ones(
                shape=self.shape2D, dtype=dtype)
