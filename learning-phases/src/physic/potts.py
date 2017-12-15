#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

from .models import Lattice2D, bond_type

init_types = {'Random': 1, 'Ordered': 0}


class Potts(Lattice2D):
    def __init__(self,
                 shape2D,
                 bond_strength=1,
                 bond_type=bond_type['Ferromagnetic']):
        self.__observables = {'Energy': 0, 'Magnetization': 0}
        self.__bond_strength = bond_strength
        self.__bond_type = bond_type
        self.__parameters = {'J', self.J}
        super().__init__(shape2D)

    @property
    def J(self):
        return self.__bond_strength

    @property
    def observables(self):
        return self.__observables

    @property
    def magnetization(self):
        return np.mean(self.lattice)

    """
    Overrides name
    """

    def name(self):
        return 'Potts'

    """
    Overrides initialize
    """

    def initialize(self, init_as):
        pass

    """
    Overrides energy_at_position
    """

    def energy_at_position(self, x, y):
        # Periodic boundary conditions
        up = self.getIndexAbove(x)
        down = self.getIndexBelow(x)
        right = self.getIndexRightOf(y)
        left = self.getIndexLeftOf(y)

        return self.__Potts_site_energy(x, y, up, down, right, left)

    """
    Overrides flip
    """

    def flip(self, x, y):
        pass

    """
    Overrides undo_flip
    """

    def undo_flip(self, x, y):
        pass

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

    def __Potts_two_site_interaction_energy(self, x, y, up, down, right, left):
        return self.__bond_type * self.J * self.lattice[x][y].get('spin')
