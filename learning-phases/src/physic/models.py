#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import numpy as np

bond_type = {'Ferromagnetic': -1, 'Antiferromagnetic': 1}


class State(object):
    """

    Represents a physical state.

    Constructor: ---
    @param spin : Physical spin of the state

    """

    def __init__(self, spin=None):
        self.__spin = spin
        self.__set_params = {'spin': self.set_spin}
        self.__get_params = {'spin': self.get_spin}

    def set(self, parameter, value):
        self.__set_params[parameter](value)

    def get(self, parameter):
        return self.__get_params[parameter]()

    def set_spin(self, value):
        self.__spin = value

    def get_spin(self):
        return self.__spin

    def __add__(self, other):
        if isinstance(other, self.__class__):
            return self.get('spin') + other.get('spin')

    def __radd__(self, other):
        if isinstance(other, int):
            return self.get('spin') + other

    def __mul__(self, other):
        if isinstance(other, int):
            return self.get('spin') * other
        if isinstance(other, self.__class__):
            return self.get('spin') * other.spin

    __rmul__ = __mul__

    def __repr__(self):
        return self.__str__()

    def __float__(self):
        return float(self.__spin)

    def __int__(self):
        return int(self.__spin)

    def __str__(self):
        return str(self.get('spin'))


class Model(ABC):
    """
    Abstract base class of a physics model.

    Please implement all  methods.
    """

    @abstractmethod
    def name(self):
        raise NotImplementedError("Override me")

    @abstractmethod
    def initialize(self, init_as):
        raise NotImplementedError("Override me")

    @abstractmethod
    def energy_at_position(self, x, y):
        raise NotImplementedError("Override me")

    @abstractmethod
    def flip(self, x, y):
        raise NotImplementedError("Override me")

    @abstractmethod
    def undo_flip(self, x, y):
        raise NotImplementedError("Override me")

    @abstractmethod
    def parameters(self):
        raise NotImplementedError("Override me")

    @abstractmethod
    def compute_observables(self):
        raise NotImplementedError("Override me")


class TwoDimensional:
    def __init__(self, shape2D):
        self.__shape2D = shape2D

    @property
    def shape2D(self):
        return self.__shape2D

    @property
    def size2D(self):
        return np.prod(self.__shape2D)

    """
    For periodic boundary conditions.

    """

    def getIndexAbove(self, index):
        return self.correctPosition(index - 1, self.shape2D[0])

    def getIndexBelow(self, index):
        return self.correctPosition(index + 1, self.shape2D[0])

    def getIndexRightOf(self, index):
        return self.correctPosition(index + 1, self.shape2D[1])

    def getIndexLeftOf(self, index):
        return self.correctPosition(index - 1, self.shape2D[1])

    def correctPosition(self, index, limit):
        return (index + limit) % limit


class ScalarLattice2D(Model, TwoDimensional):
    """
    Base class of a physics model on the 2D scalar lattice.

    Constructor: ---
    @param shape2D : lattice width and height
    """

    def __init__(self, shape2D, Obj):
        super().__init__(shape2D)
        self.__lattice = np.array([Obj()
                                   for _ in range(np.prod(shape2D))]).reshape(
                                       shape2D[0], shape2D[1])

    def get(self, x, y):
        return self.__lattice[x][y]

    def set(self, x, y, parameter, value):
        self.__lattice[x][y].set(parameter, value)

    def set_parameter_forall(self, parameter, function, *args):
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self.__lattice[i][j].set(parameter, function(*args))

    def operate_forall(self, operator, function, *args):
        result = function(0, 0, *args)
        for i in range(1, self.shape[0]):
            for j in range(self.shape[1]):
                if (i, j) == (0, 0):
                    continue
                result = operator(result, function(i, j, *args))
        return result

    @property
    def lattice(self):
        return self.__lattice

    def as_grid(self, parameter, dtype=None):
        return np.array(
            [[obj.get(parameter) for obj in row] for row in self.__lattice],
            dtype=dtype)

    def as_array(self, parameter, dtype=None):
        return self.as_grid(parameter, dtype).reshape((1, self.size))

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = [[str(e) for e in row] for row in self.__lattice]
        lens = [max(map(len, col)) for col in zip(*s)]
        fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
        table = [fmt.format(*row) for row in s]
        return '\n'.join(table)


class VectorLattice2D(Model, TwoDimensional):
    """

    Base class of a physics model on the 2D vector lattice

    Constructor: ---
    @param shape2D    : Lattice width and height
    @param dimension  : Vector dimension

    """

    def __init__(self, shape2D, dimension, Obj):
        super().__init__(shape2D)
        self.__dim = dimension
        self.__lattice = np.array([
            np.array([Obj() for _ in range(np.prod(shape2D))]).reshape(
                shape2D[0], shape2D[1]) for _ in range(dimension)
        ])

    @property
    def lattice(self):
        return self.__lattice

    @property
    def dim(self):
        return self.__dim

    def get(self, d, x, y):
        return self.__lattice[d][x][y]

    def get_all(self, x, y):
        return np.array([self.get(d, x, y) for d in range(self.dim)])

    def get_as_array(self, d, parameter, dtype=None):
        arr = np.zeros(shape=self.shape2D, dtype=dtype)
        for i in range(self.shape2D[0]):
            for j in range(self.shape2D[1]):
                arr[i][j] = self.get(d, i, j).get(parameter)
        return arr

    def set(self, d, x, y, parameter, value):
        self.__lattice[d][x][y].set(parameter, value)

    def set_forall(self, parameter, function, *args):
        for i in range(self.shape2D[0]):
            for j in range(self.shape2D[1]):
                for d in range(self.dim):
                    self.set(d, i, j, parameter, function(d, i, j, *args))

    def operate_forall(self, operator, function, *args):
        result = function(0, 0, *args)
        for i in range(self.shape2D[0]):
            for j in range(self.shape2D[1]):
                if (i, j) == (0, 0):
                    continue
                result = operator(result, function(i, j, *args))
        return result

    def as_grid(self, parameter, dtype=None):
        return np.array(
            [self.get_as_array(d, 'spin') for d in range(self.dim)])

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = [[str(self.get_all(i, j)) for j in range(self.shape2D[1])]
             for i in range(self.shape2D[0])]
        lens = [max(map(len, col)) for col in zip(*s)]
        fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
        table = [fmt.format(*row) for row in s]
        return '\n'.join(table)
