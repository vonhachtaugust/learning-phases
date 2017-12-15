#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for `ghost` package."""

import unittest
import numpy as np
import random as rand


class TestGhost(unittest.TestCase):
    """Tests for `ghost` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_state_from_magnetization(self):
        """Test something."""
        target_magnetization = 0
        size = 10
        print(
            "Trying to get as close as possible to %s" % target_magnetization)
        state = np.array(
            [1 if rand.random() > 0.5 else -1 for i in range(size)])
        print("Initial state %s with magnetization %s" % (state,
                                                          np.mean(state)))
        indices = [i for i in range(size)]
        print("Indice list %s" % indices)

        def current_magnetization(state):
            return np.mean(state)

        def minimum_resolution(size):
            return 2.0 / size

        def minimum(error, minimum_resolution):
            return error <= (minimum_resolution / 2.0)

        print("Minimum resolution %s" % minimum_resolution(size))

        err = np.abs(target_magnetization - current_magnetization(state))
        print("Initial error %s" % err)
        while len(indices) != 0:
            if minimum(err, minimum_resolution(size)):
                print("This is the minimum %s" % state)
                return

            idx = rand.randrange(0, len(indices))
            print("Randomed index %s" % idx)
            state_idx = indices.pop(idx)
            print("Indices left %s" % indices)
            state[state_idx] = -1 * state[state_idx]
            print("Changed state to %s" % state)
            new_err = np.abs(target_magnetization -
                             current_magnetization(state))
            print("Err %s vs new_err %s" % (err, new_err))
            if new_err < err:
                print("New error improved")
                err = new_err
            else:
                print("Error did not improve")
                state[state_idx] = -1 * state[state_idx]
        raise Exception(
            "Target magnetization %s unreachable" % target_magnetization)
