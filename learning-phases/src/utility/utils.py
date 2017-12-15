#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from os import listdir
from os.path import isfile, join


def path():
    """
    Returns the absolute path of the parent directory
    """
    return os.path.realpath('..')


def get_filenames(path_to_files):
    """
    Returns a list of absolute paths to each file in a directory.
    """
    absolute_path = ''.join([os.path.realpath('..'), path_to_files])
    return [
        ''.join([absolute_path, f]) for f in listdir(absolute_path)
        if isfile(join(absolute_path, f))
    ]


#def state_from_magnetization(size, target_magnetization):
#    if target_magnetization < -1 and target_magnetization > 1:
#        raise ValueError(
#            "Target magnetization can only be in the range [-1, 1]")
#    state = an_infinity_state(size)
#    indices = [i for i in range(size)]
#
#    # Initial error
#    err = np.abs(target_magnetization - current_magnetization(state))
#    while len(indices) != 0:
#        # Check new state for minimum
#        if minimum(err, minimum_resolution(size)):
#            return state
#
#        # Pop random state and flip its spin
#        idx = np.random.randrange(0, len(indices))
#        state_idx = indices.pop(idx)
#        state[state_idx] = -1 * state[state_idx]
#
#        # Compute new error and check for improvement
#        new_err = np.abs(target_magnetization - current_magnetization(state))
#        if new_err < err:
#            err = new_err
#        else:
#            state[state_idx] = -1 * state[state_idx]
#    raise Exception(
#        "Target magnetization %s unreachable" % target_magnetization)
