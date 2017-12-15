#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import h5py

from ..utility.utils import path


class ArrayRecording(object):
    """
    Class for serializing numpy ndarrays.

    Each set of numpy arrays are stored with a unique key in the recording dictionary.
    Attributes are added in the attributes dictionary and are meant for recording parameters used to generate the numpy arrays.

    Constructor params -----
    @param filename : Name of the file to write to (without suffix). If no file with this name exists a new one is created when saving.
    @param path     : Relative path to data files i.e. /data/testdata/ (the absolute path is added automagically)

    @return         : Recording object which can either record and save numpy arrays or load numpy arrays from existing datafiles.

    NOTE: loading clears every dictionary, filename and path
    """
    default = 'recording'

    def __init__(self, filename=None, path=None):
        self.__recording = {}
        self.__attributes = {}
        self.__filename = filename
        self.__path = path

    @property
    def attributes(self):
        return self.__attributes

    @property
    def recording(self):
        return self.__recording

    @property
    def path(self):
        return self.__path

    @property
    def filename(self):
        return self.__filename

    def clear(self):
        self.__recording = {}
        self.__attributes = {}
        self.__filename = None
        self.__path = None

    def add_attribute(self, name, value):
        self.__attributes[name] = value

    def add_attributes(self, attribute_dict):
        for key in attribute_dict:
            self.__attributes[key] = attribute_dict[key]

    def add_record(self, item, record_name):
        if record_name not in self.recording:
            self.__recording[record_name] = np.array(item)
        else:
            self.__recording[record_name] = np.vstack(
                [self.recording[record_name], item])

    def set_relative_path(self, path):
        self.__path = path

    def set_filename(self, filename):
        self.__filename = ''.join([path(), self.path, filename, '.h5'])

    def save(self, filename=None):
        if filename is None:
            if self.filename is None:
                raise ValueError("Set filename first")
            filename = self.filename

        with h5py.File(filename, 'w') as h5:
            for key in self.recording:
                h5.create_dataset(key, data=self.recording[key])
            for key in self.attributes:
                h5.attrs.create(key, self.attributes[key])

    def load(self, dataset_names=None, filename=None):
        if filename is None:
            if self.filename is None:
                raise ValueError("Set filename first")
            filename = self.filename

        self.clear()
        with h5py.File(filename, 'r') as h5:
            if dataset_names is None:
                # If no specific dataset name is given load all.
                dataset_names = []
                for key in h5.keys():
                    dataset_names.append(key)
            try:
                for name in dataset_names:
                    self.__recording[name] = h5[name][:]
                for key in h5.attrs.keys():
                    self.__attributes[key] = h5.attrs.get(key)
            except (EOFError, StopIteration) as e:
                raise e.what()
