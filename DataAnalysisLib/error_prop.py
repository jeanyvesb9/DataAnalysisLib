import warnings as _warnings
import typing as _typing
import inspect as _inspect


import numpy as _np
import matplotlib.pyplot as _plt
import pandas as _pd

import global_funcs as _gf
import global_enums as _ge
import dataset as _ds
from scipy import optimize as _opt




class errorProp(object):


    def __init__(self, data, fn, values):
        
        self._data = data
        self._fn = fn 
        self._values = values
    
    @property
    def data(self):
        return self._data

    @property 
    def fn(self):
        return self._fn

    @property
    def values(self):
        return self._values

    @fn.setter
    def fn(self, func):
        if hasattr(func, '__call__'): # checking  if is a function (checking if is callable, only way around duck typing)
            args = _inspect.getfullargspec(func).args  
            var_args, coef_vars = args[0], args[1] if len(args) == 2 else None
            if isinstance(var_args, list) and isinstance(coef_vars, list):
                self._fn = func
                self._var_args, self._coef_args = var_args, coef_vars
            else:
                self._fn = None
                _warnings.warn("Variables and coefficients must be lists")
        else:
            self._fn = None
            _warnings.warn(" func must be callable")


    @data.setter
    def data(self, value = None):
        if isinstance(value, _ds.Dataset):
            self._data = value
        elif isinstance(value, list):
            value = []
            for data in value:   # making big array with all dataset as columns, this form is convenient to evaluate error propagation
                value[0] = _np.array([_ds.Dataset(data).v if not isinstance(data, _ds.Dataset) else data.v for data in value]).transpose()
                value[1] = _np.array([_ds.Dataset(data).error if not isinstance(data, _ds.Dataset) else data.error for data in value]).transpose()
            self._data = value
        else:
            self._data = None
            _warnings.warn("aa")

    
    @values.setter
    def values(self, vals = None):
        if vals is not None:
            if isinstance(vals, list):
                self._values = vals
            else:
                self._values = [vals]
        else:
            self._values = vals
            _warnings.warn(" Values are nontype setting to default")


    def run(self):

        self._eps = _np.sqrt(_np.finfo(float).eps) #setting machine eps
        self._prop = _np.sqrt(_np.sum(_np.dot(self._values[0]**2, self._values[1]**2), axis = 1))
        return self._prop

        


