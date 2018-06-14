import warnings as _warnings
import typing as _typing
import inspect as _inspect


import numpy as _np
import matplotlib.pyplot as _plt
import pandas as _pd

from . import global_funcs as _gf
from . import global_enums as _ge
from . import dataset as _ds



class errorProp(_ds.Dataset):


    def __init__(self, data, fn, values):
        
        self._data = data
        self._fn = fn 
        self._values = values
    
    # Avoiding inconsistency from inherited object, we want to inherit all Dataset properties, but
    # we'll mask the attributes that refer to errors

    #-------------------------------------------------------------------------------------------------------

    @property
    def error(self):
        raise AttributeError
    
    @property
    def errorFn(self):
        raise AttributeError
    
    #--------------------------------------------------------------------------------------------------------

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
            if len(_inspect.getfullargspec(func).args) == len(self.data):  # checking that the function has the same numbers of arguments as datasets
                self._fn = func
            else:
                self._fn = None
                _warnings.warn(" #args of function does not match numbers of datasets, settings to None")
        else:
            self._fn = None
            _warnings.warn(" function is not callable")


    @data.setter
    def data(self, value = None):
        if isinstance(value, _ds.Dataset):
            self._data = value
        elif isinstance(value, list):
            for data in value:
                value = [_ds.Dataset(data) if not isinstance(data, _ds.Dataset) else data]
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

        val_max = max(self._values) # setting bounds for computing gradient
        val_min = min(self._values)    
        test_data = _np.arange(val_max, val_min)
        
        
        