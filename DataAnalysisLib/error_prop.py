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


    def __init__(self, data, fn):
        
        self._data = data
        self._fn = fn 
    
    @property
    def data(self):
        return self._data

    @property 
    def fn(self):
        return self._fn

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
            values = []
            values.append(self._data.v)
            values.append(self._data.error)
        elif isinstance(value, list):
            values = []
            for data in value:   # making big array with all dataset as columns, this form is convenient to evaluate error propagation
                values.append(_np.array([data.v for data in value if isinstance(data, _ds.Dataset)]).transpose())
                values.append(_np.array([data.error for data in value if isinstance(data, _ds.Dataset)]).transpose())
            self._data = values
        else:
            self._data = None
            _warnings.warn("aa")

    
    def run(self, params):

        self._eps = _np.sqrt(_np.finfo(float).eps) #setting machine eps
        self._grad = _np.array([_opt.approx_fprime(self._data[0][i,:], self._fn, self._eps, *params) for i in range(self._data[0].size)]).transpose()
        self._prop = _np.sqrt(_np.sum(self._grad**2 * self._data[1]**2))
        return self._prop

        

data = _ds.Dataset([1,2,3], [0.1, 0.2, 0.3])
print(data.v)
error = errorProp(_ds.Dataset([1,2,3], [0.1, 0.2, 0.3]), lambda B,x: B[0]*x[0] + B[1]*x[1])
error.run([0,1])