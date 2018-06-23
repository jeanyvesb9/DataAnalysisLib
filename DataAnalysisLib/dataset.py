import warnings as _warnings
import typing as _typing
from scipy import optimize as _opt
import inspect as _inspect

import numpy as _np
import matplotlib.pyplot as _plt
import pandas as _pd

import global_funcs as _gf
import global_enums as _ge

DEFAULT_DATASET_NAME = 'v'

class Dataset(object):
    def __init__(self, v: _typing.Any, error: _typing.Any = None, errorFn: _typing.Callable[[float], float] = None, name: str = None, units: str = None):
        self.v = _np.array(v)
        
        if self.v.ndim != 1:
            _warnings.warn('Incorrect dimension of v.')

        if error is not None:
            if isinstance(error, _np.ndarray) or isinstance(error, list):
                if errorFn is None:
                    if len(error) != len(self.v):
                        self.error = None
                        _warnings.warn('len(error) != len(v): Default error (None) selected.')
                    else:
                        self.error = error
                else:
                    self.error = None
                    _warnings.warn('error overdefined: explicit and functional definition of error given. \
                                    Default error (None) selected.')
            else:
                self.error = _np.ones(len(self.v)) * error
        else:
            if errorFn is not None:
                self.error = errorFn(self.v)
            else:
                self.error = None
        
        self.name = name #None type checking in setter
        self.units = units #empty and None type checking in setter

    #Idiot proofing the library:

    @property
    def name(self) -> str:
        return self._name
    @name.setter
    def name(self, value: str):
        self._name = value if value is not None else DEFAULT_DATASET_NAME

    @property
    def units(self) -> str:
        return self._units
    @units.setter
    def units(self, value: str):
        self._units = value if value is not None and value != '' else None
    
    #End of idiot proofing.

    def prettyName(self) -> str:
        return self.name if self.units is None else self.name + ' (' + self.units + ')'

    def cut(self, initialIndex: int = None, finalIndex: int = None):
        if (initialIndex is not None or finalIndex is not None) and isinstance(initialIndex, (int, None)) and isinstance(finalIndex, (int, None)):
            if initialIndex is not None:
                if initialIndex in range(0, len(self.v)):
                    self.v = self.v[initialIndex:]
                    self.error = self.error[initialIndex:]
                else:
                    _warnings.warn("initialIndex is out of range, setting default values")

            if finalIndex is not None:
                if finalIndex in range(0, len(self.v)):
                    self.v = self.v[:finalIndex - initialIndex + 1]
                    self.error = self.error[:finalIndex - initialIndex + 1]
                else:
                    _warnings.warn("finalIndex is out of range, setting default values")
        else:
            _warnings.warn("initialIndex/finalIndex type is not int, not executing")
    
    def purge(self, step: int): #step >= 1
        if isinstance(step, int):
            if step in range(1, len(self.v)):
                self.v = self.v[::step]
                self.error = self.error[::step] if self.error is not None else self.error
            else:
                _warnings.warn("step is out of range (1 <= step <= len(v)), not executing")
        else:
            _warnings.warn("step type is not int, not executing")
    
    def remove(self, index: int):
        self.v = _np.delete(self.v, index)
        self.error = _np.delete(self.error, index)
    
    def indexAtValue(self, value: float, exact: bool = True) -> int:
        return _np.where(self.v == value) if exact else _gf.findNearestValueIndex(self.v, value)

    def getMean(self) -> float:
        return _np.mean(self.v)
    
    def getStdDev(self) -> float:
        return _np.std(self.v, ddof = 1)
    
    def getStdDevOfMean(self) -> float:
        return self.getStdDev()/_np.sqrt(len(self.v))
    
    def getWeightedMean(self) -> float:
        if _np.count_nonzero(self.error) != len(self.error):
            _warnings.warn('Some values of self.error are 0. Returning unweighted mean.')
            return self.getMean()
        weights = 1/self.error**2
        return _np.sum(self.v * weights)/_np.sum(weights)
    
    def getWeightedMeanError(self) -> float:
        if _np.count_nonzero(self.error) != len(self.error):
            _warnings.warn('Some values of self.error are 0. Returning 0.')
            return 0
        weights = 1/self.error**2
        return 1/_np.sqrt(_np.sum(weights**2))
    
    def quickHistogram(self, bins: int = 'auto', range: _typing.Tuple[float, float] = None, normalized: bool = False):
        _plt.hist(self.v, bins, range = range, density = normalized)
        _plt.xlabel(self.prettyName())
        _plt.ylabel('Probability' if normalized else 'Counts')
        _plt.grid(True)
        _plt.show()
    
    def dataFrame(self, rounded: bool = True, separatedError: bool = False, relativeError: bool = False, saveCSVFile: str = None, \
                    CSVSep: str = ',', CSVDecimal: str = '.'):
        table = _gf.createSeriesPanda(self.v, error = self.error, label = self.name, units = self.units, relativeError = relativeError, \
                                    separated = separatedError, rounded = rounded)
        
        if saveCSVFile is not None:
            table.to_csv(saveCSVFile, sep = CSVSep, decimal = CSVDecimal)
        
        return table


    def errorProp(self, fn = None, fnPrime = None) -> _np.ndarray:

        if fnPrime is not None and hasattr(fnPrime, '__call__'):
            prop = fnPrime(self.v)*self.error
        elif fnPrime is None:
            
            # numerical derivation

            if fn is not None and hasattr(fn, '__call__'): #testing if it a function

                eps = _np.sqrt(_np.finfo(float).eps) # machine epsilon

                args = _inspect.getfullargspec(fn).args  
                coef_args, vars_args = args[0], args[1] if len(args) == 2 else None and _warnings.warn(' incorrect format on fn')
                prime = _opt.approx_fprime(self.v, fn, [eps, _np.sqrt(200) * eps], *coef_args)
                prop = _np.sqrt(_np.sum((prime**2)*(self.error)**2))

        return prop