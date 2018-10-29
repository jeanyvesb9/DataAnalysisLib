import warnings as _warnings
import typing as _typing
from scipy import optimize as _opt
import inspect as _inspect
import collections as _collections
import numbers as _numbers

import numpy as _np
import matplotlib.pyplot as _plt
import pandas as _pd

from . import global_funcs as _gf
from . import global_enums as _ge

DEFAULT_DATASET_NAME = 'v'

class Dataset(object):
    def __init__(self, data: _typing.Any, error: _typing.Any = None, errorFn: _typing.Callable[[float], float] = None, name: str = None, units: str = None):
        self.data = _gf._conv(data)
        
        if self.data.ndim != 1:
            raise ValueError('Incorrect dimension of v.')

        if error is not None:
            if isinstance(error, _np.ndarray) or isinstance(error, list):
                if errorFn is None:
                    if len(error) != len(self):
                        self.error = None
                        _warnings.warn('len(error) != len(data): Default error (None) selected.')
                    else:
                        self.error = _gf._conv(error)
                else:
                    raise Exception('error overdefined: explicit and functional definition of error given. Use only one.')
            else:
                self.error = _np.ones(len(self)) * error
        else:
            if errorFn is not None:
                self.error = errorFn(self.data)
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
                if initialIndex in range(0, len(self)):
                    self.data = self.data[initialIndex:]
                    self.error = self.error[initialIndex:]
                else:
                    raise IndexError("initialIndex is out of range.")

            if finalIndex is not None:
                if finalIndex in range(0, len(self)):
                    self.data = self.data[:finalIndex - initialIndex + 1]
                    self.error = self.error[:finalIndex - initialIndex + 1]
                else:
                    raise IndexError("finalIndex is out of range.")
        else: #in case indexes are floats or both Nones
            raise TypeError("initialIndex/finalIndex type is not int.")
    
    def purge(self, step: int): #step >= 1
        if not isinstance(step, int):
            raise TypeError("'step' type is not int.")

        if step not in range(1, len(self)):
            raise IndexError("'step' is out of range (1 <= step <= len(v)).")

        self.data = self.data[::step]
        self.error = self.error[::step] if self.error is not None else self.error
    
    def remove(self, index: int):
        if isinstance(index, int):
            self.data = _np.delete(self.data, index)
            if self.error is not None:
                self.error = _np.delete(self.error, index)
        elif isinstance(index, _collections.Iterable):
            for i in reversed(sorted(list(set(index)))):
                self.remove(int(i))
        else:
            raise TypeError("'index' type is not int or list.")

    def insert(self, index: int, data: float, error: float = None):
        if not isinstance(index, int):
            raise TypeError("'index' type is not int.")
        
        if index not in range(len(self)):
            raise IndexError("'index' is out of range.")

        self.data = _np.insert(self.data, index, data)
        if error is not None:
            if self.error is not None:
                self.error = _np.insert(self.error, index, error)
            else:
                self.error = _np.zeros(len(self))
                _warnings.warn('Error array has been initialized with 0')
                self.error[index] = error
        elif self.error is not None:
            self.error = _np.insert(self.error, index, 0)
            _warnings.warn('Error has been set to 0 on insertion')
            
    def sort(self, reversed: bool = False, indexList: _typing.Any = None):
        if indexList is None:
            indexList = _np.argsort(self.data)
        else:
            indexList = _gf._conv(indexList)
            if not isinstance(indexList, _np.ndarray):
                raise TypeError('indexList is not a list.')
            elif indexList.ndim != 1 or len(indexList) != len(self):
                raise ValueError('indexList provided has the wrong shape: (len(data)) expected.')

        if reversed:
            indexList = _np.flip(indexList, 0)
        
        self.data = self.data[indexList]
        if self.error is not None:
            self.error = self.error[indexList]

    
    def indexAtValue(self, value: float, exact: bool = True) -> int:
        return _np.where(self.data == value) if exact else _gf.findNearestValueIndex(self.data, value)

    def getMean(self) -> float:
        return _np.mean(self.data)
    
    def getStdDev(self) -> float:
        return _np.std(self.data, ddof = 1)
    
    def getStdDevOfMean(self) -> float:
        return self.getStdDev()/_np.sqrt(len(self))
    
    def getWeightedMean(self) -> float:
        if 0 in self.error or None in self.error:
            raise Exception("Some values of self.error are 0 or None. Can't calculate weights.")
        weights = 1/self.error**2
        return _np.sum(self.data * weights)/_np.sum(weights)
    
    def getWeightedMeanError(self) -> float:
        if 0 in self.error or None in self.error:
            raise Exception("Some values of self.error are 0 or None. Can't calculate weights.")
        weights = 1/self.error**2
        return 1/_np.sqrt(_np.sum(weights**2))
    
    def quickHistogram(self, bins: int = 'auto', range: _typing.Tuple[float, float] = None, normalized: bool = False):
        _plt.hist(self.data, bins, range = range, density = normalized)
        _plt.xlabel(self.prettyName())
        _plt.ylabel('Probability' if normalized else 'Counts')
        _plt.grid(True)
        _plt.show()
    
    def dataFrame(self, rounded: bool=True, signifficantDigits=1, separatedError: bool=False, relativeError: bool=False, \
                    saveCSVFile: str=None, CSVSep: str=',', CSVDecimal: str='.'):
        table = _gf.createSeriesPanda(self.data, error = self.error, label = self.name, units = self.units, relativeError = relativeError, \
                                    separated = separatedError, rounded = rounded, signifficantDigits=signifficantDigits)
        
        if saveCSVFile is not None:
            table.to_csv(saveCSVFile, sep = CSVSep, decimal = CSVDecimal)
        
        return table

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if not isinstance(index, int):
            raise TypeError()
        
        if not index in range(len(self)):
            raise IndexError()
        
        return (self.data[index], self.error[index] if self.error is not None else None)
   