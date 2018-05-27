import warnings as _warnings

import numpy as _np
import matplotlib.pyplot as _plt
import pandas as _pd

from . import global_funcs as _gf
from . import global_enums as _ge

DEFAULT_DATASET_NAME = 'v'

class Dataset(object):
    def __init__(self, v, error = None, errorFn = None, name = None, units = None, suppressWarnings = False):
        self.v = _np.array(v)
        
        if self.v.ndim != 1:
            _warnings.warn('Incorrect dimension of v.')

        if error is not None:
            if isinstance(error, _np.ndarray) or isinstance(error, list):
                if errorFn is None:
                    if len(error) != len(self.v):
                        self.error = None
                        _warnings.warn('len(error) != len(v): Default error (None) selected.') if not suppressWarnings else None
                    else:
                        self.error = error
                else:
                    self.error = None
                    _warnings.warn('error overdefined: explicit and functional definition of error given. \
                                    Default error (None) selected.') if not suppressWarnings else None
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
    def name(self):
        return self._name
    @name.setter
    def name(self, value):
        self._name = value if value is not None else DEFAULT_DATASET_NAME

    @property
    def units(self):
        return self._units
    @units.setter
    def units(self, value):
        self._units = value if value is not None and value != '' else None
    
    #End of idiot proofing.

    def prettyName(self):
        return self.name if self.units is None else self.name + ' (' + self.units + ')'

    def cut(self, initialIndex = None, finalIndex = None):
        if initialIndex is not None:
            self.v = self.v[initialIndex:]
            self.error = self.error[initialIndex:]
        if finalIndex is not None:
            self.v = self.v[:finalIndex - initialIndex + 1]
            self.error = self.error[:finalIndex - initialIndex + 1]
    
    def purge(self, step): #step >= 1
        if step <= 0:
            _warnings.warn('step has to be at least 1. Quiting function.')
            return
        self.v = self.v[::step]
        self.error = self.error[::step]
    
    def remove(self, index):
        self.v = _np.delete(self.v, index)
        self.error = _np.delete(self.error, index)
    
    def indexAtValue(self, value, exact = True):
        return _np.where(self.v == value) if exact else _gf.findNearestValueIndex(self.v, value)

    def getMean(self):
        return _np.mean(self.v)
    
    def getStdDev(self):
        return _np.std(self.v, ddof = 1)
    
    def getStdDevOfMean(self):
        return self.getStdDev()/_np.sqrt(len(self.v))
    
    def getWeightedMean(self):
        if _np.count_nonzero(self.error) != len(self.error):
            _warnings.warn('Some values of self.error are 0. Returning unweighted mean.')
            return self.getMean()
        weights = 1/self.error**2
        return _np.sum(self.v * weights)/_np.sum(weights)
    
    def getWeightedMeanError(self):
        if _np.count_nonzero(self.error) != len(self.error):
            _warnings.warn('Some values of self.error are 0. Returning 0.')
            return 0
        weights = 1/self.error**2
        return 1/_np.sqrt(_np.sum(weights**2))
    
    def quickHistogram(self, bins = 'auto', range = None, normalized = False):

        _plt.hist(self.v, bins, range = range, density = normalized)
        _plt.xlabel(self.prettyName())
        _plt.ylabel('Probability' if normalized else 'Counts')
        _plt.grid(True)
        _plt.show()
    
    def dataFrame(self, rounded = True, separatedError = False, relativeError = False, saveCSVFile = None, \
                    CSVSep = ',', CSVDecimal = '.'):
        table = _gf.createSeriesPanda(self.v, error = self.error, label = self.name, units = self.units, relativeError = relativeError, \
                                    separated = separatedError, rounded = rounded)
        
        if saveCSVFile is not None:
            table.to_csv(saveCSVFile, sep = CSVSep, decimal = CSVDecimal)
        
        return table