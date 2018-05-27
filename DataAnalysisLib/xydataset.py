import warnings as _warnings

import numpy as _np
import matplotlib.pyplot as _plt
import pandas as _pd

from . import global_funcs as _gf
from . import global_enums as _ge
from . import dataset as _ds


class XYDataset(object):
    def __init__(self, x, y, xError = None, xErrorFn = None, yError = None, yErrorFn = None, xLabel = None, yLabel = None, \
                    xUnits = None, yUnits = None, name = None):

        if isinstance(x, _ds.Dataset):
            self.xDataset = x
        else:
            x = _np.array(x)
            if x.ndim != 1:
                _warnings.warn("Incorrect dimension of x.")
            self.xDataset = _ds.Dataset(x, suppressWarnings = True)

        if isinstance(y, _ds.Dataset):
            self.yDataset = y
        else:
            y = _np.array(y)
            if y.ndim != 1:
                _warnings.warn("Incorrect dimension of y.")
            self.yDataset = _ds.Dataset(y, suppressWarnings = True)

        if self.x.size != self.y.size:
            sx = self.x.size
            sy = self.y.size
            d = _np.abs(sx - sy)
            diff = _np.zeros(d)
            if sx > sy:
                self.y = _np.concatenate(self.y, diff)
                self.yError = None
                _warnings.warn('len(x) > len(y): y has been filled with zeros to match sizes. yError has been set to None.')
            else:
                self.x = _np.concatenate(self.x, diff)
                self.xError = None
                _warnings.warn('len(y) > len(x): x has been filled with zeros to match sizes. xError has been set to None.')

        if xError is not None:
            if isinstance(xError, _np.ndarray) or isinstance(xError, list):
                if xErrorFn is None:
                    if xError.size != self.x.size:
                        self.xError = None
                        _warnings.warn('len(xError) != len(x): Default error (None) selected.')
                    else:
                        self.xError = xError
                else:
                    self.xError = None
                    _warnings.warn('xError overdefined: explicit and functional definition of xError given. Default error (zeros) selected.')
            else:
                self.xError = _np.ones(len(self.x)) * xError
        elif xErrorFn is not None:
            self.xError = xErrorFn(self.x, self.y)
        #Else, if x was initially a Dataset, use the xErrors already in x (if not removed due to len(x) != len(y).
        #Otherwise, use None (default value selected when self.xDataset was created)
            
        if yError is not None:
            if isinstance(yError, _np.ndarray) or isinstance(yError, list):
                if yErrorFn is None:
                    if yError.size != self.y.size:
                        self.yError = None
                        _warnings.warn('len(yError) != len(y): Default error (None) selected.')
                    else:
                        self.yError = yError
                else:
                    self.yError = None
                    _warnings.warn('yError overdefined: explicit and functional definition of yError given. Default error (None) selected.')
            else:
                self.yError = _np.ones(len(self.y)) * yError
        elif yErrorFn is not None:
            self.yError = yErrorFn(self.x, self.y)
        #Else, if y was initially a Dataset, use the yErrors already in y (if not removed due to len(x) != len(y).
        #Otherwise, use None (default value selected when self.yDataset was created)
        
        #Check for already set labels and units in original x, y Dataset (if provided).
        if self.xLabel == _ds.DEFAULT_DATASET_NAME:
            self.xLabel = xLabel #empty and None type checking in setter
        if self.yLabel == _ds.DEFAULT_DATASET_NAME:
            self.yLabel = yLabel #empty and None type checking in setter
        if self.xUnits is None:
            self.xUnits = xUnits #empty str checking done in setter
        if self.yUnits is None:
            self.yUnits = yUnits #empty str checking done in setter
        self.name = name #None type checking done in setter


    @property
    def x(self):
        return self.xDataset.v
    @x.setter
    def x(self, value):
        self.xDataset.v = value

    @property
    def y(self):
        return self.yDataset.v
    @y.setter
    def y(self, value):
        self.yDataset.v = value
    
    @property
    def xError(self):
        return self.xDataset.error
    @xError.setter
    def xError(self, value):
        self.xDataset.error = value

    @property
    def yError(self):
        return self.yDataset.error
    @yError.setter
    def yError(self, value):
        self.yDataset.error = value

    #Idiot proofing the library:

    @property
    def xLabel(self):
        return self.xDataset.name
    @xLabel.setter
    def xLabel(self, value):
        self.xDataset.name = value if value is not None and value != '' else 'x'
    
    @property
    def yLabel(self):
        return self.yDataset.name
    @yLabel.setter
    def yLabel(self, value):
        self.yDataset.name = value if value is not None and value != '' else 'y'

    @property
    def xUnits(self):
        return self.xDataset.units
    @xUnits.setter
    def xUnits(self, value):
        self.xDataset.units = value

    @property
    def yUnits(self):
        return self.yDataset.units
    @yUnits.setter
    def yUnits(self, value):
        self.yDataset.units = value

    @property
    def name(self):
        return self._name
    @name.setter
    def name(self, value):
        self._name = value if value is not None else ''

    #End of idiot proofing.

    def prettyXLabel(self):
        return self.xDataset.prettyName()

    def prettyYLabel(self):
        return self.yDataset.prettyName()
    
    
    def cut(self, initialIndex = None, finalIndex = None):
        self.xDataset.cut(initialIndex, finalIndex)
        self.yDataset.cut(initialIndex, finalIndex)

    def purge(self, step): #step >= 1
        self.xDataset.purge(step)
        self.yDataset.purge(step)
        
    def remove(self, index):
        self.xDataset.remove(index)
        self.yDataset.remove(index)
        
    def indexAtX(self, value, exact = True):
        return self.xDataset.indexAtValue(value, exact)
    
    def indexAtY(self, value, exact = True):
        return self.yDataset.indexAtValue(value, exact)
    
    def quickPlot(self, plotType = _ge.PlotType.ErrorBar, purgeStep = 1, initialXIndex = None, finalXIndex = None):
        if purgeStep <= 0:
            _warnings.warn('purgeStep has to be at least 1. Setting purgeStep = 1.')
            purgeStep = 1

        if initialXIndex is None:
            initialXIndex = 0
        elif initialXIndex < 0 or initialXIndex >= len(self.x):
            _warnings.warn('initialXIndex given is out of range. Default value selected.')
            initialXIndex = 0

        if finalXIndex is None:
            finalXIndex = len(self.x)
        elif finalXIndex < 0 or finalXIndex >= len(self.x):
            _warnings.warn('finalXIndex given is out of range. Default value selected.')
            finalXIndex = len(self.x)
        else:
            finalXIndex += 1 #correct for python final list index offset
        
        fig , ax = _plt.subplots(1,1)

        if plotType == _ge.PlotType.ErrorBar:
            ax.errorbar(self.x[initialXIndex:finalXIndex:purgeStep], self.y[initialXIndex:finalXIndex:purgeStep], \
                        xerr = self.xError[initialXIndex:finalXIndex:purgeStep] if self.xError is not None else None, \
                        yerr = self.yError[initialXIndex:finalXIndex:purgeStep] if self.yError is not None else None, fmt = 's')
        elif plotType == _ge.PlotType.Line:
            ax.plot(self.x[initialXIndex:finalXIndex:purgeStep], self.y[initialXIndex:finalXIndex:purgeStep], '-')
        elif plotType == _ge.PlotType.Point:
            ax.plot(self.x[initialXIndex:finalXIndex:purgeStep], self.y[initialXIndex:finalXIndex:purgeStep], 's')

        ax.set_xlabel(self.prettyXLabel())
        ax.set_ylabel(self.prettyYLabel())
        ax.set_title(self.name)

        return fig, ax
    
    def dataFrame(self, rounded = True, xSeparatedError = False, xRelativeError = False, ySeparatedError = False, \
                yRelativeError = False, saveCSVFile = None, CSVSep = ',', CSVDecimal = '.'):
        xCol = _gf.createSeriesPanda(self.x, error = self.xError, label = self.xLabel, units = self.xUnits, relativeError = xRelativeError, \
                                    separated = xSeparatedError, rounded = rounded)
        yCol = _gf.createSeriesPanda(self.y, error = self.yError, label = self.yLabel, units = self.yUnits, relativeError = yRelativeError, \
                                    separated = ySeparatedError, rounded = rounded)
        
        table = _pd.concat([xCol, yCol], axis = 1, join = 'inner')
        
        if saveCSVFile is not None:
            table.to_csv(saveCSVFile, sep = CSVSep, decimal = CSVDecimal)
        
        return table
