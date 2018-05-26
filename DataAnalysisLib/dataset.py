import warnings

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from . import global_funcs
from . import global_enums

class DataSet(object):
    def __init__(self, x, y, xError = None, xErrorFn = None, yError = None, yErrorFn = None, xLabel = 'x', yLabel = 'y', \
                    xUnits = None, yUnits = None, name = ''):
        self.x = np.array(x)
        self.y = np.array(y)

        if self.x.ndim != 1:
            warnings.warn("Incorrect dimension of x.")
        if self.y.ndim != 1:
            warnings.warn("Incorrect dimension of y.")
        if self.x.size != self.y.size:
            sx = self.x.size
            sy = self.y.size
            d = np.abs(sx - sy)
            diff = np.zeros(d)
            if sx > sy:
                self.y = np.concatenate(self.y, diff)
                warnings.warn('len(x) > len(y): y has been filled with zeros to match sizes.')
            else:
                self.x = np.concatenate(self.x, diff)
                warnings.warn('len(y) > len(x): x has been filled with zeros to match sizes.')

        if xError is not None:
            if isinstance(xError, np.ndarray) or isinstance(xError, list):
                if xErrorFn is None:
                    if xError.size != self.x.size:
                        self.xError = np.zeros(x.size)
                        warnings.warn('len(xError) != len(x): Default error (zeros) selected.')
                    else:
                        self.xError = xError
                else:
                    self.xError = np.zeros(x.size)
                    warnings.warn('xError overdefined: explicit and functional definition of xError given. Default error (zeros) selected.')
            else:
                self.xError = np.ones(len(self.x)) * xError
        else:
            if xErrorFn is not None:
                self.xError = xErrorFn(self.x, self.y)
            else:
                self.xError = np.zeros(self.x.size)
        
        if yError is not None:
            if isinstance(yError, np.ndarray) or isinstance(yError, list):
                if yErrorFn is None:
                    if yError.size != self.y.size:
                        self.yError = np.zeros(y.size)
                        warnings.warn('len(yError) != len(y): Default error (zeros) selected.')
                    else:
                        self.yError = yError
                else:
                    self.yError = np.zeros(y.size)
                    warnings.warn('yError overdefined: explicit and functional definition of yError given. Default error (zeros) selected.')
            else:
                self.yError = np.ones(len(self.y)) * yError
        else:
            if yErrorFn is not None:
                self.yError = yErrorFn(self.x, self.y)
            else:
                self.yError = np.zeros(self.y.size)

        self.xLabel = xLabel
        self.yLabel = yLabel
        self.xUnits = xUnits if xUnits is not None and xUnits != '' else None
        self.yUnits = yUnits if yUnits is not None and yUnits != '' else None
        self.name = name

    def cut(self, initialIndex = None, finalIndex = None):
        if initialIndex is not None:
            self.x = self.x[initialIndex:]
            self.y = self.y[initialIndex:]
            self.xError = self.xError[initialIndex:]
            self.yError = self.yError[initialIndex:]
        if finalIndex is not None:
            self.x = self.x[:finalIndex - initialIndex]
            self.y = self.y[:finalIndex - initialIndex]
            self.xError = self.xError[:finalIndex - initialIndex]
            self.yError = self.yError[:finalIndex - initialIndex]
    
    def purge(self, step): #step >= 1
        if step <= 0:
            warnings.warn('step has to be at least 1. Quiting function.')
            return
        self.x = self.x[::step]
        self.y = self.y[::step]
        self.xError = self.xError[::step]
        self.yError = self.yError[::step]

    def remove(self, index):
        self.x = np.delete(self.x, index)
        self.y = np.delete(self.y, index)
        self.xError = np.delete(self.xError, index)
        self.yError = np.delete(self.yError, index)
    
    def indexAtX(self, value, exact = True):
        if exact:
            return np.where(self.x == value)[0]
        else:
            return global_funcs.findNearestValueIndex(self.x, value)
    
    def indexAtY(self, value, exact = True):
        if exact:
            return np.where(self.y == value)[0]
        else:
            return global_funcs.findNearestValueIndex(self.y, value)

    def getMean(self):
        return np.mean(self.y)
    
    def getStdDev(self):
        return np.std(self.y, ddof = 1)
    
    def getStdDevOfMean(self):
        return self.getStdDev()/np.sqrt(len(self.y))
    
    def getWeightedMean(self):
        if np.count_nonzero(self.yError) != len(self.yError):
            warnings.warn('Some values of self.yError are 0. Returning unweighted mean.')
            return self.getMean
        weights = 1/self.yError**2
        return np.sum(self.y * weights)/np.sum(weights)
    
    def getWeightedMeanError(self):
        if np.count_nonzero(self.yError) != len(self.yError):
            warnings.warn('Some values of self.yError are 0. Returning 0.')
            return 0
        weights = 1/self.yError**2
        return 1/np.sqrt(np.sum(weights**2))

    def quickPlot(self, plotType = global_enums.PlotType.ErrorBar, purgeStep = 1):
        if purgeStep <= 0:
            warnings.warn('purgeStep has to be at least 1. Setting purgeStep = 1.')
            purgeStep = 1
        fig , ax = plt.subplots(1,1)
        if plotType == global_enums.PlotType.ErrorBar:
            ax.errorbar(self.x[::purgeStep], self.y[::purgeStep], xerr = self.xError[::purgeStep], \
                        yerr = self.yError[::purgeStep], fmt = 's')
        elif plotType == global_enums.PlotType.Line:
            ax.plot(self.x[::purgeStep], self.y[::purgeStep], '-')
        elif plotType == global_enums.PlotType.Point:
            ax.plot(self.x[::purgeStep], self.y[::purgeStep], 's')
        ax.set_xlabel(self.xLabel if self.xUnits is None else self.xLabel + ' (' + self.xUnits + ')')
        ax.set_ylabel(self.yLabel if self.yUnits is None else self.yLabel + ' (' + self.yUnits + ')')
        ax.set_title(self.name)
        return fig, ax
    
    def dataFrame(self, rounded = True, xSeparatedError = False, xRelativeError = False, ySeparatedError = False, \
                yRelativeError = False, saveCSVFile = None, CSVSep = ',', CSVDecimal = '.'):
        xCol = global_funcs.createSeriesPanda(self.x, error = self.xError, label = self.xLabel, unit = self.xUnits, relativeError = xRelativeError, \
                                    separated = xSeparatedError, rounded = rounded)
        yCol = global_funcs.createSeriesPanda(self.y, error = self.yError, label = self.yLabel, unit = self.yUnits, relativeError = yRelativeError, \
                                    separated = ySeparatedError, rounded = rounded)
        
        table = pd.concat([xCol, yCol], axis = 1, join = 'inner')
        
        if saveCSVFile is not None:
            table.to_csv(saveCSVFile, sep = CSVSep, decimal = CSVDecimal)
        
        return table
