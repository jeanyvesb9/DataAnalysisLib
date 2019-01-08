import warnings as _warnings
import typing as _typing

import numpy as _np
import matplotlib.pyplot as _plt
import pandas as _pd

from . import global_funcs as _gf
from . import global_enums as _ge
from . import dataset as _ds
from . import multidataset as _mds
from . import xyfit as _xyfit

class XYDataset(_mds.MultiDataset):
    def __init__(self, x: _typing.Any, y: _typing.Any, xError: _typing.Any = None, xErrorFn: _typing.Callable[[float], float] = None, \
                    yError: _typing.Any = None, yErrorFn: _typing.Callable[[float], float] = None, covMatrix: _typing.Any = None, \
                    autoGenCov: bool = False, xLabel: str = None, yLabel: str = None, xUnits: str = None, yUnits: str = None, name: str = None):
        
        xDataset = self._formatInitialDatasets(x, name = 'x')
        yDataset = self._formatInitialDatasets(y, name = 'y')

        if len(xDataset) != len(yDataset):
            sx, sy = len(xDataset), len(yDataset)
            d = _np.abs(sx - sy)
            diff = _np.zeros(d)
            if sx > sy:
                yDataset.data = _np.concatenate(yDataset.data, diff)
                yDataset.error = None
                _warnings.warn('len(x) > len(y): y has been filled with zeros to match sizes. yError has been set to None.')
            else:
                xDataset.data = _np.concatenate(xDataset.data, diff)
                xDataset.error = None
                _warnings.warn('len(y) > len(x): x has been filled with zeros to match sizes. xError has been set to None.')

        xDataset = self._computeInitialError(xDataset, xError, xErrorFn, name = 'x')
        yDataset = self._computeInitialError(yDataset, yError, yErrorFn, name = 'y')

        if covMatrix is not None and (xError is not None or xErrorFn is not None or \
                                        yError is not None or yErrorFn is not None):
            raise Exception('Both covMatrix and an error scalar/list/function were provided. Use only one.')

        _mds.MultiDataset.__init__(self, [xDataset, yDataset], name = name, covMatrix = covMatrix, autoGenCov = autoGenCov)

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

    def _formatInitialDatasets(self, data: _typing.Any, name: str):
        if isinstance(data, _ds.Dataset):
            return data
        else:
            data = _gf._conv(data)
            if data.ndim != 1:
                raise Exception("Incorrect dimension of '" + name + "'.")
            else:
                return _ds.Dataset(data)

    def _computeInitialError(self, dataset: _ds.Dataset, error: _typing.Any, errorFn: _typing.Callable[[float], float], name: str) -> _ds.Dataset:
        if error is not None:
            if isinstance(error, _np.ndarray) or isinstance(error, list):
                if errorFn is None:
                    if len(error) != len(dataset):
                        dataset.error = None
                        _warnings.warn('len(' + name + 'Error) != len(' + name + '): Default error (None) selected.')
                    else:
                        dataset.error = error
                else:
                    raise Exception(name + 'Error overdefined: explicit and functional definition of ' + name + 'Error given. Use only one.')
            else:
                dataset.error = _np.ones(len(dataset)) * error
        elif errorFn is not None:
            dataset.error = errorFn(dataset.data)
        #Else, if 'dataset' was initially a Dataset, use the errors already in it.
        #Otherwise, use None (default value selected when dataset was created)
        return dataset


    @property
    def xDataset(self) -> _ds.Dataset:
        return self.datasets[0]
    @xDataset.setter
    def xDataset(self, ds: _ds.Dataset):
        self.datasets[0] = ds

    @property
    def yDataset(self) -> _ds.Dataset:
        return self.datasets[1]
    @yDataset.setter
    def yDataset(self, ds: _ds.Dataset):
        self.datasets[1] = ds

    @property
    def x(self):
        return self.xDataset.data
    @x.setter
    def x(self, value):
        self.xDataset.data = value

    @property
    def y(self):
        return self.yDataset.data
    @y.setter
    def y(self, value):
        self.yDataset.data = value
    
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

    

    @property
    def xLabel(self) -> str:
        return self.xDataset.name
    @xLabel.setter
    def xLabel(self, value: str):
        self.xDataset.name = value if value is not None and value != '' else 'x'
    
    @property
    def yLabel(self) -> str:
        return self.yDataset.name
    @yLabel.setter
    def yLabel(self, value: str):
        self.yDataset.name = value if value is not None and value != '' else 'y'

    @property
    def xUnits(self) -> str:
        return self.xDataset.units
    @xUnits.setter
    def xUnits(self, value: str):
        self.xDataset.units = value

    @property
    def yUnits(self) -> str:
        return self.yDataset.units
    @yUnits.setter
    def yUnits(self, value: str):
        self.yDataset.units = value

    @property
    def prettyXLabel(self) -> str:
        return self.xDataset.prettyName

    @property
    def prettyYLabel(self) -> str:
        return self.yDataset.prettyName

    def indexAtX(self, value: float, exact: bool = True) -> int:
        return self.xDataset.indexAtValue(value, exact)
    
    def indexAtY(self, value: float, exact: bool = True) -> int:
        return self.yDataset.indexAtValue(value, exact)
    
    def insert(self, index: int, x: _typing.Any = None, y: _typing.Any = None, \
                xError: _typing.Any = None, yError: _typing.Any = None, \
                data: _typing.Any = None, error: _typing.Any = None, covMatrix: _typing.Any = None):
                
                if ((x is not None or y is not None) and data is not None) or \
                    ((xError is not None or yError is not None) and error is not None):
                    raise ValueError('Multiple definitions of data and error were given. Use only one.')
                if data is None:
                    if x is None or y is None:
                        raise TypeError("Values for 'x' and 'y' are needed.") 
                    data = [x, y]
                if error is None:
                    if xError is not None or yError is not None:
                        error = [xError, yError]
                _mds.MultiDataset.insert(self, index, data, error, covMatrix)

    def sortByX(self, reversed: bool = False, indexList: _typing.Any = None):
        self.sortByDataset(0, reversed = reversed, indexList = indexList)
    
    def sortByY(self, reversed: bool = False, indexList: _typing.Any = None):
        self.sortByDataset(1, reversed = reversed, indexList = indexList)

    def quickFit(self, fn, independentVar: _typing.Any='', \
                 maxIterations: int=50, name=None, fitType=_ge.FitMethods.ODR, \
                 useIndVarErrors=True, useDepVarErrors=True, \
                 useCovMatrices=False, \
                 labels: _typing.Dict[str, str]={}, units: _typing.Dict[str, str]={}, \
                 fixed: _typing.Dict[str, bool]={}, **kwargs):
        
        if not useCovMatrices:
            if useIndVarErrors and self.xError is None:
                _warnings.warn("Independent variable (x) has no errors specified. Executing fit with 'useIndVarErrors=False", RuntimeWarning)
                useIndVarErrors = False
            if useDepVarErrors and self.yError is None:
                _warnings.warn("Dependent variable (y) has no errors specified. Executing fit with 'useDepVarErrors=False", RuntimeWarning)
                useDepVarErrors = False

        XYFitGen = _xyfit.XYFitGenerator(fn, independentVar=independentVar, \
                                         maxIterations=maxIterations, name=name, fitType=fitType, \
                                         useIndVarErrors=useIndVarErrors, useDepVarErrors=useDepVarErrors, \
                                         useCovMatrices=useCovMatrices, \
                                         labels=labels, units=units, \
                                         fixed=fixed)

        return XYFitGen.fit(self, **kwargs)

    def quickPlot(self, plotType = _ge.PlotType.ErrorBar, purgeStep: int = 1, initialXIndex: int = None, finalXIndex: int = None):
        
        if isinstance(purgeStep, int):
            if purgeStep not in range(1, len(self)):
                raise ValueError("purgeStep is out of range (1 <= purgeStep <= len(x)).")
        else:
            raise TypeError("purgeStep type is not int")
        
        if initialXIndex is None:
            initialXIndex = 0
        elif initialXIndex not in range(0, len(self)):
            _warnings.warn('initialXIndex given is out of range. Default value selected.')
            initialXIndex = 0

        if finalXIndex is None:
            finalXIndex = len(self.x)
        elif finalXIndex not in range(0, len(self)):
            _warnings.warn('finalXIndex given is out of range. Default value selected.')
            finalXIndex = len(self)
        else:
            finalXIndex += 1 #correct for python final list index offset
        
        fig, ax = _plt.subplots(1,1)

        if plotType == _ge.PlotType.ErrorBar:
            ax.errorbar(self.x[initialXIndex:finalXIndex:purgeStep], self.y[initialXIndex:finalXIndex:purgeStep], \
                        xerr = self.xError[initialXIndex:finalXIndex:purgeStep] if self.xError is not None else None, \
                        yerr = self.yError[initialXIndex:finalXIndex:purgeStep] if self.yError is not None else None, fmt = 's')
        elif plotType == _ge.PlotType.Line:
            ax.plot(self.x[initialXIndex:finalXIndex:purgeStep], self.y[initialXIndex:finalXIndex:purgeStep], '-')
        elif plotType == _ge.PlotType.Point:
            ax.plot(self.x[initialXIndex:finalXIndex:purgeStep], self.y[initialXIndex:finalXIndex:purgeStep], 's')
        else:
            raise ValueError("'plotType' chosen not available.")

        ax.set_xlabel(self.prettyXLabel)
        ax.set_ylabel(self.prettyYLabel)
        ax.set_title(self.name)

        return fig, ax
    
    def dataFrame(self, rounded: bool=True, signifficantDigits=1, \
                    xSeparatedError: bool=False, xRelativeError: bool = False, \
                    ySeparatedError: bool=False, yRelativeError: bool=False, \
                    saveCSVFile: str=None, CSVSep: str=',', CSVDecimal: str='.'):

        return _mds.MultiDataset.dataFrame(self, rounded=rounded, signifficantDigits=signifficantDigits,\
                                    separatedErrors=[xSeparatedError, ySeparatedError], \
                                    relativeErrors=[xRelativeError, yRelativeError], \
                                    saveCSVFile=saveCSVFile, CSVSep=CSVSep, CSVDecimal=CSVDecimal)
