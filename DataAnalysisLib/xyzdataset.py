import warnings as _warnings
import typing as _typing

import numpy as _np
import matplotlib.pyplot as _plt
import pandas as _pd
from . import global_funcs as _gf
from . import global_enums as _ge
from . import dataset as _ds
from . import multidataset as _mds
from . import xydataset as _xyds
from . import xyzfit as _xyzfit


class XYZDataset(_xyds.XYDataset):
    def __init__(self, x: _typing.Any, y: _typing.Any, z: _typing.Any, \
                    xError: _typing.Any = None, xErrorFn: _typing.Callable[[float], float] = None, \
                    yError: _typing.Any = None, yErrorFn: _typing.Callable[[float], float] = None, \
                    zError: _typing.Any = None, zErrorFn: _typing.Callable[[float], float] = None, \
                    covMatrix: _typing.Any = None, autoGenCov: bool = False, \
                    xLabel: str = None, yLabel: str = None, zLabel: str = None, \
                    xUnits: str = None, yUnits: str = None, zUnits: str = None, name: str = None):

        xDataset = self._formatInitialDatasets(x, name = 'x')
        yDataset = self._formatInitialDatasets(y, name = 'y')
        zDataset = self._formatInitialDatasets(z, name = 'z')

        if len(xDataset) != len(yDataset) or len(xDataset) != len(zDataset):
            raise Exception("len(x), len(y) and len(z) don't match. Datasets/lists needs to have the same length.")

        xDataset = self._computeInitialError(xDataset, xError, xErrorFn, name = 'x')
        yDataset = self._computeInitialError(yDataset, yError, yErrorFn, name = 'y')
        zDataset = self._computeInitialError(zDataset, zError, zErrorFn, name = 'z')
        
        if covMatrix is not None and (xError is not None or xErrorFn is not None or \
                                        yError is not None or yErrorFn is not None or \
                                        zError is not None or zErrorFn is not None):
            raise Exception('Both covMatrix and an error scalar/list/function were provided. Use only one.')

        _mds.MultiDataset.__init__(self, [xDataset, yDataset, zDataset], name = name, covMatrix = covMatrix, autoGenCov = autoGenCov)

        #Check for already set labels and units in original x, y Dataset (if provided).
        if self.xLabel == _ds.DEFAULT_DATASET_NAME:
            self.xLabel = xLabel #empty and None type checking in setter
        if self.yLabel == _ds.DEFAULT_DATASET_NAME:
            self.yLabel = yLabel #empty and None type checking in setter
        if self.zLabel == _ds.DEFAULT_DATASET_NAME:
            self.zLabel = zLabel #empty and None type checking in setter
        if self.xUnits is None:
            self.xUnits = xUnits #empty str checking done in setter
        if self.yUnits is None:
            self.yUnits = yUnits #empty str checking done in setter
        if self.zUnits is None:
            self.zUnits = zUnits #empty str checking done in setter
        self.name = name #None type checking done in setter

    @property
    def zDataset(self) -> _ds.Dataset:
        return self.datasets[2]
    @zDataset.setter
    def zDataset(self, ds: _ds.Dataset):
        self.datasets[2] = ds

    @property
    def z(self):
        return self.zDataset.data
    @z.setter
    def z(self, value):
        self.zDataset.data = value

    @property
    def zError(self):
        return self.zDataset.error
    @zError.setter
    def zError(self, value):
        self.zDataset.error = value

    @property
    def zLabel(self) -> str:
        return self.zDataset.name
    @zLabel.setter
    def zLabel(self, value: str):
        self.zDataset.name = value if value is not None and value != '' else 'z'

    @property
    def zUnits(self) -> str:
        return self.zDataset.units
    @zUnits.setter
    def zUnits(self, value: str):
        self.zDataset.units = value

    @property
    def prettyZLabel(self) -> str:
        return self.zDataset.prettyName

    def indexAtZ(self, value: float, exact: bool = True) -> int:
        return self.zDataset.indexAtValue(value, exact)

    def insert(self, index: int, x: _typing.Any = None, y: _typing.Any = None, z: _typing.Any = None, \
                xError: _typing.Any = None, yError: _typing.Any = None, zError: _typing.Any = None, \
                data: _typing.Any = None, error: _typing.Any = None, covMatrix: _typing.Any = None):

                if ((x is not None or y is not None or z is not None) and data is not None) or \
                    ((xError is not None or yError is not None or zError is not None) and error is not None):
                    raise ValueError('Multiple definitions of data and error were given. Use only one.')
                
                if data is None:
                    if x is None or y is None or z is None:
                        raise TypeError("Values for 'x', 'y' and 'z' are needed.")
                    data = [x, y, z]
                
                if error is None:
                    if xError is not None or yError is not None or zError is not None:
                        error = [xError, yError, zError]

                _mds.MultiDataset.insert(self, index, data, error, covMatrix)
    
    def sortByZ(self, reversed: bool = False, indexList: _typing.Any = None):
        self.sortByDataset(2, reversed = reversed, indexList = indexList)

    def quickFit(self, fn, independentVar: _typing.Any = '', \
                 maxIterations: int = 50, name = None, fitType = _ge.FitMethods.ODR, \
                 useIndVarErrors = True, useDepVarErrors = True, \
                 useCovMatrices=False, \
                 labels: _typing.Dict[str, str] = {}, units: _typing.Dict[str, str] = {}, \
                 fixed: _typing.Dict[str, bool] = {}, **kwargs):

        if not useCovMatrices:
            if useIndVarErrors and (self.xError is None or self.yError is None):
                _warnings.warn("One of the independent variables (x, y) have no errors specified. Executing fit with 'useIndVarErrors=False", RuntimeWarning)
                useIndVarErrors = False
            if useDepVarErrors and self.yError is None:
                _warnings.warn("Dependent variable (z) has no errors specified. Executing fit with 'useDepVarErrors=False", RuntimeWarning)
                useDepVarErrors = False

        XYZFitGen = _xyzfit.XYZFitGenerator(fn, independentVar=independentVar, \
                                         maxIterations=maxIterations, name=name, fitType=fitType, \
                                         useIndVarErrors=useIndVarErrors, useDepVarErrors=useDepVarErrors, \
                                         useCovMatrices=useCovMatrices, \
                                         labels=labels, units=units, \
                                         fixed=fixed)

        return XYZFitGen.fit(self, **kwargs)

    def quickPlot(self, plotType=_ge.PlotType.Point, zMin: float=None, zMax: float=None):
        fig = _plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        if plotType == _ge.PlotType.Point:
            ax.scatter(self.x, self.y, self.z)
        else:
            raise ValueError("'plotType' chosen not available.")

        ax.set_xlabel(self.prettyXLabel)
        ax.set_ylabel(self.prettyYLabel)
        ax.set_zlabel(self.prettyZLabel)
        ax.set_zlim(zMin, zMax)

        ax.set_title(self.name)

        return fig, ax
    
    def dataFrame(self, rounded: bool=True, signifficantDigits=1, \
                xSeparatedError: bool=False, xRelativeError: bool=False, \
                ySeparatedError: bool=False, yRelativeError: bool=False, \
                zSeparatedError: bool=False, zRelativeError: bool=False, \
                saveCSVFile: str=None, CSVSep: str=',', CSVDecimal: str='.'):

        return _mds.MultiDataset.dataFrame(self, rounded = rounded, signifficantDigits=signifficantDigits, \
                                    separatedErrors=[xSeparatedError, ySeparatedError, zSeparatedError], \
                                    relativeErrors=[xRelativeError, yRelativeError, zRelativeError], \
                                    saveCSVFile=saveCSVFile, CSVSep=CSVSep, CSVDecimal=CSVDecimal)

    