import warnings as _warnings
import typing as _typing

import numpy as _np
import matplotlib.pyplot as _plt
import pandas as _pd
from . import global_funcs as _gf
from . import global_enums as _ge
from . import dataset as _ds


class MultiDataset(object):
    def __init__(self, datasets: [_ds.Dataset], name: str = None, covMatrix: _typing.Any = None, autoGenCov: bool = False):
        
        self.datasets = datasets

        for dataset in self.datasets:
            if len(dataset) != len(self):
                raise ValueError('Datasets provided have different lengths.')

        if covMatrix is not None and autoGenCov:
            raise Exception('Both covMatrix and autoGenCov were provided. Use Only one.')
        elif autoGenCov:
            self.generateCovMatricesFromErrors()
        else:
            self.covMatrices = covMatrix

        self._name = name

    @property
    def covMatrices(self) -> _typing.Any:
        return self._covMatrices
    @covMatrices.setter
    def covMatrices(self, covMatrix: _typing.Any):
        if covMatrix is None:
            self._covMatrices = None
            return

        covMatricesListShape = (self.shape[0], self.shape[1], self.shape[1]) #(n, p, p)
        self._covMatrices = _gf._conv(covMatrix)
        
        if self._covMatrices.ndim == 2:
            #User has provided a single cov matrix for all data points in MultiDataset
            if self._covMatrices.shape[0] != covMatricesListShape[1] \
                or self._covMatrices.shape[1] != covMatricesListShape[2]:
                raise ValueError('covMatrix provided has the wrong shape: (len(datasets), len(datasets)) expected.')
            cm = self._covMatrices
            self._covMatrices = _np.repeat(cm[_np.newaxis, :, :], covMatricesListShape[0], axis=0)
        elif self._covMatrices.ndim == 3:
            if self._covMatrices.shape != covMatricesListShape:
                raise ValueError("covMatrix provided has the wrong shape: (n, len(datasets), len(datasets)) expected, where 'n' is the number of points.")
            pass #all done here
        else:
            raise ValueError('covMatrix provided has the wrong shape.')

    def generateCovMatricesFromErrors(self):
        for dataset in self.datasets:
            if dataset.error is None:
                raise Exception('One or more datasets have no errors specified.')
        
        covMatricesListShape = (self.shape[0], self.shape[1], self.shape[1]) #(n, p, p)
        self.covMatrices = _np.zeros(shape = covMatricesListShape)
        for i, m in enumerate(self.covMatrices):
            for j in range(covMatricesListShape[1]):
                m[j][j] = self.datasets[j].error[i]

    def updateErrorsFromCovMatrices(self):
        if self.covMatrices is None:
            raise Exception('No available covMatrices.')
        
        for i, ds in enumerate(self.datasets):
            ds.error = _np.zeros(len(self))
            for j, m in enumerate(self.covMatrices):
                ds.error[j] = m[i][i]



    @property
    def name(self) -> str:
        return self._name
    @name.setter
    def name(self, value: str):
        self._name = value if value is not None else ''



    def cut(self, initialIndex: int = None, finalIndex: int = None):
        for ds in self.datasets:
            ds.cut(initialIndex, finalIndex)
        if initialIndex is not None:
            self.covMatrices = self.covMatrices[initialIndex:]
        if finalIndex is not None:
            self.covMatrices = self.covMatrices[:finalIndex - initialIndex + 1]

    def purge(self, step: int): #step >= 1
        for ds in self.datasets:
            ds.purge(step)
        self.covMatrices = self.covMatrices[::step] if self.covMatrices is not None else self.covMatrices

    def remove(self, index: int):
        for ds in self.datasets:
            ds.remove(index)
        self.covMatrices = _np.delete(self.covMatrices, index, axis = 0)
        
    def insert(self, index: int, data: _typing.Any, error: _typing.Any = None, covMatrix: _typing.Any = None):
        if not isinstance(index, int):
            raise TypeError("'index' type is not int.")
        if index not in range(len(self)):
            raise IndexError("'index' is out of range.")
        
        if len(data) != self.shape[1]:
            raise ValueError("len(data) != no. of datasets.")
        
        dsNoErr = False
        for dataset in self.datasets:
            if dataset.error is None:
                dsNoErr = True
        
        if covMatrix is not None:
            covMatrix = _gf._conv(covMatrix)
            if covMatrix.shape != (self.shape[1], self.shape[1]):
                raise ValueError('covMatrix provided has the wrong shape: (len(datasets), len(datasets)) expected.')
            if error is not None:
                raise Exception('Both covMatrix and error were provided. Use Only one.')

            if not dsNoErr:
                error = _np.diag(covMatrix)

        elif error is not None:
            if len(error) != self.shape[1]:
                raise ValueError("len(error) != num. of datasets.")
            if self.covMatrices is not None:
                covMatrix = _np.diag(error)

        for i, ds in enumerate(self.datasets):
            ds.insert(index, data[i], error[i] if error is not None else None)

        if covMatrix is not None:
            if self.covMatrices is None:
                _warnings.warn('covMatrices array has been initialized with null matrices.')
                nullMatrix = _np.zeros(shape = (self.shape[1], self.shape[1]))
                self.covMatrices = _np.repeat(nullMatrix[_np.newaxis, :, :], len(self), axis = 0)
            else:
                self.covMatrices = _np.insert(self.covMatrices, index, covMatrix, axis = 0)
        elif self.covMatrices is not None:
            nullMatrix = _np.zeros(shape = (self.shape[1], self.shape[1]))
            self.covMatrices = _np.insert(self.covMatrices, index, nullMatrix, axis = 0)
            _warnings.warn("Covariance matrix has been set to null matrix on insertion.")


    def sortByDataset(self, datasetIndex: int, reversed: bool = False, indexList: _typing.Any = None):
        if indexList is None:
            indexList = _np.argsort(self.datasets[datasetIndex])
        #Type and format checking done in Dataset.sort() next:

        for ds in self.datasets:
            ds.sort(reversed = reversed, indexList = indexList)

        if reversed:
            indexList = _np.flip(indexList, 0)
        
        if self.covMatrices is not None:
            self.covMatrices = self.covMatrices[indexList]

    def indexAtDataset(self, datasetIndex, value: float, exact: bool = True) -> int:
        return self.datasets[datasetIndex].indexAtValue(value, exact)

    def dataFrame(self, rounded: bool=True, signifficantDigits=1, separatedErrors: _typing.Any=False, relativeErrors: _typing.Any=False, \
                saveCSVFile: str=None, CSVSep: str=',', CSVDecimal: str='.'):
        cols = []
        for i, ds in enumerate(self.datasets):
            cols += [ds.dataFrame(rounded = rounded, signifficantDigits=signifficantDigits, \
                                    separatedError = separatedErrors if isinstance(separatedErrors, bool) else separatedErrors[i], \
                                    relativeError = relativeErrors if isinstance(relativeErrors, bool) else relativeErrors[i])]
        
        table = _pd.concat(cols, axis = 1, join = 'inner')
        
        if saveCSVFile is not None:
            table.to_csv(saveCSVFile, sep = CSVSep, decimal = CSVDecimal)
        
        return table

    @property
    def shape(self):
        #return (n, p), where n is the number of data points and p is the number of datasets
        return (len(self.datasets[0]), len(self.datasets))

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        if isinstance(key, str):
            foundDs = None
            for ds in self.datasets:
                if ds.name == key:
                    if foundDs is not None:
                        raise KeyError('MultiDataset contains more than one Dataset with the same name.')
                    foundDs = ds
            if foundDs is None:
                raise KeyError("Dataset with key '" + key + "'not found.")
            return ds
        elif isinstance(key, int):
            if key not in range(len(self)):
                raise IndexError()
            return tuple([ds[key] for ds in self.datasets])
        else:
            raise TypeError()
