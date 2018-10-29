import warnings as _warnings
import functools as _functools
import typing as _typing
import inspect as _inspect
import copy as _copy
import collections as _collections

import numpy as _np
import scipy.odr as _odr
import matplotlib.pyplot as _plt
import pandas as _pd

from . import global_funcs as _gf
from . import global_enums as _ge
from . import dataset as _ds
from . import multidataset as _mds

class Parameter(object):
    class Empty:
        def __str__(self):
            return 'Empty Object'
    
    def __init__(self, name: str, label = None, units = None, initialValue = Empty(), fixed = None):
        self._name = name
        self._label = label
        self.units = units
        self._value = None
        self._error = None
        self.initialValue = initialValue
        if fixed is None:
            if type(self.initialValue) != int and type(self.initialValue) != float \
                and type(self.initialValue) != type(self.Empty()):
                self.fixed = True
            else:
                self.fixed = False
        else:
            self.fixed = fixed #checking done on property setter
        
    @property
    def name(self):
        return self._name
    
    @property
    def label(self):
        return self._label if self._label is not None else self.name
    @label.setter
    def label(self, value: str):
        if value is None:
            self._label = value
        elif isinstance(value, str):
            if value == '':
                raise ValueError('label cannot be an empty str. Use None instead.')
            self._label = value
        else:
            raise TypeError('label has to be of type str or None.')
            
    def prettyLabel(self):
        return self.label + (' (' + self.units + ')' if self.units is not None else '')
    
    @property
    def units(self):
        return self._units
    @units.setter
    def units(self, value: str):
        if value is None:
            self._units = value
        elif isinstance(value, str):
            if value == '':
                raise ValueError("'unit' cannot be an empty str. Use None instead.")
            self._units = value
        else:
            raise TypeError("'unit' has to be of type str or None.")
            
    @property
    def initialValue(self):
        return self._initialValue
    @initialValue.setter
    def initialValue(self, value: _typing.Any):
        if self._value is not None:
            raise Exception("Fit already initialized. Can't modify values.")
        if type(value) != int and type(value) != float and type(value) != type(self.Empty()):
            self._fixed = True
        self._initialValue = value
    
    @property
    def fixed(self):
        return self._fixed
    @fixed.setter
    def fixed(self, value):
        if self._value is not None:
            raise Exception("Fit already initialized. Can't modify values.")
        if type(self.initialValue) != int and type(self.initialValue) != float \
                and type(self.initialValue) != type(self.Empty()) and value is False:
            raise ValueError("Parameter 'initialValue' type is not int or float. Parameter has to be fixed.")
        self._fixed = value
        
    @property
    def value(self):
        if self._value is None:
            raise Exception('Value not set by fitting method.')
        return self._value
    
    @property
    def error(self):
        if self._value is None:
            raise Exception('Error not set by fitting method.')
        return self._error
    
    def __str__(self):
        ret = "Parameter name: " + self.name
        ret += "\n\tLabel: '" + self.label + "'"
        if self._label is None:
            ret += " (Default)"
        if self.units is not None:
            ret += "\n\tUnits: '" + self.units + "'"
        if self._value is not None and not self.fixed:
            ret += "\n\tValue: '" + str(self.value) + " +/- " + str(self.error) + "'"
        ret += "\n\tInitial Value: " + str(self.initialValue)
        ret += "\n\tFixed: " + str(self.fixed)
        return ret
    









class Fit(object):
    def __init__(self, fn, parameters, multiDataset, dependentDatasetIndex, independentVars, \
                    name, fitType, fitObj, fitOutput, initialIndex, finalIndex):
        self._fn = fn
        self._parameters = parameters
        self._multiDataset = multiDataset
        self._dependentDatasetIndex = dependentDatasetIndex
        self._independentVars = independentVars
        self._name = name
        self._fitType = fitType
        self._fitObj = fitObj
        self._fitOutput = fitOutput
        self._initialIndex = initialIndex
        self._finalIndex = finalIndex
        
        self._updateAfterFit()

    @property
    def fn(self):
        return self._fn
    
    @property
    def parameters(self):
        return self._parameters
    
    @property
    def multiDataset(self):
        return self._multiDataset

    @property
    def data(self): #alias for usefulness
        return self._multiDataset
    
    @property
    def dependentDatasetIndex(self):
        return self._dependentDatasetIndex
    
    @property
    def independentVars(self):
        return self._independentVars
    
    @property
    def name(self):
        return self._name
    
    @property
    def prettyName(self):
        return self.multiDataset.name + ( (' - ' + self.name) if self.name != '' else '')
    
    @property
    def fitType(self):
        return self._fitType
    
    @property
    def fitOutput(self):
        return self._fitOutput

    @property
    def R2(self):
        return self._R2
    
    @property
    def reducedChi2(self):
        return self.fitOutput.res_var
    
    @property
    def parametersCovMatrix(self):
        return self._parametersCovMatrix
    
    @property
    def fittedFn(self):
        args = {}
        for name, param in self.parameters.items():
            args[name] = param.value
                        
        return _functools.partial(self.fn, **args)

    @property
    def initialIndex(self):
        return self._initialIndex

    @property
    def finalIndex(self):
        return self._finalIndex
    
    def _updateAfterFit(self):
        #Update self.parameters
        i = 0
        for param in self._parameters.values():
            if not param.fixed:
                param._value = self.fitOutput.beta[i]
                param._error = self.fitOutput.sd_beta[i]
                i += 1
        
        self._parametersCovMatrix = self.fitOutput.cov_beta * self.fitOutput.res_var
        self._R2 = self._R2Compute()
        
    def _R2Compute(self):
        y = self.multiDataset.datasets[self.dependentDatasetIndex]
        average = _np.average(y)
        
        setsX = []
        for i in range(len(y)):
            d = {}
            for varName, dsIndex in self.independentVars.items():
                d[varName] = self.multiDataset.datasets[dsIndex][i]
            setsX += [d]
            
        #The following line raises an error in pylint. The code is ok though
        SSres = _np.sum(_np.array([(y[i] - self.fittedFn(**(setsX[i])))**2 for i in range(len(y))]))
        SStot = _np.sum(_np.array([(point - average)**2 for point in y]))
        return 1 - SSres / SStot

    
    def continueFit(self, iterations = 10):
        self._fitOutput = self._fitObj.restart(iter = iterations)
        self._updateAfterFit()
        
    def dataFrame(self, rounded=True, signifficantDigits=1, separatedError=False, relativeError=False, \
                  transpose=False, saveCSVFile=None, CSVSep=',', CSVDecimal='.'):
        
        R2col = [_np.round(self.R2, 5)]
        reducedChi2col = [_np.round(self.reducedChi2, 5)]
        rowNames = ['$B$']
        if separatedError:
            rowNames += ['$\\Delta B$']
            R2col += ['-']
            reducedChi2col += ['-']
        if relativeError:
            rowNames += ['$\\Delta B$ (rel)']
            R2col += ['-']
            reducedChi2col += ['-']
        rowNames += ['$B_0$', 'Fixed']
        R2col += ['-', '-']
        reducedChi2col += ['-', '-']
        
        colNames = [param.label for param in self.parameters.values()] + ['$R^2$', '$\\chi^2_{red}$']
        
        tblCols = {}

        for colName, param in zip(colNames, self.parameters.values()):
            value = param.value
            error = param.error

            if rounded and error is not None:
                sigDigits = _gf.getSignifficantDigitLocation(error, signifficantDigits)
                value = round(value, sigDigits)
                error = round(error, sigDigits)
            
            if separatedError:
                tblCols[colName] = [value, error if error is not None else '-']
            else:
                tblCols[colName] = [_gf.latexValuePM(value, error) if error is not None else value]
            
            if relativeError:
                tblCols[colName] += [error/value if value != 0 and error is not None else '-']
            
            tblCols[colName] += [param.initialValue]
        
        tblCols['$R^2$'] = R2col
        tblCols['$\\chi^2_{red}$'] = reducedChi2col
        
        table = _pd.DataFrame(tblCols, columns = colNames, index = rowNames)
        table = table.T if transpose else table
        
        if saveCSVFile is not None:
            table.to_csv(saveCSVFile, sep = CSVSep, decimal = CSVDecimal)
        
        return table




class FitGenerator(object):
    def __init__(self, fn, dependentDatasetIndex: int, independentVars: _typing.Any = [], \
                 maxIterations: int = 50, name = None, fitType = _ge.FitMethods.ODR, \
                 useIndVarErrors = True, useDepVarErrors = True, \
                 labels: _typing.Dict[str, str] = {}, units: _typing.Dict[str, str] = {}, \
                 fixed: _typing.Dict[str, bool] = {}):
        
        '''
        independentVars can be:
            [] or {} - auto initialize, use first function variable. 
                    Select first available column on dataset:
                        if dependentDatasetIndex == 0:
                            select 1
                        else:
                            select 0
            [str] - use only those function parameters. Select first n available column on dataset
            {str : int} - link "keys" function parameters to "value" columns on dataset
        '''
        
        self._fn = fn
        
        if dependentDatasetIndex < 0:
            raise ValueError("dependentDatasetIndex has to be a non-negative integer.")
        self.dependentDatasetIndex = int(dependentDatasetIndex)
        
        if maxIterations < 1:
            raise ValueError("maxIterations has to a positive integer.")
        self.maxIterations = int(maxIterations)
        
        self.name = name
        self.fitType = fitType
        self.useIndVarErrors = useIndVarErrors
        self.useDepVarErrors = useDepVarErrors
        
        sig = _inspect.signature(fn)
        self.parameters = _collections.OrderedDict()
        
        for name in independentVars:
            if name not in sig.parameters:
                raise ValueError("Elements of the 'independentVars' specified are not 'fn' arguments.")
        for name in labels:
            if name not in sig.parameters:
                raise ValueError("Elements in 'labels' are not 'fn' arguments.")
        for name in units:
            if name not in sig.parameters:
                raise ValueError("Elements in 'units' are not 'fn' arguments.")
        for name in fixed:
            if name not in sig.parameters:
                raise ValueError("Elements in 'fixed' are not 'fn' arguments.")
        
        self.independentVars = _collections.OrderedDict()
        
        if len(independentVars) == 0:
            #auto initialize to first function variable
            varName = sig.parameters[list(sig.parameters.keys())[0]].name
            column = 1 if self.dependentDatasetIndex == 0 else 0
            self.independentVars[varName] = column
        elif isinstance(independentVars, dict):
            columns = _np.array(list(independentVars.values()))
            if len(_np.unique(columns)) != len(columns):
                raise ValueError("Multiple 'independentVars' dict values bounded to same dataset column.")
            if self.dependentDatasetIndex in columns:
                raise ValueError("'dependentDatasetIndex' in 'independentVars' dict values.")
            for varName in sig.parameters:
                if varName in independentVars:
                    self.independentVars[varName] = independentVars[varName] #load them in order
            self.independentVars = independentVars
        elif isinstance(independentVars, list):
            i = 0
            for varName in sig.parameters:
                if varName in independentVars: #load them in order
                    if self.dependentDatasetIndex == i:
                        i += 1
                    self.independentVars[varName] = i
        else:
            raise TypeError("'independentVars' has wrong data type: list or dict.")
        
        for value in sig.parameters.values():
            if value.name in self.independentVars:
                continue
            
            param = Parameter(value.name)
            name = value.name
            
            if name in labels:
                param.label = labels[name]
            elif type(value.annotation) != type(_inspect._empty) and isinstance(value.annotation, str):
                param.label = value.annotation
            
            if type(value.default) != type(_inspect._empty):
                param.initialValue = value.default
            
            if name in units:
                param.units = units[name]
            
            if name in fixed:
                param.fixed = fixed[name]
            
            self.parameters[name] = param
        
    _fitClass = Fit

    @property
    def fn(self):
        return self._fn
    
    @property
    def name(self):
        return self._name
    @name.setter
    def name(self, value):
        self._name = value if value is not None else ''
    
    def fit(self, multiDataset, maxIterations=None, fitType=None, useIndVarErrors=None, useDepVarErrors=None, \
            initialIndex=None, finalIndex=None, **kwargs) -> Fit:
        parameters = _copy.deepcopy(self.parameters)

        if maxIterations is None:
            maxIterations = self.maxIterations
        elif int(maxIterations) < 1:
            raise ValueError("maxIterations has to a positive integer.")
        
        if fitType is None:
            fitType = self.fitType
            
        if useIndVarErrors is None:
            useIndVarErrors = self.useIndVarErrors
        if useDepVarErrors is None:
            useDepVarErrors = self.useDepVarErrors
                    
        for key, value in kwargs.items():
            if key not in parameters:
                raise ValueError("Arguments passed are not 'fn' arguments.")
            if type(value) != int and type(value) != float:
                parameters[key].fixed = True
            parameters[key].initialValue = value
        
        paramLabels = set()
        for key, value in parameters.items():
            if isinstance(value.initialValue, Parameter.Empty):
                raise ValueError('Some of the function parameters are not initialized (Empty()). Cannot execute fit.')
            if value.fixed:
                value._value = value.initialValue
            if value.label is None:
                _warnings.warn("No label provided for parameter: '" + value.name + "'. Using parameter name instead.", RuntimeWarning)
                value.label = value.name
            if value.label in paramLabels:
                raise Exception('Multiple parameters have the same label.')
            paramLabels.add(value.label)
                

        if not isinstance(initialIndex, int):
            initialIndex = 0
        elif initialIndex not in range(0, len(multiDataset)):
            raise ValueError("'initialIndex' is out of range.")

        if not isinstance(finalIndex, int):
            finalIndex = len(multiDataset)
        elif finalIndex not in range(0, len(multiDataset)):
            raise ValueError("'finalindex' is out of range.")
        else:
            finalIndex += 1 #Correct for Python final list index offset

        Xdsets = multiDataset.datasets[list(self.independentVars.values())]
        Ydset = multiDataset.datasets[self.dependentDatasetIndex]

        XValues = [ds.data[initialIndex:finalIndex] for ds in Xdsets]
        YValues = Ydset.data[initialIndex, finalIndex]
        
        def _generateFitFunction(fn, X, params):
            def ret(beta, x):
                args = {}
                
                i = 0
                for name, param in params.items():
                    if param.fixed:
                        args[name] = param.value
                    else:
                        args[name] = beta[i]
                        i += 1
                
                for i, xName in enumerate(X.keys()):
                    args[xName] = x[i]

                #The following line raises an error in pylint. The code is ok though
                self.fn(**args)
            
            return ret
        
        fn = _generateFitFunction(self.fn, self.independentVars, parameters)
        
        def _odrFit():
            model = _odr.Model(fn)
            covX = None
            if useIndVarErrors:
                if multiDataset.covMatrix is None:
                    _warnings.warn("No covMatrix provided for 'multiDataset'. Attempting to create covMatrices from datasets errors.", RuntimeWarning)
                    multiDataset.generateCovMatricesFromErrors()

                totalColumns = range(0, multiDataset.shape[1])
                independentColumns = list(self.independentVars.values())
                toDelete = list(set(totalColumns) - set(independentColumns))
                toDelete = sorted(toDelete)[::-1]
                covX = []
                for mat in multiDataset.covMatrices[initialIndex:finalIndex]:
                    for elem in toDelete:
                        mat = _np.delete(mat, elem, axis=0)
                        mat = _np.delete(mat, elem, axis=1)
                    covX += [mat]
                    
                covX = np.array(covX)

                for covMatrix in covX:
                    diag = _np.diag(covMatrix)
                    if _np.count_nonzero(diag) != len(diag):
                        raise Exception("Some covMatrices have zeros in the diagonal. Cannot execute fitting with 'useIndVarErrors' == True.")
            
            covX = _gf.covListToODRPACKcovList(covX)

            sY = None
            if useDepVarErrors:
                if Ydset.error is None:
                    raise Exception("No errors provided for the dependent variable. Cannot execute fitting with 'useDepVarErrors' == True.")
                sY = Ydset.error[initialIndex:finalIndex]
                if _np.count_nonzero(sY) != len(sY):
                    raise Exception("Some errors on the dependent variable are zero. Cannot execute fitting with 'useDepVarErrors' == True.")
            
            data = _odr.RealData(XValues, YValues, covx=covX if useIndVarErrors else None, \
                                   sy=sY if useDepVarErrors else None)
            beta0 = [param.initialValue for param in parameters.values() if not param.fixed]
            
            fit = _odr.ODR(data, model, beta0=beta0, maxit=maxIterations)
            fit.set_job(fit_type = 0)
            output = fit.run()
            
            return self._fitClass(self.fn, parameters, multiDataset, self.dependentDatasetIndex, self.independentVars, \
                        self.name, fitType, fit, output, initialIndex, finalIndex)
            
                    
        def _olsFit():
            model = _odr.Model(fn)
            sY = None
            if useDepVarErrors:
                if Ydset.error is None:
                    raise Exception("No errors provided for the dependent variable. Cannot execute fitting with 'useDepVarErrors' == True.")
                sY = Ydset.error[initialIndex:finalIndex]
                if _np.count_nonzero(sY) != len(sY):
                    raise Exception("Some errors on the dependent variable are zero. Cannot execute fitting with 'useDepVarErrors' == True.")
            
            data = _odr.RealData(XValues, YValues, sy=sY if useDepVarErrors else None)
            beta0 = [param.initialValue for param in parameters.values() if not param.fixed]
            
            fit = _odr.ODR(data, model, beta0=beta0, maxit=maxIterations)
            fit.set_job(fit_type = 2)
            output = fit.run()
            
            return self._fitClass(self.fn, parameters, multiDataset, self.dependentDatasetIndex, self.independentVars, \
                        self.name, fitType, fit, output, initialIndex, finalIndex)
        
        
        if fitType == _ge.FitMethods.ODR:
            return _odrFit()
        elif fitType == _ge.FitMethods.OLS:
            if useIndVarErrors:
                _warnings.warn("Independent Variable errors not used in OLS fit.", RuntimeWarning)
                useIndVarErrors = False
            return _olsFit()
        
    def __str__(self):
        ret = "Fit Generator:"
        ret += "\n\tFn: " + str(self.fn)
        if self.name is not None:
            ret += "\n\tName: '" + self.name + "'"
        ret += "\n\tIndependent Vars: " + str(self.independentVars)
        ret += "\n\tParameters:"
        ret += '\n'
        for value in self.parameters.values():
            ret += _gf.tabulateStrBlock(str(value), 2)
            ret += '\n'
        return ret
        

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#External functions:

def reportManyFits(fitList, fitNames=None, rounded=True, signifficantDigits=1, \
                    separatedError=False, relativeError=False, \
                    saveCSVFile=None, CSVSep=',', CSVDecimal='.'):

    paramNumber = len(fitList[0].parameters)
    for i, fit in enumerate(fitList):
        if len(fit.parameters) != paramNumber:
            raise Exception("'fit[" + str(i) + "]' in 'fitList' has different number of fitted parameters.")
    
    if fitNames is None:
        flag = False
        for fit in fitList:
            if fit.name != '' or fit.data.name != '':
                flag = True
        if flag:
            fitNames = [fit.prettyName for fit in fitList]
    elif len(fitNames) != len(fitList):
        raise Exception("len(fitNames) != len(fitList)")
    
    rows = []
    for index, fit in enumerate(fitList):
        columns = []

        if fitNames is not None:
            columns += [fitNames[index]]
        
        for param in fit.parameters.values():
            value = param.valu
            error = param.error

            if rounded and error is not None:
                sigDigits = _gf.getSignifficantDigitLocation(error, signifficantDigits)
                value = round(value, sigDigits)
                error = round(error, sigDigits)

            if separatedError:
                columns += [value, error if error is not None else '-']
            else:
                columns += [_gf.latexValuePM(value, error) if error is not None else value]
            
            if relativeError:
                columns += [error/value if value != 0 or error is not None else '-']
        
        columns += [_np.round(fit.R2, 5), _np.round(fit.reducedChi2, 5)]
        
        rows += [columns]
    
    colNames = ['Name'] if fitNames is not None else []
    
    paramNames = [param.prettyLabel for param in fitList[0].parameters]
    
    for paramName, param in zip(paramNames, fitList[0].parameters):
        colNames += [paramName]
        if separatedError:
            colNames += ['$\\Delta$ ' + paramName]
        if relativeError:
            colNames += ['$\\Delta$ ' + param.label + ' (rel)']
    
    colNames += ['$R^2$', '$\\chi^2_{red}$']

    table = _pd.DataFrame(rows, columns=colNames)
        
    if saveCSVFile is not None:
        table.to_csv(saveCSVFile, sep=CSVSep, decimal=CSVDecimal)
    
    return table
