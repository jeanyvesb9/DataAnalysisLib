import warnings as _warnings
import functools as _functools

import numpy as _np
import scipy.odr as _odr
import matplotlib.pyplot as _plt
import pandas as _pd

from . import global_funcs as _gf
from . import global_enums as _ge


class XYFit(object):
    def __init__(self, data, fn, initialParams, paramNames = None, paramUnits = None, initialXIndex = None, finalXIndex = None, name = '', method = _ge.FitMethods.ODR):
        self._data = data
        self._fn = fn
        self._initialParams = initialParams
        self._method = method

        self.name = name #use setter for None-value checking
        self._paramNames = None
        self._paramUnits = None

        if initialXIndex is None:
            self._initialXIndex = 0
        elif initialXIndex < 0 or initialXIndex >= len(self.data.x):
            _warnings.warn('initialXIndex given is out of range. Default value selected.')
            self._initialXIndex = 0

        if finalXIndex is None:
            self._finalXIndex = len(self.data.x) - 1
        elif finalXIndex < 0 or finalXIndex >= len(self.data.x):
            _warnings.warn('finalXIndex given is out of range. Default value selected.')
            self._finalXIndex = len(self.data.x) - 1

        self._fitObj = None
        self._fitParams = None
        self._fitParamsStdError = None
        self._reducedChi2 = None
        self._R2 = None

        x = self.data.x[self.initialXIndex:self.finalXIndex + 1]
        y = self.data.y[self.initialXIndex:self.finalXIndex + 1]

        xError = self.data.xError[self.initialXIndex:self.finalXIndex + 1] if self.data.xError is not None else None
        yError = self.data.yError[self.initialXIndex:self.finalXIndex + 1] if self.data.yError is not None else None

        def _odrFit():
            Rdata = _odr.RealData(x, y, sx = xError, sy = yError)
            self._fitObj = _odr.ODR(Rdata, _odr.Model(self.fn), self.initialParams).run()
        
            self._fitParams = self.fitObj.beta
            self._fitParamsStdError = self.fitObj.sd_beta
            self._R2 = _gf.R2(x, y, self.fittedFn)

            #The following line raises warnings in pylint. The code is OK tough.
            self._reducedChi2 = self.fitObj.res_var #See http://mail.scipy.org/pipermail/scipy-user/2012-May/032207.html

        def _olsFit():
            if yError is not None and _np.count_nonzero(yError) != len(yError):
                _warnings.warn('data.yError contains 0: executing OLS fitting without yError instead.')
                return
            Rdata = _odr.RealData(x, y, sy = yError)
            odrObj = _odr.ODR(Rdata, _odr.Model(self.fn), self.initialParams)
            odrObj.set_job(fit_type = 2)
            self._fitObj = odrObj.run()
        
            self._fitParams = self.fitObj.beta
            self._fitParamsStdError = self.fitObj.sd_beta
            self._R2 = _gf.R2(x, y, self.fittedFn)

            #The following line raises warnings in pylint. The code is OK tough.
            self._reducedChi2 = self.fitObj.res_var #See http://mail.scipy.org/pipermail/scipy-user/2012-May/032207.html

            #For an OLS fit, ODR scales the covariance matrix by res_var=reducedChi2, so we need to get rid of that scaling:
            self._fitObj.cov_beta = self._fitObj.cov_beta * self._reducedChi2

        if self.method == _ge.FitMethods.ODR:
            if xError is not None and _np.count_nonzero(xError) == len(xError):
                _odrFit()
            else:
                _warnings.warn('data.xError contains 0: executing OLS fitting instead.')
                _olsFit()
        elif self.method == _ge.FitMethods.OLS:
                _olsFit()
        
        self.paramNames = paramNames
        self.paramUnits = paramUnits


    @property
    def data(self):
        return self._data
    
    @property
    def fn(self):
        return self._fn

    @property
    def initialParams(self):
        return self._initialParams
    
    @property
    def method(self):
        return self._method
    
    #Idiot proofing the library:

    @property
    def name(self):
        return self._name
    @name.setter
    def name(self, value):
        self._name = value if value is not None else ''

    @property
    def paramNames(self):
        return self._paramNames
    @paramNames.setter
    def paramNames(self, value):
        if self.fitParams is None:
            return
        if value is not None:
            if len(value) != len(self.fitParams):
                _warnings.warn('len(paramNames) != len(fitParams): Default parameter names selected.')
                self._paramNames = ['$B_{' + str(i) + '}$' for i in range(len(self.fitParams))]
            else:
                seen = set()
                flag = False
                for name in value:
                    if name in seen:
                        flag = True
                        break
                    else:
                        seen.add(name)
                if flag:
                    _warnings.warn('Found repeated values in paramNames: Default parameter names selected.')
                    self._paramNames = ['$B_{' + str(i) + '}$' for i in range(len(self.fitParams))]
                else:
                    self._paramNames = value
        else:
            self._paramNames = ['$B_{' + str(i) + '}$' for i in range(len(self.fitParams))]

    @property
    def paramUnits(self):
        return self._paramUnits
    @paramUnits.setter
    def paramUnits(self, value):
        if self.fitParams is None:
            return
        if value is not None:
            if len(value) != len(self.fitParams):
                _warnings.warn('len(paramUnits) != len(fitParams): paramUnits removed.')
                self._paramUnits = None
            else:
                self._paramUnits = value
        else:
            self._paramUnits = None
    
    #End of idiot proofing.

    @property
    def initialXIndex(self):
        return self._initialXIndex
    
    @property
    def finalXIndex(self):
        return self._finalXIndex

    def prettyName(self):
        return self.data.name + (' - ' if self.name != '' else '') + self.name

    @property
    def fitObj(self):
        return self._fitObj

    @property
    def fitParams(self):
        return self._fitParams

    @property
    def fitParamsStdError(self):
        return self._fitParamsStdError
    
    @property
    def reducedChi2(self):
        return self._reducedChi2
    
    @property
    def R2(self):
        return self._R2

    @property
    def fittedFn(self):
        return _functools.partial(self.fn, self.fitParams)
    
    def quickPlot(self, plotType = _ge.PlotType.ErrorBar, purgeStep = 1, initialXIndex = None, finalXIndex = None):
        if purgeStep <= 0:
            _warnings.warn('purgeStep has to be at least 1. Setting purgeStep = 1.')
            purgeStep = 1

        if initialXIndex is None:
            initialXIndex = 0
        elif initialXIndex < 0 or initialXIndex >= len(self.data.x):
            _warnings.warn('initialXIndex given is out of range. Default value selected.')
            initialXIndex = 0

        if finalXIndex is None:
            finalXIndex = len(self.data.x)
        elif finalXIndex < 0 or finalXIndex >= len(self.data.x):
            _warnings.warn('finalXIndex given is out of range. Default value selected.')
            finalXIndex = len(self.data.x)

        fig , ax = _plt.subplots(1,1)

        if plotType == _ge.PlotType.ErrorBar:
            ax.errorbar(self.data.x[initialXIndex:finalXIndex:purgeStep], self.data.y[initialXIndex:finalXIndex:purgeStep], \
            xerr = self.data.xError[initialXIndex:finalXIndex:purgeStep] if self.data.xError is not None else None, \
            yerr = self.data.yError[initialXIndex:finalXIndex:purgeStep] if self.data.yError is not None else None, fmt = 's')
        elif plotType == _ge.PlotType.Line:
            ax.plot(self.data.x[initialXIndex:finalXIndex:purgeStep], self.data.y[initialXIndex:finalXIndex:purgeStep], '-')
        elif plotType == _ge.PlotType.Point:
            ax.plot(self.data.x[initialXIndex:finalXIndex:purgeStep], self.data.y[initialXIndex:finalXIndex:purgeStep], 's')

        x = _np.linspace(self.data.x[initialXIndex], self.data.x[finalXIndex - 1], 1000) #correct finalXIndex for initial correction above.

        #The following line raises warnings in pylint. The code is OK tough.
        ax.plot(x, self.fittedFn(x))

        ax.set_xlabel(self.data.prettyXLabel())
        ax.set_ylabel(self.data.prettyYLabel())
        ax.set_title(self.prettyName())
        return fig, ax

    def dataFrame(self, rounded = True, separatedError = False, relativeError = False, transpose = False, saveCSVFile = None, CSVSep = ',', CSVDecimal = '.'):
        perrors = [_gf.roundToFirstSignifficantDigit(x) for x in self.fitParamsStdError] if rounded else self.fitParamsStdError
        pvalues = [_gf.roundToError(self.fitParams[i], perrors[i]) for i in range(len(self.fitParams))] if rounded else self.fitParams
        
        R2col = [_np.round(self.R2, 5)]
        rowNames = ['B']
        if separatedError:
            rowNames += ['$\\Delta B$']
            R2col += ['-']
        if relativeError:
            rowNames += ['$\\Delta B$ (rel)']
            R2col += ['-']
        rowNames += ['$B_0$']
        R2col += ['-']
        
        colNames = []
        if self.paramUnits is not None:
            colNames = [self.paramNames[i] + ' (' + self.paramUnits[i] + ')' if self.paramUnits[i] != '' \
                        else self.paramNames[i] for i in range(len(self.paramNames))]
        else:
            colNames = self.paramNames
        colNames += ['$R^2$']
        
        tblCols = {}
        for i in range(len(pvalues)):
            if relativeError:
                relError = perrors[i]/pvalues[i] if pvalues[i] != 0 else '-'
                if separatedError:
                    tblCols[colNames[i]] = [pvalues[i], perrors[i], relError, self.initialParams[i]]
                else:
                    tblCols[colNames[i]] = [_gf.latexValuePM(pvalues[i], perrors[i]), relError, self.initialParams[i]]
            else:
                if separatedError:
                    tblCols[colNames[i]] = [pvalues[i], perrors[i], self.initialParams[i]]
                else:
                    tblCols[colNames[i]] = [_gf.latexValuePM(pvalues[i], perrors[i]), self.initialParams[i]]
                    
        tblCols['$R^2$'] = R2col
        
        table = _pd.DataFrame(tblCols, columns = colNames, index = rowNames)
        table = table.T if transpose else table
        
        if saveCSVFile is not None:
            table.to_csv(saveCSVFile, sep = CSVSep, decimal = CSVDecimal)
        
        return table

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#External functions:

def reportManyFits(fitList, fitNames = None, rounded = True, separatedError = False, relativeError = False, saveCSVFile = None, CSVSep = ',', CSVDecimal = '.'):
    paramNumber = len(fitList[0].fitParams)
    for i in range(len(fitList)):
        if len(fitList[i].fitParams) != paramNumber:
            _warnings.warn('fit[' + str(i) + '] in fitList has different number of fitted parameters.')
            return None
    
    if fitNames is None:
        flag = False
        for fit in fitList:
            if fit.name != '' or fit.data.name != '':
                flag = True
        if flag:
            fitNames = [fit.prettyName for fit in fitList]
    elif len(fitNames) != len(fitList):
        _warnings.warn('len(fitNames) != len(fitList): Fit names removed from table.')
        fitNames = None
    
    rows = []
    for index in range(len(fitList)):
        fit = fitList[index]

        perrors = [_gf.roundToFirstSignifficantDigit(x) for x in fit.fitParamsStdError] if rounded else fit.fitParamsStdError
        pvalues = [_gf.roundToError(fit.fitParams[i], perrors[i]) for i in range(len(fit.fitParams))] if rounded else fit.fitParams

        columns = []
        if fitNames is not None:
            columns += [fitNames[index]]

        for i in range(len(pvalues)):
            if separatedError:
                columns += [pvalues[i], perrors[i]]
            else:
                columns += [_gf.latexValuePM(pvalues[i], perrors[i])]
            if relativeError:
                columns += [perrors[i]/pvalues[i]if pvalues[i] != 0 else '-']
        
        columns += [_np.round(fit.R2, 5)]
        
        rows += [columns]
    
    colNames = ['Name'] if fitNames is not None else []
    
    paramNames = []
    if fitList[0].paramUnits is not None:
        paramNames = [fitList[0].paramNames[i] + ' (' + fitList[0].paramUnits[i] + ')' if fitList[0].paramNames[i] != '' \
                        else fitList[0].paramNames[i] for i in range(len(fitList[0].paramNames))]
    else:
        paramNames = fitList[0].paramNames
    
    for i in range(len(paramNames)):
        colNames += [paramNames[i]]
        if separatedError:
            colNames += ['$\\Delta$ ' + paramNames[i]]
        if relativeError:
            colNames += ['$\\Delta$ ' + fitList[0].paramNames[i] + ' (rel)']
    
    colNames += ['$R^2$']

    table = _pd.DataFrame(rows, columns = colNames)
        
    if saveCSVFile is not None:
        table.to_csv(saveCSVFile, sep = CSVSep, decimal = CSVDecimal)
    
    return table
