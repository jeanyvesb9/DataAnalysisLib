import warnings
import functools

import numpy as np
import scipy.odr as odr
import matplotlib.pyplot as plt
import pandas as pd

import global_funcs
import global_enums


class DataFit(object):
    def __init__(self, data, fn, initialParams, paramNames = None, paramUnits = None, method = global_enums.FitMethods.ODR):
        self.data = data
        self.fn = fn
        self.initialParams = initialParams
        self.method = method

        xError = self.data.xError if np.count_nonzero(self.data.xError) != 0 else None
        yError = self.data.yError if np.count_nonzero(self.data.yError) != 0 else None

        self.fitObj = None
        self.fitParams = None
        self.fitParamsStdError = None
        self.reducedChi2 = None
        self.R2 = None

        if self.method == global_enums.FitMethods.ODR:
            Rdata = odr.RealData(self.data.x, self.data.y, xError, yError)
            self.fitObj = odr.ODR(Rdata, odr.Model(self.fn), self.initialParams).run()
        
            self.fitParams = self.fitObj.beta
            self.fitParamsStdError = self.fitObj.sd_beta

            #The following 2 lines raise warnings in pylint. The code is OK tough.
            self.reducedChi2 = self.fitObj.res_var #See http://mail.scipy.org/pipermail/scipy-user/2012-May/032207.html
            self.R2 = 1 - np.sum(self.fitObj.eps**2)/self.fitObj.sum_square if np.argwhere(np.array(self.fitObj.sd_beta) == 0).size == 0  else 1

        self.paramNames = ['$B_{' + str(i) + '}$' for i in range(len(self.fitParams))]
        if paramNames is not None:
            if len(paramNames) != len(self.fitParams):
                warnings.warn('len(paramsName) != len(fitParams): Default parameter names selected.')
            else:
                seen = set()
                flag = False
                for name in paramNames:
                    if name not in seen:
                        seen.add(name)
                    else:
                        flag = True
                if flag:
                    warnings.warn('Found repeated values in paramNames: Default parameter names selected.')
                else:
                    self.paramNames = paramNames
        self.paramUnits = paramUnits

    def getFitFn(self):
        return functools.partial(self.fn, self.fitParams)
    
    def quickPlot(self, plotType = global_enums.PlotType.ErrorBar, purgeStep = 1):
        if purgeStep <= 0:
            warnings.warn('purgeStep has to be at least 1. Setting purgeStep = 1.')
            purgeStep = 1
        fig , ax = plt.subplots(1,1)
        if plotType == global_enums.PlotType.ErrorBar:
            ax.errorbar(self.data.x[::purgeStep], self.data.y[::purgeStep], xerr = self.data.xError[::purgeStep], \
                        yerr = self.data.yError[::purgeStep], fmt = 's')
        elif plotType == global_enums.PlotType.Line:
            ax.plot(self.data.x[::purgeStep], self.data.y[::purgeStep], '-')
        elif plotType == global_enums.PlotType.Point:
            ax.plot(self.data.x[::purgeStep], self.data.y[::purgeStep], 's')

        x = np.linspace(self.data.x[0], self.data.x[-1], 1000)
        ax.plot(x, self.getFitFn()(x))

        ax.set_xlabel(self.data.xLabel if self.data.xUnits is None else self.data.xLabel + ' (' + self.data.xUnits + ')')
        ax.set_ylabel(self.data.yLabel if self.data.yUnits is None else self.data.yLabel + ' (' + self.data.yUnits + ')')
        ax.set_title(self.data.name)
        return fig, ax

    def dataFrame(self, rounded = True, separatedError = False, relativeError = False, saveCSVFile = None, CSVSep = ',', CSVDecimal = '.'):
        perrors = [global_funcs.roundToFirstSignifficantDigit(x) for x in self.fitParamsStdError] if rounded else self.fitParamsStdError
        pvalues = [global_funcs.roundToError(self.fitParams[i], perrors[i]) for i in range(len(self.fitParams))] if rounded else self.fitParams
        
        R2col = [np.round(self.R2, 5)]
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
        colNames.append('$R^2$')
        
        tblCols = {}
        for i in range(len(pvalues)):
            if relativeError:
                relError = perrors[i]/pvalues[i] if pvalues[i] != 0 else '-'
                if separatedError:
                    tblCols[colNames[i]] = [pvalues[i], perrors[i], relError, self.initialParams[i]]
                else:
                    tblCols[colNames[i]] = [str(pvalues[i]) + ' +/- ' + str(perrors[i]), relError, self.initialParams[i]]
            else:
                if separatedError:
                    tblCols[colNames[i]] = [pvalues[i], perrors[i], self.initialParams[i]]
                else:
                    tblCols[colNames[i]] = [str(pvalues[i]) + ' +/- ' + str(perrors[i]), self.initialParams[i]]
                    
        tblCols['$R^2$'] = R2col
        
        table = pd.DataFrame(tblCols, columns = colNames, index = rowNames)
        
        if saveCSVFile is not None:
            table.to_csv(saveCSVFile, sep = CSVSep, decimal = CSVDecimal)
        
        return table
