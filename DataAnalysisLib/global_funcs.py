import math
import warnings

import numpy as np
import pandas as pd


def cleanText(t):
    #Cleans text from LaTeX equation preamble ($)
    t = t[1:] if t.startswith('$') else t
    return t[:-1] if t.endswith('$') else t
    
def findNearestValueIndex(array, value):
    #returns index of nearest element to value in array
    return (np.abs(np.asarray(array) - value)).argmin()

def roundToFirstSignifficantDigit(value):
    #returns error rounded to first signifficant digit
    return round(value, -int(math.floor(math.log10(abs(value))))) if value != 0 else 0

def roundToError(value, error):
    #returns value rounded to error's first signifficant digit
    s = np.array(list(str('{0:.1000f}').format(abs(error))))
    if error == 0:
        return value
    elif s[0] == '0':
        d = np.where(s[2:] != '0')[0][0]
        return round(value, d + 1)
    else:
        error = int(error)
        digits = int(math.log10(error))+1
        return round(int(value), -(digits))

def createSeriesPanda(values, error = None, label = '$x$', unit = None, relativeError = False, separated = False, rounded = True):
    #Returns a panda DataFrame with values and error, formatted according to function params 
    pLabel = label + ' (' + unit + ')' if unit is not None else label
    pErrorLabel = '$\\Delta$ ' + label
    pErrorLabel = pErrorLabel + ' (' + unit + ')' if unit is not None else pErrorLabel

    if error is None or np.count_nonzero(error) == 0:
        return pd.DataFrame([values], columns = [pLabel])
    else:
        col = None
        perror = [roundToFirstSignifficantDigit(x) for x in error] if rounded else error
        pvalues = [roundToError(values[i], perror[i]) for i in range(len(values))] if rounded else values
        if separated:
            col = pd.DataFrame({pLabel: pvalues, pErrorLabel: perror}, columns = [pLabel, pErrorLabel])
        else:
            col = pd.DataFrame([str(pvalues[i]) + ' +/- ' + str(perror[i]) for i in range(len(pvalues))], columns = [pLabel])
        
        if relativeError:
            relErrors = pd.DataFrame([error[i]/values[i] if values[i] != 0 else '-' for i in range(len(values))], \
                                        columns = ['$\\Delta$ ' + label + ' (rel)'])
            return pd.concat([col, relErrors], axis = 1, join = 'inner')
        else:
            return col

def reportManyFits(fitList, fitNames = None, rounded = True, separatedError = False, relativeError = False, saveCSVFile = None, CSVSep = ',', CSVDecimal = '.'):
    paramNumber = len(fitList[0].fitParams)
    for i in range(len(fitList)):
        if len(fitList[i].fitParams) != paramNumber:
            warnings.warn('fit[' + str(i) + '] in fitList has different number of fitted parameters.')
            return None
    
    if fitNames is None:
        flag = False
        for fit in fitList:
            if fit.data.name != '':
                flag = True
        if flag:
            fitNames = [fit.data.name for fit in fitList]
    elif len(fitNames) != len(fitList):
        warnings.warn('len(fitNames) != len(fitList): Fit names removed from table.')
        fitNames = None
    
    rows = []
    for index in range(len(fitList)):
        fit = fitList[index]

        perrors = [roundToFirstSignifficantDigit(x) for x in fit.fitParamsStdError] if rounded else fit.fitParamsStdError
        pvalues = [roundToError(fit.fitParams[i], perrors[i]) for i in range(len(fit.fitParams))] if rounded else fit.fitParams

        columns = []
        if fitNames is not None:
            columns += [fitNames[index]]

        for i in range(len(pvalues)):
            if separatedError:
                columns += [pvalues[i], perrors[i]]
            else:
                columns += [str(pvalues[i]) + ' +/- ' + str(perrors[i])]
            if relativeError:
                columns += [perrors[i]/pvalues[i] if pvalues[i] != 0 else '-']
        
        columns += [np.round(fit.R2, 5)]
        
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

    table = pd.DataFrame(rows, columns = colNames)
        
    if saveCSVFile is not None:
        table.to_csv(saveCSVFile, sep = CSVSep, decimal = CSVDecimal)
    
    return table
