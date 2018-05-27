import math as _math

import numpy as _np
import pandas as _pd


def cleanText(t):
    #Cleans text from LaTeX equation wrapper ($)
    t = t[1:] if t.startswith('$') else t
    return t[:-1] if t.endswith('$') else t
    
def findNearestValueIndex(array, value):
    #returns index of nearest element to value in array
    return (_np.abs(_np.asarray(array) - value)).argmin()

def roundToFirstSignifficantDigit(value):
    #returns error rounded to first signifficant digit
    return round(value, -int(_math.floor(_math.log10(abs(value))))) if value != 0 else 0

def roundToError(value, error):
    #returns value rounded to error's first signifficant digit
    s = _np.array(list(str('{0:.1000f}').format(abs(error))))
    if error == 0:
        return value
    elif s[0] == '0':
        d = _np.where(s[2:] != '0')[0][0]
        return round(value, d + 1)
    else:
        error = int(error)
        digits = int(_math.log10(error))+1
        return round(int(value), -(digits))

def latexValuePM(value, error):
    return str(value) + ' $\\pm$ ' + str(error)

def createSeriesPanda(values, error = None, label = '$x$', units = None, relativeError = False, separated = False, rounded = True):
    #Returns a panda DataFrame with values and error, formatted according to function params 
    pLabel = label + ' (' + units + ')' if units is not None else label
    pErrorLabel = '$\\Delta$ ' + pLabel

    if error is None:
        return _pd.DataFrame(values, columns = [pLabel])
    else:
        col = None
        perrors = [roundToFirstSignifficantDigit(x) for x in error] if rounded else error
        pvalues = [roundToError(values[i], perrors[i]) for i in range(len(values))] if rounded else values
        if separated:
            col = _pd.DataFrame({pLabel: pvalues, pErrorLabel: perrors}, columns = [pLabel, pErrorLabel])
        else:
            col = _pd.DataFrame([latexValuePM(pvalues[i], perrors[i]) for i in range(len(pvalues))], columns = [pLabel])
        
        if relativeError:
            relErrors = _pd.DataFrame([error[i]/values[i] if values[i] != 0 else '-' for i in range(len(values))], \
                                        columns = ['$\\Delta$ ' + label + ' (rel)'])
            return _pd.concat([col, relErrors], axis = 1, join = 'inner')
        else:
            return col

def R2(x, y, fn):
    average = _np.average(y)
    SSres = _np.sum(_np.array([(y[i] - fn(x[i]))**2 for i in range(len(x))]))
    SStot = _np.sum(_np.array([(point - average)**2 for point in y]))
    return 1 - SSres / SStot