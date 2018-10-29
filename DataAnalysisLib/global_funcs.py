import math as _math

import numpy as _np
import pandas as _pd
import re as _re


def cleanText(t):
    #Cleans text from LaTeX equation wrapper ($)
    t = t[1:] if t.startswith('$') else t
    return t[:-1] if t.endswith('$') else t
    
def findNearestValueIndex(array, value):
    #returns index of nearest element to value in array
    return (_np.abs(_np.asarray(array) - value)).argmin()

def getSignifficantDigitLocation(value, signifficantDigits=1):
    if signifficantDigits < 1:
        raise ValueError("'numberOfSignifficantDigits' has to be a positive integer.")
    v = -int(_np.floor(_np.log10(abs(value)))) if value != 0 else 0
    v = v + (signifficantDigits - 1)
    return v

def latexValuePM(value, error):
    return str(value) + ' $\\pm$ ' + str(error)

def createSeriesPanda(values, error=None, label='$x$', units=None, relativeError=False, \
                        separated=False, rounded=True, signifficantDigits=1):
    #Returns a panda DataFrame with values and error, formatted according to function params 
    pLabel = label + ' (' + units + ')' if units is not None else label
    pErrorLabel = '$\\Delta$ ' + pLabel

    if error is None:
        return _pd.DataFrame(values, columns = [pLabel])
    else:
        col = None
        sigDigits = [getSignifficantDigitLocation(x, signifficantDigits) for x in error]
        perrors = [round(x, s) for x, s in zip(error, sigDigits)] if rounded else error
        pvalues = [round(x, s) for x, s in zip(values, sigDigits)] if rounded else values
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

def _conv(obj, dtype=None):
    #Converts obj to a numpy array if possible (obj is not None and is not a scalar)
    if obj is None:
        return obj
    else:
        if dtype is None:
            obj = _np.asarray(obj)
        else:
            obj = _np.asarray(obj, dtype)
            
        if obj.shape == ():
            # Scalar.
            return obj.dtype.type(obj)
        else:
            return obj

def tabulateStrBlock(s, times=1):
    l = [m.start() for m in _re.finditer('\t', s)]
    for index in reversed(l):
        s = s[:index] + '\t' + s[index:]
    s = '\t' + s
    return s if times == 1 else tabulateStrBlock(s, times - 1)


def covListToODRPACKcovList(covList):
    if len(covList.shape) != 3:
        raise ValueError("'covList' is not an 'ndarray' of shape (p, p, n).")
    n = covList.shape[0]
    p = covList.shape[1]

    c = _np.zeros(shape=(p, p, n))

    for i in range(n):
        for j in range(p):
            for k in range(p):
                c[k, j, i] = covList[i, j, k]
    
    return c

    