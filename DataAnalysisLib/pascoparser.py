import warnings as _warnings

import numpy as _np
import pandas as _pd

import xydataset as _xyds
import dataset as _ds

class PascoParser(object):
    def __init__(self, csvFile, numberOfColumnsPerSeries, separator = ',', decimal = '.'):
        self._data = _pd.read_csv(csvFile, separator, decimal = decimal)
        self._numberOfColumnsPerSeries = numberOfColumnsPerSeries
        self._decimal = decimal
        
        numberOfSeries = int(len(self.data.columns)/numberOfColumnsPerSeries)
        
        self._series = []
        self._seriesNames = []
        for s in range(numberOfSeries):
            df = _pd.DataFrame(self.data[self.data.columns[s * self.numberOfColumnsPerSeries : (s + 1) * self.numberOfColumnsPerSeries]])
            df.columns = [df[colName][0] for colName in df.columns]
            self._seriesNames += [self.data.columns[s*self.numberOfColumnsPerSeries]]
            df = df.drop(0)
            df.index = range(len(df.index))
            df = df.dropna(how = 'all')
            self._series += [df]

    @property
    def data(self):
        return self._data
    
    @property
    def numberOfColumnsPerSeries(self):
        return self._numberOfColumnsPerSeries
    
    @property
    def decimal(self):
        return self._decimal
    
    @property
    def series(self):
        return self._series

    @property
    def seriesNames(self):
        return self._seriesNames

    @staticmethod
    def parseColumnName(colName):
        pIndex = colName.rfind('(')
        if pIndex == -1:
            return colName, ''
        else:
            return colName[:pIndex - 1], colName[pIndex + 1 : -1]

    def getColumnDataset(self, seriesNumber, column, error = None, errorFn = None, autoLabel = True, name = None, units = None):
        df = self.series[seriesNumber]
        col = df[df.columns[column]].dropna()

        if autoLabel:
            if name is not None or units is not None:
                _warnings.warn('autoLabel selected and manual name/units paramenters set. Defaulting to manual name parameters where available.')
            
            n, u = self.parseColumnName(df.columns[column])
            
            name = n if name is None else name
            units = u if units is None else units
        
        return _ds.Dataset(_np.array([s.replace(self.decimal, '.') if isinstance(s, str) else s for s in col], 'float64'), error, errorFn, name, units)

    def getXYDataset(self, seriesNumber, xCol, yCol, xError = None, yError = None, xErrorFn = None, yErrorFn = None, autoLabel = True, \
                    xLabel = None, yLabel = None, xUnits = None, yUnits = None, name = None):

        df = self.series[seriesNumber]
        col = df[[df.columns[xCol]] + [df.columns[yCol]]].dropna()
        
        if autoLabel:
            if name is not None or xLabel is not None or yLabel is not None or xUnits is not None or yUnits is not None:
                _warnings.warn('autoLabel selected and manual name/units paramenters set. Defaulting to manual name parameters where available.')
            
            xl, xu = self.parseColumnName(df.columns[xCol])
            yl, yu = self.parseColumnName(df.columns[yCol])
            
            xLabel = xl if xLabel is None else xLabel
            xUnits = xu if xUnits is None else xUnits
            yLabel = yl if yLabel is None else yLabel
            yUnits = yu if yUnits is None else yUnits
            name = self.seriesNames[seriesNumber] if name is None else name

        return _xyds.XYDataset(_np.array([s.replace(self.decimal, '.') if isinstance(s, str) else s for s in col[df.columns[xCol]]], 'float64'), \
                                _np.array([s.replace(self.decimal, '.') if isinstance(s, str) else s for s in col[df.columns[yCol]]], 'float64'), \
                                xError = xError, yError = yError, xErrorFn = xErrorFn, yErrorFn = yErrorFn, \
                                xLabel = xLabel, yLabel = yLabel, xUnits = xUnits, yUnits = yUnits, name = name)
