import warnings

import numpy as np
import pandas as pd

import dataset

class PascoParser(object):
    def __init__(self, csvFile, numberOfColumnsPerSeries, separator = ',', decimal = '.'):
        self.data = pd.read_csv(csvFile, separator, decimal = decimal)
        self.numberOfColumnsPerSeries = numberOfColumnsPerSeries
        self.decimal = decimal
        
        numberOfSeries = int(len(self.data.columns)/numberOfColumnsPerSeries)
        
        self.series = []
        for s in range(numberOfSeries):
            df = pd.DataFrame(self.data[self.data.columns[s * self.numberOfColumnsPerSeries : (s + 1) * self.numberOfColumnsPerSeries]])
            df.columns = [df[colName][0] for colName in df.columns]
            df = df.drop(0)
            df.index = range(len(df.index))
            df = df.dropna(how = 'all')
            self.series.append(df)

    def getColumnsFromSeries(self, seriesNumber, xCol, yCol):
        df = self.series[seriesNumber]
        dftemp = pd.DataFrame(df[[df.columns[xCol], df.columns[yCol]]]).dropna()
        xCol = dftemp[df.columns[xCol]]
        yCol = dftemp[df.columns[yCol]]
        return (np.array([s.replace(self.decimal, '.') for s in xCol], 'float64'), np.array([s.replace(self.decimal, '.') for s in yCol], 'float64'))

    @staticmethod
    def parseColumnName(colName):
        pIndex = colName.find('(')
        if pIndex == -1:
            return colName, ''
        else:
            return colName[:pIndex - 1], colName[pIndex + 1 : -1]

    def makeDataSet(self, seriesNumber, xCol, yCol, xError = None, yError = None, xErrorFn = None, yErrorFn = None, autoLabel = True, \
                    xLabel = None, yLabel = None, xUnits = None, yUnits = None, name = ''):
        x, y = self.getColumnsFromSeries(seriesNumber, xCol, yCol)

        if autoLabel:
            if xLabel is not None or yLabel is not None or xUnits is not None or yUnits is not None:
                warnings.warn('autoLabel selected and manual label/units paramenters set. Defaulting to manual label parameters where available.')
            
            xl, xu = self.parseColumnName(self.series[seriesNumber].columns[xCol])
            yl, yu = self.parseColumnName(self.series[seriesNumber].columns[yCol])
            
            xLabel = xl if xLabel is None else xLabel
            xUnits = xu if xUnits is None or xUnits == '' else xUnits
            yLabel = yl if yLabel is None else yLabel
            yUnits = yu if yUnits is None or xUnits == '' else yUnits

        return dataset.DataSet(x, y, xError, xErrorFn, yError, yErrorFn, xLabel, yLabel, xUnits, yUnits, name)
