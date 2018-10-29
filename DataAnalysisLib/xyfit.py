import warnings as _warnings
import functools as _functools

import numpy as _np
import scipy.odr as _odr
import matplotlib.pyplot as _plt
import pandas as _pd

from . import global_funcs as _gf
from . import global_enums as _ge
from . import fit as _fit


class XYFit(_fit.Fit):
    def quickPlot(self, plotType=_ge.PlotType.ErrorBar, purgeStep=1, initialIndex=None, finalIndex=None, \
                    dataPlotArgs={}, functionPlotArgs={}):
        
        if isinstance(purgeStep, int):
            if purgeStep not in range(1, len(self.data.x)):
                raise ValueError("'purgeStep' is out of range (1 <= purgeStep <= len(x)).")
        else:
            raise TypeError("'purgeStep' type is not int")
            
        if not isinstance(initialIndex, int):
            initialIndex = 0
        elif initialIndex not in range(0, len(self.data)):
            raise ValueError("'initialIndex' is out of range.")

        if not isinstance(finalIndex, int):
            finalIndex = len(self.data)
        elif finalIndex not in range(0, len(self.data)):
            raise ValueError("'finalindex' is out of range.")
        else:
            finalIndex += 1 #Correct for Python final list index offset


        fig , ax = _plt.subplots(1,1)

        if plotType == _ge.PlotType.ErrorBar:
            ax.errorbar(self.data.x[initialIndex:finalIndex:purgeStep], \
                        self.data.y[initialIndex:finalIndex:purgeStep], \
                        xerr=self.data.xError[initialIndex:finalIndex:purgeStep] if self.data.xError is not None else None, \
                        yerr=self.data.yError[initialIndex:finalIndex:purgeStep] if self.data.yError is not None else None, \
                        fmt = 's', **dataPlotArgs)
        elif plotType == _ge.PlotType.Line:
            ax.plot(self.data.x[initialIndex:finalIndex:purgeStep], self.data.y[initialIndex:finalIndex:purgeStep], \
            '-', **dataPlotArgs)
        elif plotType == _ge.PlotType.Point:
            ax.plot(self.data.x[initialIndex:finalIndex:purgeStep], self.data.y[initialIndex:finalIndex:purgeStep], \
            's', **dataPlotArgs)

        x = _np.linspace(_np.min(self.data.x[initialIndex:finalIndex]), \
                        _np.max(self.data.x[initialIndex:finalIndex - 1]), 1000) #correct finalXIndex for initial correction above.

        #The following line raises warnings in pylint. The code is OK tough.
        ax.plot(x, self.fittedFn(x), **functionPlotArgs)

        ax.set_xlabel(self.data.prettyXLabel())
        ax.set_ylabel(self.data.prettyYLabel())
        ax.set_title(self.prettyName)
        return fig, ax


class XYFitGenerator(_fit.FitGenerator):
    _fitClass = XYFit