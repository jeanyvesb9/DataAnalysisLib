import warnings as _warnings
import functools as _functools

import numpy as _np
import scipy.odr as _odr
import matplotlib.pyplot as _plt
import pandas as _pd

from . import global_funcs as _gf
from . import global_enums as _ge
from . import fit as _fit


class XYZFit(_fit.Fit):
    def quickPlot(self, plotType=_ge.PlotType.ErrorBar, purgeStep=1, initialIndex=None, finalIndex=None, \
                    zMin: float=None, zMax: float=None, dataPlotArgs={}, functionPlotArgs={}):
        
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


        fig = _plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(self.data.x[initialIndex:finalIndex:purgeStep], \
                    self.data.y[initialIndex:finalIndex:purgeStep], \
                    self.data.z[initialIndex:finalIndex:purgeStep], \
                    **dataPlotArgs)


        x = _np.linspace(_np.min(self.data.x[initialIndex:finalIndex]), \
                        _np.max(self.data.x[initialIndex:finalIndex - 1]), 200)
        y = _np.linspace(_np.min(self.data.y[initialIndex:finalIndex]), \
                        _np.max(self.data.y[initialIndex:finalIndex - 1]), 200)

        X, Y = np.meshgrid(x, y)

        args = {}
        args[self.independentVars.keys()[0]] = X
        args[self.independentVars.keys()[1]] = Y

        ax.plot_wireframe(X, Y, self.fittedFn(**args), color = 'black', **functionPlotArgs)

        ax.set_xlabel(self.data.prettyXLabel)
        ax.set_ylabel(self.data.prettyYLabel)
        ax.set_zlabel(self.data.prettyZLabel)
        ax.set_zlim(zMin, zMax)

        ax.set_title(self.prettyName)

        return fig, ax


class XYZFitGenerator(_fit.FitGenerator):
    _fitClass = XYZFit