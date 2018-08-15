import warnings as _warnings
import typing as _typing
import sympy as _sp

import numpy as _np
import matplotlib.pyplot as _plt

import dataset as _ds
import global_funcs as _gf
import global_enums as _ge


class ErrorProp:
    
    def __init__(self, func, data = [] , eps = 0.1):

        if isinstance(data, _ds.Dataset):
            self.__data = data
        elif isinstance(data, list):
            self.__data = {}
            for ds in data:
                if isinstance(ds, _ds.Dataset) == False:
                   raise Warning
                self.__data[ds.name] = ds     #appending to dict datasets and name (explicty required for propagation)

        if isinstance(func, str): # getting the expression in the right form
            self.__func = _sp.sympify(func)
        elif isinstance(func, _sp.expr.Expr):
            self.__func = func
        
         
        self.__eps = eps
        self.__tree = []
        self.__parse(self.__func)
        self.__variables = list(dict.fromkeys(self.__tree)) # removing repetead symbols


        self.__errorSymbs = [_sp.Symbol('d{}'.format(var)) for var in self.__variables]
        self.__propExpr = self.__buildProp(self.__variables)
        
    
    def __parse(self, expr):  # parser for getting all the variables of the expression, it searches like a tree struct        
        for exp in expr.args:
            if isinstance(exp, _sp.symbol.Symbol):
                self.__tree.append(exp)
            self.__parse(exp)
    
    def __buildProp(self, variables):
        diffs = [_sp.diff(self.__func, symb) for symb in self.__variables]
        tmp = diffs[0]**2
        for diff, deltas in zip(diffs, self.__errorSymbs):
            tmp = _sp.Add(tmp, _sp.MatMul(diff**2, deltas**2))
        return (tmp - diffs[0]**2)**(1/2)     # propagation assuming no correlations between errors
        
            
    @property
    def data(self):
        return self.__data

    @property
    def variables(self):
        return self.__variables

    @property
    def func(self):
        return self.__func

    @property
    def propExpr(self):
        return self.__propExpr
    @property
    def deltas(self):
        return self.__errorSymbs
    
    """
    def run(self):
        # check if the names of dataset equals the variables from the function expression
        # use sets so we can ignore the order of the list builded from the dict keys
        
        Prop = []
        
        if not set(self.__variables) == set(list(self.__data.keys())):
            for var in list(self.__data.keys()):
                for idx in range(self.__data[var].v.size):
                    Prop.append(_sp.substitution([(var, self.__data[var].v[idx])]))
                    
        
        return Prop


    """