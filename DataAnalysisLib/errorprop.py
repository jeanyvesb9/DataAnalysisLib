import warnings as _warnings
import typing as _typing
import sympy as _sp
import copy as _copy

import numpy as _np
import matplotlib.pyplot as _plt

from . import dataset as _ds
from . import global_funcs as _gf
from . import global_enums as _ge
from . import multidataset as _mds


class ErrorProp:
    
    def __init__(self, func, data):
        
        
        # sanity checks for dataset input o multidataset  input

        if isinstance(data, _ds.Dataset):
            self.__data = data
        elif isinstance(data, list):
            self.__data = {}
            for ds in data:
                if not isinstance(ds, _ds.Dataset):
                   raise TypeError("Only lists of Datasets can be parsed")
                self.__data[ds.name] = ds 
                
        elif isinstance(data, _mds.MultiDataset):
            self.__data = {}
            for ds in data.datasets:
                self.__data[ds.name] = ds
            if data.autoGenCov == True:
                self.__covs = data.covMatrices
            else:
                for cov in data.covMatrices:
                    if _np.trace(cov) == 0:
                        raise ValueError("Some covariance matrix is singular")
                    else:
                        self.__covs = data.covMatrices
                        

        if isinstance(func, str):    # getting the expression in the right form
            self.__func = _sp.sympify(func)
        elif isinstance(func, _sp.expr.Expr):
            self.__func = func
        
        self.__variables = _sp.Basic.atoms(self.__func) # get atomic vars
        
        self.__numbers = []
        
        for var in self.__variables:       # made this way because sets can't change size on iteration
            if var.is_number:              #
                self.__numbers.append(var) #
                                           #
        for num in self.__numbers:         #
            self.__variables.remove(num)   #

        self.__errorSymbs = [_sp.Symbol('d{}'.format(var)) for var in self.__variables]
        
        self.__jacobian = [_sp.diff(self.__func, symb) for symb in self.__variables]
        
        # general expresion for error of scalar functions (using covariances)
        
        self.__error = (_np.dot( _np.dot(self.__jacobian, self.__covs) , _np.transpose(self.__jacobian) )[0])**0.5 
        
        # final checks, so we can point each atomic variable to one dataset
        
        self.__temp_names = set([_sp.sympify(name) for name in self.__data.keys()])
        if not self.__temp_names == self.__variables:
            raise ValueError("Names of datasets do not match the variables in the expression")
        del self.__temp_names
        
        
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
    def jacobian(self):
        return self.__jacobian
    @property
    def errorExpr(self):
        return self.__error
    
    @property
    def propExpr(self):
        return self.__propExpr
    @property
    def deltas(self):
        return self.__errorSymbs
    

    def run(self):
        
        pairs = []
        replace_values = []
        
        # first we get a list of all tuples (symbol, value) for every dataset and point
        
        for key in self.__data.keys():
            for i,val in enumerate(self.__data[key]):
                pairs.append( (key, self.__data[key][i]) )
                
        # here we get parsed the different lists ordered by index to be replaced
                
        L = len(self.__variables)
        for n in range(L+1):
            replace_values.append(pairs[n::L + 1])
            
        new_func = _copy.deepcopy(self.__func)         # just to stay safe we don't
        new_error_expr = _copy.deepcopy(self.__error)  # modify the original expresion 
        
        new_values = [new_func.subs(values) for values in replace_values]
        new_errors = [new_error_expr.subs(values) for values in replace_values]
        
        new_ds = _ds.Dataset(data = new_values, error = new_errors, errorFn = None)
        
        return new_ds
                

