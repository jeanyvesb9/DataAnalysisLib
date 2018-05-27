import enum as _enum

@_enum.unique
class PlotType(_enum.Enum):
    ErrorBar = 0
    Line = 1
    Point = 2

@_enum.unique
class FitMethods(_enum.Enum):
    ODR = 0
    OLS = 1