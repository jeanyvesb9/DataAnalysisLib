import enum

@enum.unique
class PlotType(enum.Enum):
    ErrorBar = 0
    Line = 1
    Point = 2

@enum.unique
class FitMethods(enum.Enum):
    ODR = 0