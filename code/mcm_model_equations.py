from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
from helper import pyutils
from mcm import MCM
from root_finding import RootFinding
# from sympy import init_printing

mcm = MCM()
sys = mcm.constructSystem()
mcm.printSystem()
mcm.printDimensionlessSystem()
mcm.findStationaryPoints()
