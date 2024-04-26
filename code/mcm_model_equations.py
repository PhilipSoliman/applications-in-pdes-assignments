from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
from helper import pyutils
from mcm import MCM
from root_finding import RootFinding
from continuation import Continuation

############### Setup Non-linear System of Equations #####
mcm = MCM()
sys = mcm.constructSystem()
# mcm.printSystem()
# mcm.printDimensionlessSystem()
# mcm.findStationaryPoints()
stable_points = mcm.getStableStationaryPoints()
print("Stable stationary points:")
pprint(stable_points)

############### Continuation parameters ##################
tolerance = 1e-5  # tolerance
maxiter = 100  # maximum number of iterations (per continuation)
continuation = Continuation(tolerance, maxiter)
continuation.output = True
parameter_name = "p1"  # parameter to be continued (should correspond to an attribute of the EBM object)
parameter_range = (0.5, 1)  # range of the parametervalues of interest
stepsize = 0.1  # 0.1  # stepsize (needs to be small, why?)
tune_factor = 0.00001  # 0.001 # tune factor (needs to be small, why?)
maxContinuations = 1000  # maximum number of continuations

############### Continuation (first test) #################
for stable_point in stable_points:
    print(f"Continuation starting at {stable_point}")
    mcm.x = stable_point
    errors = continuation.arclength(mcm, parameter_name, stepsize, tune_factor)

