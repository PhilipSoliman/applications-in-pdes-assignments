import matplotlib.pyplot as plt
import numpy as np
from sympy import init_printing


from pprint import pprint

import sympy as sym

from helper import pyutils
from mcm import MCM
from root_finding import RootFinding

mcm = MCM()
mcm.printSystem()
sys = mcm.constructSystem()
mcm.printSystem()
