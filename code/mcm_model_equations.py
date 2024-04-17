import matplotlib.pyplot as plt
import numpy as np
from sympy import init_printing

init_printing()
from pprint import pprint

import sympy as sym

sym.init_printing()
from helper import pyutils
from mcm import MCM
from root_finding import RootFinding

mcm = MCM()
print(mcm.getDimensionlessSystem())
