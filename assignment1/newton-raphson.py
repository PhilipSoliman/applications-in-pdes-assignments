import numpy as np
import matplotlib.pyplot as plt

# constants arameters (capitalized is default value)
MU = 30 # greenhouse gas and particle parameter
a_1 = 0.278 # albedo of water
a_2 = 0.7 # albedo of ice
delta = 0 # heat flux at poles
Q_0 = 341.3 # solar radiation constant
sigma = 5.67e-8 # Stefan-Boltzmann constant
C_T = 5.0e8 # heat capacity of the Earth

#  solar radiation 
def R_A(x, T, mu = MU):
    return Q_solar(x) * (1-albedo(T)) + mu

def Q_solar(x):
    return Q_0 * (1 - 0.241 * (1 - x**2))

def albedo(T):
    return a_1 + (a_2 - a_1) * (1 + np.tanh(0.01 * (T - 273.15)))