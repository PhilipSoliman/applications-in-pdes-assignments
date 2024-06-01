"""
Newton method
"""
import numpy as np


def Newton(function, Jac, Jac_l, y, l, yprev, lprev, ds, params):   # Newton for extended system (incl parameter)
    diff = np.inf
    counter = 0
    while diff>1e-6 and counter<300:
        # Residual
        F = np.zeros((3))
        F[:2] = function(y, l, params)
        F[-1] = arclengthEq(y, yprev, l, lprev, ds)

        # Jacobian
        J = np.zeros((3,3))
        J[:2,:2] = Jac(y, l, params)
        J[:2, -1] = Jac_l(y, l, params)
        J[-1, :] = arclengthJac(y, yprev, l, lprev, ds)

        dz = - np.linalg.solve(J, F)
        dy = dz[:-1]
        dl = dz[-1]
        y = y + dy
        l = l + dl
        counter += 1
        diff = np.linalg.norm(dy)/np.linalg.norm(y)
    print('parameter value: ', np.round(l, 6), 'Newton iterations: ', counter, 'relative change: ', diff)
    return y, l

def Newton_small(function, Jac, y, l, params):          # Newton for original system (excl parameter)
    diff = np.inf
    counter = 0
    while diff>1e-6 and counter<300:
        # Residual & Jacobian
        F = function(y, l, params)
        J = Jac(y, l, params)

        Jnum = np.zeros((2,2))
        for j in range(2):
            ynew = y + 1e-8*np.eye(2)[:, j]
            Jnum[:, j] = (function(ynew, l, params) - F)/1e-8

        dy = - np.linalg.solve(J, F)
        y = y + dy
        counter += 1
        diff = np.linalg.norm(dy)/np.linalg.norm(y)
    print('parameter value: ', np.round(l, 6), ', Newton iterations: ', counter, ', relative change: ', diff)
    return y


def arclengthEq(y, yprev, l, lprev, ds):    ## Arclength equation
    return np.sum((y-yprev)**2)/2+(l-lprev)**2 - ds**2

def arclengthJac(y, yprev, l, lprev, ds):   ## Derivative of arclength equation
    J = np.zeros((3))

    J[:2] = y-yprev
    J[-1] = 2*(l-lprev)
    return J