# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 19:13:33 2022

@author: paual
"""

import numpy as np
from molli_bloch.bloch_matrices import R_alfa

def inversion(M_pre, B1, alfa = 180):
    '''
    Returns the net magnetization vector M_post that has been inverted alfa * B1 
    degrees with respect to M_pre. Used to perform the 180º inversion during
    the acquisition of a MOLLI sequence before the first and fifth readout.

    Parameters
    ----------
    M_pre : numpy.array
        The resting net magnetization vector.
    B1 : float
        The uncertainty parameter of the angle by which we invert the net 
        magnetization vector.
    alfa : float, optional
        The angle by which the net magnetization vector will be inverted. The
        default is 180.

    Returns
    -------
    M_post : numpy.array
        The inverted net magnetization vector.

    '''
    R = R_alfa(B1, alfa)
    
    M_post = np.matmul(R, M_pre)
    
    return M_post