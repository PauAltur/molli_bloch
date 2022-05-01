# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 17:55:28 2022

@author: paual
"""

import numpy as np


def R_alfa(B1, alfa):
    
    ''' 
    Returns a rotation matrix with a rotation of alfa degrees with respect
    to the x-axis.

    Parameters
    ----------
    alfa : float
        The amount of degrees to rotate about the x-axis.
    B1 : float
        The uncertainty parameter of the angle by which we invert the net magnetization vector.

    Returns
    -------
    R : numpy.array
        The rotation matrix with a rotation of alfa degrees about the x-axis.

    '''
    
    alfa_rad = np.deg2rad(alfa)
    
    alfa_B1_rad = alfa_rad * B1
    
    R = np.array([[1, 0, 0], 
                  [0, np.cos(alfa_B1_rad), np.sin(alfa_B1_rad)], 
                  [0, -np.sin(alfa_B1_rad), np.cos(alfa_B1_rad)]])
    return R