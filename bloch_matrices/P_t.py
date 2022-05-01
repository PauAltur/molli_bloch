# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 17:55:38 2022

@author: paual
"""

import numpy as np

def P_t(t, delta_f = 0):
    ''' 
    Returns a rotation matrix P that represents free precession during a 
    period t
    
    Parameters
    ----------
    t : float
        Length of time during which free precession takes place.
    delta_f : float, optional
        Off-resonance factor. The default is 0.

    Returns
    -------
    P : numpy.array
        Rotation matrix that represents free precession during a 
        period t .

    '''
    
    P = np.array([[np.cos(2 * np.pi * delta_f * t), np.sin(2 * np.pi * delta_f * t), 0], 
                  [-np.sin(2 * np.pi * delta_f * t), np.cos(2 * np.pi * delta_f * t), 0], 
                  [0, 0, 1]])
    
    return P