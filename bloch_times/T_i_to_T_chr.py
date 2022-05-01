# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 18:01:39 2022

@author: paual
"""

import numpy as np

def T_i_to_T_chr(T_i):
    '''
    Returns a vector of the time between acquisitions in order to compute
    the recovery of the net magnetization vector between readouts along a
    whole MOLLI sequence

    Parameters
    ----------
    T_i : numpy.array
        numpy.array of the inversion times, in milliseconds, at which readouts 
        have been acquired.

    Returns
    -------
    T_rec : tuple
        numpy.array of the times, in milliseconds, between readouts and inversions
        needed to compute the recovery of the net magnetization vector.

    '''
    
    T_i_first = T_i[[0, 2, 4, 6, 7]] # T_i of readouts acquired in 1st inversion
    T_i_second = T_i[[1, 3, 5]] # T_i of readouts acquired in 2nd inversion
    
    T_chr_first = T_i_first - np.concatenate((np.array([0.0]), T_i_first[:-1]))
    T_chr_second = T_i_second - np.concatenate((np.array([0.0]), T_i_second[:-1]))
    
    T_chr = np.concatenate((T_chr_first, np.array([3850.0]), T_chr_second))
    
    return T_chr