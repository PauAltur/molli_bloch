# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 11:39:51 2022

@author: paual
"""

import numpy as np
import matplotlib.pyplot as plt

def bland_altman_plot(ax, diff, xaxis):
    '''
    Generates the Bland Altman plot of a set of estimations using Matpltolib.

    Parameters
    ----------
    ax : matplotlib.pyplot.axes
        An axes object where the plot will be drawn.
    diff : numpy.array
        Difference between the estimated and true values which are the result of
        a given estimation task.
    xaxis : numpy.array
        Array that defines the x axis of the Bland Altman plot. Can be the true
        values or some other parameter whose influence on the estimation wants 
        to be explored.

    Returns
    -------
    ax : matplotlib.pyplot.axes
        The axes object with the Bland-Altman plot drawn upon it.

    '''
    
    md = np.mean(diff)
    sd = np.std(diff)
    
    ax.scatter(xaxis, diff, s = 10)
    ax.axhline(md, label = 'Mean : {}'.format(md))
    ax.axhline(1.96 * sd, label = '+1.96 SD : {}'.format(1.96 * sd))
    ax.axhline(-1.96 * sd, label = '-1.96 SD : {}'.format(-1.96 * sd))
    
    return ax