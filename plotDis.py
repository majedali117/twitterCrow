# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 13:22:02 2018

@author: HP
"""
import numpy as np
from scipy.stats import expon
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1)
for x in range(0,100):
    
    
    val=expon.pdf(x, scale=1)
    val=skewnorm.pdf(x, a=-1,loc=50)
    print(x,val)
    plt.plot(x, val, linewidth=2.0)

plt.draw()