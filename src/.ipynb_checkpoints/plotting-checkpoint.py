#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 01:10:36 2019

@author: arjun
"""
#Plotting

import time
import numpy as np
import os 
import sys
import shutil
import warnings
import matplotlib.pyplot as plt
from options import OptionsC


"""
to get hell nice box plots. 


diceeee is [array of dices0,array of dices1,array of dices2,array of dices3]
"""
fig, ax = plt.subplots()
ax.boxplot(diceeeee, 0,'', showmeans=True, whis=[5,95], meanprops=dict(marker='D', markeredgecolor='firebrick', markerfacecolor='firebrick'))
ax.set_xticklabels(['IM', '2DG', 'ADM', 'ALL'], fontsize=10)
ax.set_ylim(0.9,1)
plt.show()