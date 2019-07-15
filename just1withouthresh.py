#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 19:24:16 2019

@author: arjun
"""

import numpy as np
mask = np.zeros((10,10))
mask[3:-3, 3:-3] = 1 # white square in black background
im = mask + np.random.randn(10,10) * 0.01 # random image
masked = np.ma.masked_where(mask == 0, mask)

#whichone = 900

import matplotlib.pyplot as plt
from skimage import filters
import string

name = '0003092.npy'
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 3),
                        subplot_kw={'xticks': [], 'yticks': []})

i=0
image = np.array(d.print_pred(name)[0])[0,0]

capsout = np.array(d.print_pred(name, threshold=False)[1])[0,0]
mask = np.ma.masked_where(capsout==0,capsout)

mag = np.array(d.print_pred(name, threshold=0.95)[3])[0,0]
mag = filters.sobel(mag)
mag[mag>0.2] = 1
mag[mag<0.2] = 0
mag = 100*mag
mal = np.ma.masked_where(mag==0, mag)


#im = ax.imshow(image,'gray', interpolation='none')
la = ax.imshow(mask, 'gray', interpolation='none')
acla = ax.imshow(mal, 'RdYlGn', interpolation='none')

ax.set_title('Unthresholded' + '    ' + str(np.round(d.dicepairs[ting],4)),size=9)
i+=1
plt.tight_layout()
plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=None, hspace=None)
plt.savefig('just1.jpg', bbox_inches='tight',dpi=1000)
plt.show()