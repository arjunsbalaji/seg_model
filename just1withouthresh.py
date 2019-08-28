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
closed = False


f = open("/media/arjun/VascLab EVO/projects/oct_ca_seg/runsaves/Final1-pawsey/analysis/optimizedthresholds.json")
othresh = json.load(f)
f.close()


f = open("/media/arjun/VascLab EVO/projects/oct_ca_seg/runsaves/Final1-pawsey/analysis/hardicepairs.json")
harddice = json.load(f)
f.close()

dicenames =sorted(harddice, key=harddice.get)

#'0003092.npy' #100 is good

thresholdo=othresh[name]

def handle_close(evt):
    global closed
    closed = True

def waitforbuttonpress():
    while plt.waitforbuttonpress(0.2) is None:
        if closed:
            return False
    return True

fig.canvas.mpl_connect('close_event', handle_close)

fig, (axim, ax) = plt.subplots(nrows=1, ncols=2, figsize=(3, 2),
                        subplot_kw={'xticks': [], 'yticks': []})

i=150
while True:
    name = dicenames[i]

    
    
    image = np.array(d.pred_arrays(name)[0])[0,0]
    
    capsout = np.array(d.pred_arrays(name, threshold=False)[1])[0,0]
    mask = np.ma.masked_where(capsout==0,capsout)
    
    mag = np.array(d.pred_arrays(name, threshold=0.95)[3])[0,0]
    mag = filters.sobel(mag)
    mag[mag>0.2] = 1
    mag[mag<0.2] = 0
    mag = 100*mag
    mal = np.ma.masked_where(mag==0, mag)
    
    
    im = axim.imshow(image,'gray', interpolation='none')
    la = ax.imshow(mask, 'gray', interpolation='none')
    acla = ax.imshow(mal, 'RdYlGn', interpolation='none')
    
    ax.set_title('Unthresholded' + '    ' + str(np.round(harddice[name],4)),size=9)
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=None, hspace=None)
    #plt.savefig('just1.jpg', bbox_inches='tight',dpi=1000)
    plt.draw()
    i+=1
    
    if not waitforbuttonpress():
        break
    
print('.')
