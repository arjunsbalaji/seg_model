#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 14:02:53 2019

@author: arjun
"""
#run deployBAYES before you can run this

import matplotlib.pylab as plt
import numpy as np
from skimage import filters
import string


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 3),
                        subplot_kw={'xticks': [], 'yticks': []})
closed = False

dicenames = list(np.load('/media/arjun/VascLab EVO/projects/oct_ca_seg/runsaves/Final1-pawsey/analysis/diceordered.npy'))

#load opimized threshold from deployBAYES run If no bayes then ignore this(youll get an error here)
f = open("/media/arjun/VascLab EVO/projects/oct_ca_seg/runsaves/Final1-pawsey/analysis/optimizedthresholds.json")
othresh = json.load(f)
f.close()

f = open("/media/arjun/VascLab EVO/projects/oct_ca_seg/runsaves/Final1-pawsey/analysis/hardicepairs.json")
harddice = json.load(f)
f.close()


dicenames =sorted(harddice, key=harddice.get)


def handle_close(evt):
    global closed
    closed = True

def waitforbuttonpress():
    while plt.waitforbuttonpress(0.2) is None:
        if closed:
            return False
    return True

fig.canvas.mpl_connect('close_event', handle_close)

i=200
while True:
    name = dicenames[i]
    thresholdo = othresh[name]
    image = np.array(d.pred_arrays(name)[0])[0,0]

    capsout = np.array(d.pred_arrays(name, threshold=thresholdo)[1])[0,0]
    mask = np.ma.masked_where(capsout==0,capsout)
    
    mag = np.array(d.pred_arrays(name, threshold=0.20)[3])[0,0]
    mag = filters.sobel(mag)
    #mag[mag>0.2] = 1
    #mag[mag<0.2] = 0
    mag = 100*mag
    mal = np.ma.masked_where(mag==0, mag)

    im = ax.imshow(image, 'gray', interpolation='none')
    
    la = ax.imshow(mask, 'RdYlGn', interpolation='none', alpha=0.5)
    #la = ax.imshow(mask, 'gray', interpolation='none') use this for no thresh
    
    acla = ax.imshow(mal, 'inferno', interpolation='none')#RdYlGn
    
    ax.set_title(name + '    d: ' +str(np.round(harddice[name],4)) + '    ot: ' + str(np.round(othresh[name],4)),size=9)
    
    plt.draw()
    
    i+=1 #advance by 1 pic
    
    if not waitforbuttonpress():
        break
print('.')



