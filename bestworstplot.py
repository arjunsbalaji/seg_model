#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 12:56:28 2019

@author: arjun
"""

#run deploy before this

import numpy as np
mask = np.zeros((10,10))
mask[3:-3, 3:-3] = 1 # white square in black background
im = mask + np.random.randn(10,10) * 0.01 # random image
masked = np.ma.masked_where(mask == 0, mask)

#whichone = 900

import matplotlib.pyplot as plt
from skimage import filters
import string
'''
image = np.array(d.print_pred(diceorderednames[whichone])[0])[0,0]
capsout = np.array(d.print_pred(diceorderednames[whichone], threshold=0.95)[1])[0,0]
mask = np.ma.masked_where(capsout==0,capsout)
fig, ax = plt.subplots()
im = ax.imshow(image,'gray', interpolation='none')
la = ax.imshow(mask, 'RdYlGn', interpolation='none', alpha=0.7)
ax.axis('off')
'''
#whichones = diceorderednames[3:8] + diceorderednames[598:603] + diceorderednames[1197:1202]

f = open("/media/arjun/VascLab EVO/projects/oct_ca_seg/runsaves/Final1-pawsey/analysis/hardicepairs.json")
harddice = json.load(f)
f.close()

f = open("/media/arjun/VascLab EVO/projects/oct_ca_seg/runsaves/Final1-pawsey/analysis/optimizedthresholds.json")
othresh = json.load(f)
f.close()


dicenames =sorted(harddice, key=harddice.get)
#whichones = dicenames[3:8] + dicenames[598:603] + dicenames[2397:2402]
whichones = ['0003070.npy','0003073.npy','0010525.npy','0002846.npy','0007566.npy',]+dicenames[2397:2402]+['0011734.npy','0011035.npy','0003619.npy','0007331.npy','0002151.npy','0010883.npy']
#Issues = ['Artefact + blood', 'Artefact + blood', 'Beginning of Bifurcation']
fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(6.5, 4.77),
                        subplot_kw={'xticks': [], 'yticks': []})

i=0
for ax, ting in zip(axs.flat, whichones):
    
    image = np.array(d.pred_arrays(whichones[i])[0])[0,0]
    
    capsout = np.array(d.pred_arrays(whichones[i], threshold=othresh[whichones[i]])[1])[0,0]
    mask = np.ma.masked_where(capsout==0,capsout)
    
    mag = np.array(d.pred_arrays(whichones[i], threshold=0.95)[3])[0,0]
    mag = filters.sobel(mag)
    #mag[mag>0.2] = 1
    #mag[mag<0.2] = 0
    mag = 100*mag
    mal = np.ma.masked_where(mag==0, mag)
    
    
    im = ax.imshow(image,'gray', interpolation='none')
    la = ax.imshow(mask, 'RdYlGn', interpolation='none', alpha=0.7)
    acla = ax.imshow(mal, 'inferno', interpolation='none')
    
    #ax.set_title(string.ascii_uppercase[i] + '    ' + str(np.round(d.dicepairs[ting],4)),size=9)
    #ax.set_title(string.ascii_uppercase[i] + '    ' + str(np.round(harddice[ting],4)),size=9)
    ax.set_title(whichones[i] + '. | d: ' +str(np.round(100*harddice[whichones[i]],4)) + ' | ot: ' + str(np.round(othresh[whichones[i]],4)),size=9)
    i+=1


plt.tight_layout()
plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=None, hspace=None)
plt.savefig('a10worstbestmiddle.jpg', bbox_inches='tight',dpi=1000)
plt.show()


'''
plt.figure()
plt.subplot(1,2,1)
plt.axis('Off')
plt.imshow(image,'gray', interpolation='none')
plt.imshow(mask, 'RdYlGn', interpolation='none', alpha=0.7)
plt.show()
'''


#plt.imshow(np.array(d.print_pred(diceorderednames[0])[1])[0,0], cmap='gray')gist_rainbow