#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 12:56:28 2019

@author: arjun
"""

#run deploy before this

import numpy as np

#whichone = 900

import matplotlib.pyplot as plt
'''
image = np.array(d.print_pred(diceorderednames[whichone])[0])[0,0]
capsout = np.array(d.print_pred(diceorderednames[whichone], threshold=0.95)[1])[0,0]
mask = np.ma.masked_where(capsout==0,capsout)
fig, ax = plt.subplots()
im = ax.imshow(image,'gray', interpolation='none')
la = ax.imshow(mask, 'RdYlGn', interpolation='none', alpha=0.7)
ax.axis('off')
'''


whichimage = '0011631.npy'#'0009192.npy' 
whichones = [0,1,2,3,4,0,1,2,3,4,0,1,2,3,4]



x_prim = np.array(d.model.x_prim.detach())
x_1 = np.array(d.model.x_1.detach())
x_2 = np.array(d.model.x_2.detach())

fig, axs = plt.subplots(nrows=5, ncols=3, figsize=(3.5, 5.5),
                        subplot_kw={'xticks': [], 'yticks': []},
                        constrained_layout=True)

j=0
titles = ['Prim Maps','1st Maps','2nd Maps']
i=0
for ax, ting in zip(axs.T.flat, whichones):
    '''
    if ting == 'a':
        image = np.array(d.print_pred(whichimage)[0])[0,0]
    elif ting == 'b':
        image = np.array(d.print_pred(whichimage)[1])[0,0]
    '''
    if i<5:
        image = 1-np.array(x_prim[0,3,:,:,ting])
    elif i>=5 and i<10:
        image = np.array(x_1[0,4,:,:,ting])
    else:
        image = np.array(x_2[0,10,:,:,ting])
    
    if ting==0:
        ax.set_title(titles[j])
        j+=1
    
    im = ax.imshow(image,'gray', interpolation='none', aspect='auto')
    ax.set_adjustable('box-forced')
    i+=1
plt.axis('off')
#plt.subplots_adjust(wspace=0.001,hspace=0.001)
plt.tight_layout()
plt.savefig('afeaturemapsplot.jpg', bbox_inches='tight',dpi=1000)
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

