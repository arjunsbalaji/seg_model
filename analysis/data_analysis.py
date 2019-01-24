#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 15:53:52 2019

@author: arjun
"""
import numpy as np
import matplotlib.pyplot as plt

save_folder = '/media/arjun/VascLab EVO/projects/oct_ca_seg/run_saves/'
save_folder = '/home/arjunbalaji/Documents/Projects/oct_ca_seg/run_saves/'

run_name = 'capsnetupsample-pawsey-Tue-Jan-15-06:45:55-2019'
run_name = 'capsnetupsample-pawsey-Fri-Jan-18-10:43:32-2019'
#run_name = 'capsnetupsample-pawsey-Tue-Jan-22-09:16:40-2019'

dice = np.load(save_folder + run_name + '/DICE.npy')
bce = np.load(save_folder + run_name + '/BCE.npy')
recon = np.load(save_folder + run_name + '/MSERecon.npy')
pics = np.load(save_folder + run_name + '/pics.npy')


def moving_ave(loss_list, move_by):
    new_loss_list = loss_list
    for i in range(move_by):
        new_loss_list = np.insert(new_loss_list, 0, 0, axis=0)
        new_loss_list = new_loss_list[:-1]
        loss_list = loss_list + new_loss_list
        
    loss_list = loss_list / move_by 
    return loss_list

def print_sample(sample_pics, lumen_threshold = None):
    f, (ax1i, ax1l, ax1lt) = plt.subplots(1,3, sharey=True)
    f.subplots_adjust(hspace=0.3)
    plt.tight_layout()
    
    #raw image
    image = ax1i.imshow(sample_pics[0,:,:],
                        aspect = 'equal')
    f.colorbar(image, ax=ax1i, orientation='vertical', fraction = 0.05)
    
    #pred lumen
    if lumen_threshold:
        lumen_pred = ax1l.imshow(sample_pics[1,:,:]>lumen_threshold,
                                 aspect = 'equal')
    else:
        lumen_pred = ax1l.imshow(sample_pics[1,:,:],
                                 aspect = 'equal')
        f.colorbar(lumen_pred, ax=ax1l, orientation='vertical', fraction = 0.05)
    
    #label lumen
    ax1lt.imshow(sample_pics[2,:,:],
                 aspect = 'equal')
    
    plt.show()
    
    
    