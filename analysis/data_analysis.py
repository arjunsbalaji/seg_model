#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 15:53:52 2019

@author: arjun
"""
import numpy as np
import matplotlib.pyplot as plt


save_folder = '/home/arjunbalaji/Documents/Projects/oct_ca_seg/run_saves/'
#save_folder = '/media/arjun/VascLab EVO/projects/oct_ca_seg/run_saves/'


run_name = 'pawsey--lr-0.0001--trans-True-Wed-Jan-30-01:27:21-2019'



testdice = np.load(save_folder + run_name + '/analysis/test_DICE.npy')
testbce = np.load(save_folder + run_name + '/analysis/test_BCE.npy')
testrecon = np.load(save_folder + run_name + '/analysis/test_MSERecon.npy')
testpics = np.load(save_folder + run_name + '/analysis/test_pics.npy')

traindice = np.load(save_folder + run_name + '/analysis/DICE.npy')
trainbce = np.load(save_folder + run_name + '/analysis/BCE.npy')
trainrecon = np.load(save_folder + run_name + '/analysis/MSERecon.npy')
trainpics = np.load(save_folder + run_name + '/analysis/pics.npy')

class DataAnalysis(object):
    def __init__(self, a):
        self.a = a
        
    def moving_ave(self, loss_list, move_by):
        new_loss_list = loss_list
        for i in range(move_by):
            new_loss_list = np.insert(new_loss_list, 0, 0, axis=0)
            new_loss_list = new_loss_list[:-1]
            loss_list = loss_list + new_loss_list
            
        loss_list = loss_list / move_by 
        return loss_list
    
    def print_sample(self, sample_pics, loss, lumen_threshold = None):
        
        
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
        f.suptitle('BCE = '+str(loss), fontsize=16)
        plt.show()
    
    def present_sample(self, pics, losses, idx):
        sample = pics[idx]
        loss = losses[idx]
        
        self.print_sample(sample, loss, lumen_threshold=None)
    
da = DataAnalysis(1)

