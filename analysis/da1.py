#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 15:53:52 2019

@author: arjun
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import visdom 

'''
#save_folder = '/home/arjunbalaji/Documents/Projects/oct_ca_seg/run_saves/'
save_folder = '/media/arjun/VascLab EVO/projects/oct_ca_seg/runsaves/'


run_name = 'up20e_0-pawsey-Tue-Jun-11-04:43:55-2019'

testdice = np.load(save_folder + run_name + '/analysis/testDICElosses.npy')
testbce = np.load(save_folder + run_name + '/analysis/testBCElosses.npy')
testrecon = np.load(save_folder + run_name + '/analysis/testRECONlosses.npy')
testtotal = np.load(save_folder + run_name + '/analysis/testTOTALlosses.npy')

traindice = np.load(save_folder + run_name + '/analysis/trainPAIRlosses.npy')
trainbce = np.load(save_folder + run_name + '/analysis/trainMSElosses.npy')
trainrecon = np.load(save_folder + run_name + '/analysis/trainReconlosses.npy')
traintotal = np.load(save_folder + run_name + '/analysis/trainTOTALlosses.npy')

valloss = np.load(save_folder + run_name + '/analysis/vallossdata.npy')

gpulog = np.load(save_folder + run_name + '/analysis/gpumemlogs.npy')

testsample_names = os.listdir(save_folder + run_name +'/testsamples')
'''

class DataAnalysis(object):
    def __init__(self, run_name):
        self.run_name = run_name
        
        save_folder = '/media/arjun/VascLab EVO/projects/oct_ca_seg/runsaves/'

        self.testdice = np.load(save_folder + run_name + '/analysis/testDICElosses.npy')
        self.testbce = np.load(save_folder + run_name + '/analysis/testBCElosses.npy')
        self.testrecon = np.load(save_folder + run_name + '/analysis/testRECONlosses.npy')
        self.testtotal = np.load(save_folder + run_name + '/analysis/testTOTALlosses.npy')
        
        #pair to dice, mse to bce
        self.traindice = np.load(save_folder + run_name + '/analysis/trainPAIRlosses.npy')
        self.trainbce = np.load(save_folder + run_name + '/analysis/trainMSElosses.npy')
        self.trainrecon = np.load(save_folder + run_name + '/analysis/trainReconlosses.npy')
        self.traintotal = np.load(save_folder + run_name + '/analysis/trainTOTALlosses.npy')
        
        self.valloss = np.load(save_folder + run_name + '/analysis/vallossdata.npy')
        
        self.gpulog = np.load(save_folder + run_name + '/analysis/gpumemlogs.npy')
        
        self.testsample_names = os.listdir(save_folder + run_name +'/testsamples')
        
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
    
da = DataAnalysis('up20e_0-pawsey-Tue-Jun-11-04:43:55-2019')

''' how to load all acc and dice csvs
accdata = {}
dicedata = {}
for name in listsaves[2:]:
    accdir = os.path.join(dird, name, 'analysis/acc.csv')
    dicedir = os.path.join(dird, name, 'analysis/dice.csv')
    acc1 = np.loadtxt(accdir, delimiter=',')
    dice1 = np.loadtxt(dicedir, delimiter=',')
    accdata[name]=acc1
    dicedata[name]=dice1
'''

