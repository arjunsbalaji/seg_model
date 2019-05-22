#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 22:40:55 2019

@author: arjun
"""

import torch
import time
import numpy as np
import os 
import sys
import shutil
import warnings
import model as m

torch.manual_seed(7)
np.random.seed(7)



start_time = time.time()

class Test(object):
    def __init__(self, opt, model, testdata, experiment):
        self.opt = opt
        self.model = model
        self.testdata = testdata
        self.loss_fn1 = m.Dice_loss()
        self.loss_fn2 = torch.nn.BCELoss()
        self.loss_fn2 = torch.nn.MSELoss()
        self.experiment = experiment
        
        if self.opt.save:
            os.mkdir(os.path.join(self.opt.runsaves_dir, self.opt.name, 'testsamples'))
            self.testsamples = {}
        
    def test(self):
        starttime = time.time()
        
        self.testloader = torch.utils.data.DataLoader(self.testdata, batch_size = self.opt.batch_size, shuffle= False)#
        
        self.testnames = []
        
        self.col_losses1 = []
        self.col_losses2 = []
        self.col_losses3 = []
        self.col_lossestotal = []
        
        for i, sample in enumerate(self.testloader):
            input_data = sample['input']
            input_data = input_data.to(self.opt.device)
            
            label_data = sample['label']
            label_data = label_data.to(self.opt.device)
            
            capsout, recon = self.model(input_data)
            self.testsamples[sample['name'][0]] = np.array([capsout, recon])
            
            lumen_masked = input_data[:,0,:,:] * label_data
            
            loss1 = self.loss_fn1(capsout, label_data) #this is for my custom dice loss
            loss2 = self.loss_fn2(capsout, label_data)
            loss3 = self.loss_fn3(recon, lumen_masked)
            
            self.loss = self.opt.la * loss1 + self.opt.lb * loss2 + self.opt.lc * loss3

            self.col_losses1.append(loss1.data)
            self.col_losses2.append(loss2.data)
            self.col_losses3.append(loss3.data)
            self.col_lossestotal.append(self.loss.data)
            self.testnames.append(sample['name'][0])
            
            self.experiment.log_metric('val_dice', self.col_losses1[-1])
            self.experiment.log_metric('va_lbce', self.col_losses2[-1])
            self.experiment.log_metric('val_recon', self.col_losses3[-1])
            self.experiment.log_metric('val_total', self.col_lossestotal[-1])
            
            sys.stdout.write(sample['name'][0] + ' loss: ' + str(self.col_losses1[i]) + '\n')
            
        self.testtime = time.time()-starttime
            
        sys.stdout.write('Mean ' + str(np.mean(self.col_lossestotal)) + '\n' + \
                         'Std ' + str(np.std(self.col_lossestotal)) + '\n' + \
                         'Max ' +str(np.max(self.col_lossestotal)) + '\n' + \
                         'Min ' + str(np.min(self.col_lossestotal)) + '\n')
        
        if self.opt.save:
            np.save(os.path.join(self.opt.runsaves_dir, self.opt.name, 'analysis', 'testDICElosses.npy'), np.array(self.col_losses1))
            np.save(os.path.join(self.opt.runsaves_dir, self.opt.name, 'analysis', 'testBCElosses.npy'), np.array(self.col_losses2))
            np.save(os.path.join(self.opt.runsaves_dir, self.opt.name, 'analysis', 'testRECONlosses.npy'), np.array(self.col_losses3))
            np.save(os.path.join(self.opt.runsaves_dir, self.opt.name, 'analysis', 'testTOTALlosses.npy'), np.array(self.col_lossestotal))
            #np.savetxt(os.path.join(self.opt.runsaves_dir, self.opt.name, 'analysis', 'testnames.txt'), self.testnames)
            
            #now to save all the test predictions as their case name
            for name in self.testnames:
                np.save(os.path.join(self.opt.runsaves_dir, self.opt.name, 'testsamples', name), self.testsamples[name].detach().cpu().numpy())
        