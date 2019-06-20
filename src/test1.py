#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 22:40:55 2019

@author: arjun
"""

import torch
import torch.utils.data.sampler as sampler
import time
import numpy as np
import os 
import sys
import shutil
import warnings
import model as m


start_time = time.time()

class Test(object):
    def __init__(self, opt, model, testdata, setsize,  experiment):
        self.opt = opt
        self.model = model
        self.testdata = testdata
        self.setsize = setsize
        self.loss_fn1 = m.Dice_Loss()
        self.loss_fn2 = torch.nn.BCELoss(size_average=True)
        self.loss_fn3 = torch.nn.MSELoss(size_average=True)
        self.experiment = experiment
        
        if self.opt.save:
            os.mkdir(os.path.join(self.opt.runsaves_dir,
                                  self.opt.name,
                                  'testsamples'))
            self.testsamples = {}
            
        self.testloader = torch.utils.data.DataLoader(self.testdata,
                                                      batch_size = self.opt.batch_size,
                                                      shuffle= False)
        

    def test(self):
        starttime = time.time()
        
        self.testloader = torch.utils.data.DataLoader(self.testdata,
                                                      batch_size = self.opt.batch_size,
                                                      shuffle= False)
        
        
        
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
            
            #print(self.testsamples,sample['case_name'])
            if self.opt.save:
                self.testsamples[sample['case_name'][0]] = torch.tensor(
                        torch.cat((capsout.detach(),
                                   recon.detach()), dim=0))
            

            lumen_masked = input_data[:,0,:,:].unsqueeze(1) * label_data
            
            
            loss1 = self.loss_fn1(capsout, label_data) #this is for my custom dice loss
            loss2 = self.loss_fn2(capsout, label_data)
            loss3 = self.loss_fn3(recon, lumen_masked)
            
            self.loss = self.opt.la * loss1 + self.opt.lb * loss2 + self.opt.lc * loss3

            self.col_losses1.append(loss1.data)
            self.col_losses2.append(loss2.data)
            self.col_losses3.append(loss3.data)
            self.col_lossestotal.append(self.loss.data)
            self.testnames.append(sample['case_name'][0])
            
            if self.opt.comet:
                self.experiment.log_metric('test_dice', self.col_losses1[-1])
                self.experiment.log_metric('test_lbce', self.col_losses2[-1])
                self.experiment.log_metric('test_recon', self.col_losses3[-1])
                self.experiment.log_metric('test_total', self.col_lossestotal[-1])
            
            sys.stdout.write(sample['case_name'][0] + ' loss: ' + str(1 - self.col_losses1[i]) + '\n')
            
        self.testtime = time.time()-starttime
            
        sys.stdout.write('Mean ' + str(1-np.mean(self.col_losses1)) + '\n' + \
                         'Std ' + str(np.std(self.col_losses1)) + '\n' + \
                         'Max ' +str(1-np.max(self.col_losses1)) + '\n' + \
                         'Min ' + str(1-np.min(self.col_losses1)) + '\n')
        
        if self.opt.save:
            np.save(os.path.join(self.opt.runsaves_dir, self.opt.name, 'analysis', 'testDICElosses.npy'), np.array(self.col_losses1))
            np.save(os.path.join(self.opt.runsaves_dir, self.opt.name, 'analysis', 'testBCElosses.npy'), np.array(self.col_losses2))
            np.save(os.path.join(self.opt.runsaves_dir, self.opt.name, 'analysis', 'testRECONlosses.npy'), np.array(self.col_losses3))
            np.save(os.path.join(self.opt.runsaves_dir, self.opt.name, 'analysis', 'testTOTALlosses.npy'), np.array(self.col_lossestotal))
            #np.savetxt(os.path.join(self.opt.runsaves_dir, self.opt.name, 'analysis', 'testnames.txt'), self.testnames)
            
            #now to save all the test predictions as their case name
            for name in self.testnames:
                np.save(os.path.join(self.opt.runsaves_dir, self.opt.name, 'analysis', 'testNAMES.npy'), self.testnames)
                np.save(os.path.join(self.opt.runsaves_dir, self.opt.name, 'testsamples', name), self.testsamples[name].cpu().numpy())
        