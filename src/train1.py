#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 18:27:41 2019

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

class Train(object):
    def __init__(self, opt, model, traindata, valdata, trainsetsize, valsetsize, experiment, checkpoint):
        self.opt = opt
        self.model = model
        self.traindata = traindata
        self.valdata = valdata
        self.trainsetsize = trainsetsize
        self.valsetsize = valsetsize
        self.loss_fn1 = m.Dice_Loss()
        #self.loss_fn2 = torch.nn.BCELoss(size_average=True)
        #self.loss_fn3 = torch.nn.MSELoss(size_average=True)
        self.loss_fn2 = torch.nn.BCELoss(size_average=False)
        self.loss_fn3 = torch.nn.MSELoss(size_average=False)
        self.experiment = experiment
        self.checkpoint = checkpoint
        
    def train(self):
        
        starttime = time.time()
        
        self.trainloader = torch.utils.data.DataLoader(self.traindata,
                                                       batch_size = self.opt.batch_size,
                                                       shuffle= False)#,
                                                       #sampler = sampler.SubsetRandomSampler(self.trainsetsize))

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.opt.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5, verbose=True)
        
        if self.opt.loadcheckpoint is not None:
            self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(self.checkpoint['scheduler_state_dict'])
            
            
        self.col_losses1 = []
        self.col_losses2 = []
        self.col_losses3 = []
        self.col_lossestotal = []
        self.val_loss_data = []
        
        if self.opt.logging:
            self.logs = []
            
        self.worsttobest = {}
        
        if self.opt.loadcheckpoint is not None:
            epochs = range(self.checkpoint['epoch'], self.checkpoint['epoch']+self.opt.epochs)
        else:
            epochs = range(self.opt.epochs)
            
        for epoch in epochs:
            sys.stdout.write('Epoch ' + str(epoch) + '\n')
            
            self.model.train()
            for i, sample in enumerate(self.trainloader):

                loss1, loss2, loss3 = self.train_step(sample)
                
                self.col_losses1.append(1-loss1.data)
                self.col_losses2.append(loss2.data)
                self.col_losses3.append(loss3.data)
                self.col_lossestotal.append(self.loss.data)
                #self.experiment.log_metric('training-dice')

                if self.opt.logging:
                    self.logs.append(torch.cuda.memory_allocated()/torch.cuda.memory_cached())
                        
            self.traintime = time.time() - starttime
            sys.stdout.write('ave sample time: ' + str(self.traintime/ ((epoch + 1) * len(self.trainloader))) + '\n')
            
            
            
            
            
            self.model.eval()
            valdata = self.validate()
            self.val_loss_data.append([valdata])
            
            
            self.scheduler.step(self.loss.data)
            
            if self.opt.comet:
                self.experiment.log_metric('val_dice', valdata[0])
                self.experiment.log_metric('va_lbce', valdata[1])
                self.experiment.log_metric('val_recon', valdata[2])
                self.experiment.log_metric('val_total', valdata[3])
        

        if self.opt.save:
            np.save(os.path.join(self.opt.runsaves_dir, self.opt.name, 'analysis', 'trainDICElosses.npy'), np.array(self.col_losses1))
            np.save(os.path.join(self.opt.runsaves_dir, self.opt.name, 'analysis', 'trainBCElosses.npy'), np.array(self.col_losses2))
            np.save(os.path.join(self.opt.runsaves_dir, self.opt.name, 'analysis', 'trainReconlosses.npy'), np.array(self.col_losses3))
            np.save(os.path.join(self.opt.runsaves_dir, self.opt.name, 'analysis', 'trainTOTALlosses.npy'), np.array(self.col_lossestotal))
            np.save(os.path.join(self.opt.runsaves_dir, self.opt.name, 'analysis', 'vallossdata.npy'), np.array(self.val_loss_data))
            
            if self.opt.logging:
                np.save(os.path.join(self.opt.runsaves_dir,
                                     self.opt.name,
                                     'analysis',
                                     'gpumemlogs.npy'),
                        np.array(self.logs))

    def train_step(self, sample):
        input_data = sample['input']
        input_data = input_data.to(self.opt.device)

        label_data = sample['label']
        label_data = label_data.to(self.opt.device)
                
        label_data = label_data.squeeze()
                
        #wut
        if len(label_data.size()) == 2:
            #print(label_data.size(), 'l2')
            label_data = torch.unsqueeze(label_data, 0)
            #print(label_data.size(), 'l3')
            label_data = torch.unsqueeze(label_data, 1)
        else:
            label_data = torch.unsqueeze(label_data, 1)
        
        
        #use these for troubleshooting
        #self.input_data = input_data
        #self.label_data = label_data
        caps_out, reconstruct = self.model(input_data)
        
        #use these for troubleshooting           
        #self.caps_out = caps_out
        #self.reconstruct = reconstruct
                
        lumen_masked = (input_data[:,0,:,:].unsqueeze(1)) * label_data
                
        self.optimizer.zero_grad()
                
        loss1 = self.loss_fn1(caps_out, label_data) #this is for my custom dice loss
        loss2 = self.loss_fn2(caps_out, label_data.float())
        loss3 = self.loss_fn3(reconstruct, lumen_masked)
                
        self.loss = self.opt.la * loss1 + self.opt.lb * loss2 + self.opt.lc * loss3
                
        self.loss.backward()
                
        self.optimizer.step()
        
        return loss1, loss2, loss3
        
                
    def validate(self):
        sys.stdout.write('Validating...' + '\n')
        starttime = time.time()

        self.valloader = torch.utils.data.DataLoader(self.valdata,
                                                     batch_size = self.opt.batch_size,
                                                     shuffle= False)#,
                                                     #sampler = sampler.SubsetRandomSampler(self.valsetsize))
        
        self.valcol_losses1 = []
        self.valcol_losses2 = []
        self.valcol_losses3 = []
        self.valcol_lossestotal = []
                
                
        for i, sample in enumerate(self.trainloader):
            input_data = sample['input']
            input_data = input_data.to(self.opt.device)

            label_data = sample['label']
            label_data = label_data.to(self.opt.device)
            
            label_data = label_data.squeeze()
            
            #wut
            if len(label_data.size()) == 2:
                #print(label_data.size(), 'l2')
                label_data = torch.unsqueeze(label_data, 0)
                #print(label_data.size(), 'l3')
                label_data = torch.unsqueeze(label_data, 1)
            else:
                label_data = torch.unsqueeze(label_data, 1)
                
            caps_out, reconstruct = self.model(input_data)
                
    
            lumen_masked = (input_data[:,0,:,:].unsqueeze(1)) * label_data
            
            self.optimizer.zero_grad()
            
            loss1 = self.loss_fn1(caps_out, label_data) #this is for my custom dice loss
            loss2 = self.loss_fn2(caps_out, label_data.float())
            loss3 = self.loss_fn3(reconstruct, lumen_masked)
            
            self.valloss = self.opt.la * loss1 + self.opt.lb * loss2 + self.opt.lc * loss3
            
            self.valcol_losses1.append(1-loss1.data)
            self.valcol_losses2.append(loss2.data)
            self.valcol_losses3.append(loss3.data)
            self.valcol_lossestotal.append(self.valloss.data)
            
        sys.stdout.write('Average Validation loss for epoch:' + str(np.mean(self.valcol_losses1)) \
                         + ', validation took '+ str(time.time()-starttime) + 'secs ' + '\n' + '\n')
        
        return [np.mean(self.valcol_losses1), np.mean(self.valcol_losses2), np.mean(self.valcol_losses3), np.mean(self.valcol_lossestotal)]