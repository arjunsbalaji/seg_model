#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 22:40:55 2019

@author: arjun
"""

import torch, os, sys, time
import torch.utils.data.sampler as sampler
import numpy as np
import model as m
import jutils as j
from pathlib import Path


start_time = time.time()
def print_results(data_dict, name):
    
    listofvalues = np.array(list(data_dict.values()))
    sys.stdout.write(name + ' data' + '\n' + \
                     'Mean ' + str(np.mean(listofvalues)) + '\n' + \
                     'Std ' + str(np.std(listofvalues)) + '\n' + \
                     'Max ' +str(np.max(listofvalues)) + '\n' + \
                     'Min ' + str(np.min(listofvalues)) + '\n' +'\n')
    return True




class Test(object):
    def __init__(self, opt, model, testdata, setsize, thresholds):
        self.opt = opt
        self.model = model
        self.testdata = testdata
        self.setsize = setsize
        self.loss_fn1 = m.Dice_Loss()
        self.loss_fn2 = torch.nn.BCELoss(size_average=True)
        self.loss_fn3 = torch.nn.MSELoss(size_average=True)
        
        self.thresholds = thresholds
        
            
    def test(self):
        starttime = time.time()
        
        self.testloader = torch.utils.data.DataLoader(self.testdata,
                                                      batch_size = 1,
                                                      shuffle= False)#,
                                                      #sampler = sampler.SubsetRandomSampler(self.setsize))
        
        
        
        self.testnames = []
        
        self.col_losses1 = []
        self.col_losses2 = []
        self.col_losses3 = []
        self.col_lossestotal = []
        
        self.sensdata = {}
        self.specdata = {}
        self.paccdata = {}
        self.dicedata = {}
        
        self.model.eval()
        
        for i, sample in enumerate(self.testloader):
            input_data = sample['input']
            input_data = input_data.to(self.opt.device)
            
            label_data = sample['label']
            label_data = label_data.to(self.opt.device)
            
            capsout, recon = self.model(input_data)
            
            '''
            #print(self.testsamples,sample['case_name'])
            if self.opt.save:softdice
                self.testsamples[sample['case_name'][0]] = torch.tensor(
                        torch.cat((capsout.detach(),
                                   recon.detach()), dim=0))
            '''


            lumen_masked = input_data[:,0,:,:].unsqueeze(1) * label_data
            
            
            loss1 = self.loss_fn1(capsout, label_data) #this is for my custom dice loss
            loss2 = self.loss_fn2(capsout, label_data)
            loss3 = self.loss_fn3(recon, lumen_masked)
            
            self.loss = self.opt.la * loss1 + self.opt.lb * loss2 + self.opt.lc * loss3

            self.col_losses1.append(loss1.item())
            self.col_losses2.append(loss2.item())
            self.col_losses3.append(loss3.item())
            self.col_lossestotal.append(self.loss.data)
            self.testnames.append(sample['case_name'][0])
            
            #for threshold in self.thresholds:
            
            #needs to be on cuda for broadcasting w capsout
            self.thresholds = torch.tensor(self.thresholds).to(self.opt.device)
            
            threshed = j.scalar_thresh(capsout, self.thresholds).to(self.opt.device)
            label_data = label_data[0]
            sens = j.sens(threshed, label_data).cpu()
            spec = j.spec(threshed, label_data).cpu()
            p_acc = j.acc(threshed, label_data).cpu()
            
            
            self.sensdata[sample['case_name'][0]] = list(np.array(sens).astype(float))
            self.specdata[sample['case_name'][0]] = list(np.array(spec).astype(float))
            self.paccdata[sample['case_name'][0]] = list(np.array(p_acc).astype(float))
            self.dicedata[sample['case_name'][0]] = 1-loss1.item()
            
            
            #sys.stdout.write(sample['case_name'][0] + ' loss: ' + str(1 - self.col_losses1[i]) + '\n')
            
        self.testtime = time.time()-starttime
            
        print_results(self.sensdata, 'Sensitivity')
        print_results(self.specdata, 'Specificity')
        print_results(self.paccdata, 'Pixel Accuracy')
        print_results(self.dicedata, 'Dice Score')
        
        if self.opt.save:
            savey = Path(self.opt.runsaves_dir+'/'+self.opt.name+'/analysis')
            np.save(savey/'testDICElosses.npy', np.array(self.col_losses1))
            np.save(savey/'testBCElosses.npy', np.array(self.col_losses2))
            np.save(savey/'testRECONlosses.npy', np.array(self.col_losses3))
            np.save(savey/'testTOTALlosses.npy', np.array(self.col_lossestotal))
            
            j.jsonsaveddict(self.sensdata, savey, 'sensdata.json')
            j.jsonsaveddict(self.specdata, savey, 'specdata.json')
            j.jsonsaveddict(self.paccdata, savey, 'paccdata.json')
            j.jsonsaveddict(self.dicedata, savey, 'dicedata.json')
            
            '''
            #now to save all the test predictions as their case name
            for name in self.testnames:
                np.save(os.path.join(self.opt.runsaves_dir, self.opt.name, 'analysis', 'testNAMES.npy'), self.testnames)
                np.save(os.path.join(self.opt.runsaves_dir, self.opt.name, 'testsamples', name), self.testsamples[name].cpu().numpy())
            '''
            
