#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 15:40:34 2019

@author: arjun
"""

import sys, os, torch, dataset, time, warnings, model
from pathlib import Path
import model
import torch.utils.data.sampler as sampler
import math

#must run main.py up to the point where we init Trainer

optimizer = torch.optim.Adam(Trainer.model.parameters(), lr = 0.0001)


def find_lr(iv=1e-8,fv=15.,beta=0.98, num=1000):
    batchs=4
    loader = torch.utils.data.DataLoader(traindata,batch_size = 4,shuffle= False,sampler = sampler.SubsetRandomSampler(range(num)))
    mult = (fv/iv)**(1/(num/batchs))
    lr = iv
    av_l = 0.
    best_l = 0.
    batch_num = 0
    losses = []
    log_lrs = []
    optimizer.param_groups[0]['lr'] = lr
    for i, sample in enumerate(loader):
        batch_num+=1
        input_data = sample['input']
        input_data = input_data.to(o.opt.device)

        label_data = sample['label']
        label_data = label_data.to(o.opt.device)
        caps_out, reconstruct = Trainer.model(input_data)
        
        lumen_masked = (input_data[:,0,:,:].unsqueeze(1)) * label_data
        optimizer.zero_grad()
        loss1 = Trainer.loss_fn1(caps_out, label_data) #this is for my custom dice loss
        loss2 = Trainer.loss_fn2(caps_out, label_data.float())
        loss3 = Trainer.loss_fn3(reconstruct, lumen_masked)
        loss = o.opt.la * loss1 + o.opt.lb * loss2 + o.opt.lc * loss3
        av_l = beta * av_l + (1-beta)*loss.data.item()
        smoothloss = av_l / (1-beta**batch_num)
        if batch_num > 1 and smoothloss > 4*best_l:
            return losses, log_lrs
        
        if smoothloss<best_l or batch_num==1:
            best_l=smoothloss
        losses.append(smoothloss)
        log_lrs.append(math.log10(lr))
        loss.backward()
        optimizer.step()
        lr*=mult
        optimizer.param_groups[0]['lr']=lr
        
    return losses, log_lrs