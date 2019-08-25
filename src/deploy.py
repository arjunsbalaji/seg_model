#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 18:49:40 2019

@author: arjun
"""
#DEPLOY is to test the model on untransfromed members of the test set
#IT ONLY WORKS ON THE HOME COMPUTER!!!!!!

import torch
import time
import numpy as np
import os 
import sys
import shutil
import warnings
import model as m
from options import OptionsHome
import skimage.transform as skitransforms

warnings.simplefilter('ignore')


torch.manual_seed(7)
np.random.seed(7)

#options must be same for model as the loaded model.!
o = OptionsHome()
o.parse()


start_time = time.time()
def sens(c,l):
    intersection = torch.sum(c * l)
    union = torch.sum(c) + torch.sum(l) - intersection
    loss = (intersection) / (union)
    return loss

def spec(c,l):
    c=1-c
    l=1-l
    intersection = torch.sum(c * l)
    union = torch.sum(c) + torch.sum(l) - intersection
    loss = (intersection) / (union)
    return loss
    
class Deploy(object):
    def __init__(self, opt, model, testnames):
        self.opt = opt
        self.model = model
        self.testnames = testnames
        self.loss_fn1 = m.Dice_Loss()
        self.loss_fn2 = torch.nn.BCELoss(size_average=True)
        self.loss_fn3 = torch.nn.MSELoss(size_average=True)
    
    def print_pred(self, name, threshold=False):
        model.eval()
        data_dir = '/media/arjun/VascLab EVO/projects/oct_ca_seg/actual final data'
        image_dir = os.path.join(data_dir, 'images')
        label_dir = os.path.join(data_dir, 'labels')
        
        inputdata = np.load(os.path.join(image_dir, name))
        labeldata = np.load(os.path.join(label_dir, name))
        
        inputdata = inputdata.astype(float)
        labeldata = labeldata.astype(float)
        #print(inputdata.shape)
        #print(labeldata.shape)
        
        labeldata = np.transpose(labeldata, (1, 2, 0))
        inputdata = np.transpose(inputdata, (1, 2, 0))
        #print(inputdata.shape)
        #print(labeldata.shape)
        inputdata = skitransforms.resize(inputdata, output_shape=(256, 256))
        labeldata = skitransforms.resize(labeldata, output_shape=(256, 256))
        #print(inputdata.shape)
        #print(labeldata.shape)
        labeldata = np.transpose(labeldata.copy(), (2, 0, 1))
        inputdata = np.transpose(inputdata.copy(), (2, 0, 1))
        #print(inputdata.shape)
        #print(labeldata.shape)
        labeldata = torch.tensor(labeldata).to('cuda').unsqueeze(0).float()
        inputdata = torch.tensor(inputdata).to('cuda').unsqueeze(0).float()
        
        
        capsout, recon = self.model(inputdata)
        
        if threshold:
            capsout[capsout>threshold] = 1
            capsout[capsout<threshold] = 0
        
        return inputdata, capsout.detach(), recon.detach(), labeldata
        
    def deploy(self, case_names, threshold=False): #threshold only affects accuracy not dice
        starttime = time.time()
        
        model.eval()
        data_dir = '/media/arjun/VascLab EVO/projects/oct_ca_seg/actual final data'
        image_dir = os.path.join(data_dir, 'images')
        label_dir = os.path.join(data_dir, 'labels')
        
        self.col_losses1 = []
        self.col_losses2 = []
        self.col_losses3 = []
        self.col_lossestotal = []
        
        self.dicepairs = {}
        
        self.accpairs = {}
        self.senspairs = {}
        self.specpairs = {}
        
        for name in case_names:
            imtime=time.time()
            inputdata = np.load(os.path.join(image_dir, name))
            labeldata = np.load(os.path.join(label_dir, name))
            self.inn = inputdata
            inputdata = inputdata.astype(float)
            labeldata = labeldata.astype(float)
            #print(inputdata.shape)
            #print(labeldata.shape)
            self.inn = inputdata
            labeldata = np.transpose(labeldata, (1, 2, 0))
            inputdata = np.transpose(inputdata, (1, 2, 0))
            #print(inputdata.shape)
            #print(labeldata.shape)
            inputdata = skitransforms.resize(inputdata, output_shape=o.opt.c_size)
            labeldata = skitransforms.resize(labeldata, output_shape=o.opt.c_size)
            #print(inputdata.shape)
            #print(labeldata.shape)
            labeldata = np.transpose(labeldata.copy(), (2, 0, 1))
            inputdata = np.transpose(inputdata.copy(), (2, 0, 1))
            #print(inputdata.shape)
            #print(labeldata.shape)
            labeldata = torch.tensor(labeldata).to('cuda').unsqueeze(0).float()
            inputdata = torch.tensor(inputdata).to('cuda').unsqueeze(0).float()
            #self.inputdata=inputdata
            #self.labeldata=labeldata
            
            capsout, recon = self.model(inputdata)
            #print(time.time()-imtime)
            self.inputdata=inputdata
            self.labeldata=labeldata
            self.caps = capsout
            self.recon = recon
            #print(capsout.size(), recon.size())
            lumen_masked = inputdata[0,0].unsqueeze(0).unsqueeze(0) * labeldata
            
            loss1 = self.loss_fn1(capsout, labeldata) #this is for my custom dice loss
            loss2 = self.loss_fn2(capsout, labeldata)
            loss3 = self.loss_fn3(recon, lumen_masked)
            
            loss = self.opt.la * loss1 + self.opt.lb * loss2 + self.opt.lc * loss3
            
            self.col_losses1.append(loss1.data)
            self.col_losses2.append(loss2.data)
            self.col_losses3.append(loss3.data)
            self.col_lossestotal.append(loss.data)
            
            if threshold:
                capsout = capsout.detach()
                capsout[capsout>threshold] = 1
                capsout[capsout<threshold] = 0
                total = (capsout == labeldata).sum()
                self.senspairs[name] = float(sens(capsout,labeldata))
                self.specpairs[name] = float(spec(capsout,labeldata))
                self.accpairs[name] = int(total) / np.prod(labeldata.size()[2:4])
            #print(loss1.data)
            #self.dicepairs[name] = np.array(loss1.data)[0].astype(float)
            
            
            #print(name, 'DONE')
        self.endtime = time.time()-starttime
        self.col_losses1 = np.array(self.col_losses1)
        self.col_losses2 = np.array(self.col_losses2)
        self.col_losses3 = np.array(self.col_losses3)
        self.col_lossestotal = np.array(self.col_lossestotal)
        
        
        for name, loss in zip(case_names, 1-self.col_losses1):
            self.dicepairs[name] = loss
            
        return capsout, recon
        '''
        sys.stdout.write('Mean ' + str(1-np.mean(self.col_losses1)) + '\n' + \
                         'Std ' + str(np.std(self.col_losses1)) + '\n' + \
                         'Min ' +str(1-np.max(self.col_losses1)) + '\n' + \
                         'Max ' + str(1-np.min(self.col_losses1)) + '\n')
        '''




#path to whichever model you want. usually will live in a ehckpoint
checkpoint = torch.load('/media/arjun/VascLab EVO/projects/oct_ca_seg/runsaves/Final-pawsey/checkpoints/checkpoint.pt')

model = m.CapsNet(o.opt)
model.load_state_dict(checkpoint['model_state_dict'])
model.to('cuda')

#this should be the testsamples from your loaded model
testnames = os.listdir('/media/arjun/VascLab EVO/projects/oct_ca_seg/runsaves/Final-pawsey/testsamples')

d = Deploy(o.opt, model, testnames)

d.deploy(testnames, 0.95) #can omit threshold.

dices = 1-d.col_losses1
sys.stdout.write('Mean ' + str(np.mean(dices)) + '\n' + \
                 'Std ' + str(np.std(dices)) + '\n' + \
                 'Max ' +str(np.max(dices)) + '\n' + \
                 'Min ' + str(np.min(dices)) + '\n')

diceorderednames = sorted(d.dicepairs, key=d.dicepairs.get) #orders names worst to best
accorderednames = sorted(d.accpairs, key=d.accpairs.get)

u0d = d.dicepairs
u0a = d.accpairs

