#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 17:12:18 2019

@author: arjun
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 18:49:40 2019

@author: arjun
"""
#uncertaintymeasure
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
from bayes_opt import BayesianOptimization
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import json

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


class DeployOCTDataset(Dataset):
    """
    First we create a dataset that will encapsulate our data. It has 3 special 
    functions which will be explained as they go. We will pass this dataset object
    to the torch dataloader object later which will make training easier.
    """
    def __init__ (self,
                  main_data_dir,
                  names):
        self.main_data_dir = main_data_dir
        self.names = names    
        
        self.imagedir = os.path.join(self.main_data_dir, 'images')
        self.labeldir = os.path.join(self.main_data_dir, 'labels')
        
        
    def visualise(self, idx):
        
        sample = self.__getitem__(idx)
        #print(sample['input'].size())
        #print(sample['label'].size())
        input_data = sample['deployin'].cpu().numpy()[0,0,:,:]
        l_data = sample['deploylabel'].cpu().numpy()[0,0,:,:]

        
        
        f, (axin, axl, ax1comb) = plt.subplots(1,3, sharey=True)
        f.subplots_adjust(hspace=0.3)
        plt.tight_layout()
        
        #plot image
        image = axin.imshow(input_data,
                            aspect = 'equal')
        f.colorbar(image, ax=axin, orientation='vertical', fraction = 0.05)
        
        axl.imshow(l_data,
                   aspect = 'equal')
        
        
        combined = input_data + 1 * l_data 
        
        ax1comb.imshow(combined, aspect = 'equal')
        plt.show()
        
    def __getitem__(self, idx):
        """This function will allow us to index the data object and it will 
        return a sample."""
        name = self.names[idx]
        
        
        inputdata = np.load(os.path.join(self.imagedir, name))
        labeldata = np.load(os.path.join(self.labeldir, name))
        #self.inn = inputdata
        inputdata = inputdata.astype(float)
        labeldata = labeldata.astype(float)
        #print(inputdata.shape)
        #print(labeldata.shape)
        #self.inn = inputdata
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
        labeldata = torch.tensor(labeldata).to('cuda').float()
        inputdata = torch.tensor(inputdata).to('cuda').float()
        
        return {'deployin': inputdata,
                'deploylabel':labeldata,
                'name': name}
        
    def __len__(self):    
        """This function is mandated by Pytorch and allows us to see how many 
        data points we have in our dataset"""
        return len(self.names)

class UncMeasure(object):
    def __init__(self, opt, model, testnames, dataset):
        self.opt = opt
        self.model = model
        self.testnames = testnames
        self.dataset = dataset
        self.loss_fn1 = m.Dice_Loss()
        self.loss_fn2 = torch.nn.BCELoss(size_average=True)
        self.loss_fn3 = torch.nn.MSELoss(size_average=True)
        self.deployloader = DataLoader(self.dataset,
                                     batch_size = 1,
                                     shuffle= False)
        
    def print_data(self, name, threshold=False):
        model.eval()
        
        data_dir = '/media/arjun/VascLab EVO/projects/oct_ca_seg/actual final data'
        image_dir = os.path.join(data_dir, 'images')
        label_dir = os.path.join(data_dir, 'labels')
        
        inputdata, labeldata = self.getinandlabel(name, image_dir, label_dir)
        
        capsout, recon = self.model(inputdata)
        
        #detach() on variables
        capsout = capsout.detach()
        recon = recon.detach()
        
        f, (axin, axl, axc ,ax1comb) = plt.subplots(1,4, sharey=True)
        f.subplots_adjust(hspace=0.3)
        plt.tight_layout()
        
        #plot image
        image = axin.imshow(inputdata[0,0],
                            aspect = 'equal')
        
        
        axl.imshow(labeldata[0,0],
                   aspect = 'equal')
        
        lol=axc.imshow(capsout[0,0],
                   aspect = 'equal')
        
        combined = inputdata + 1 * capsout 
        
        ax1comb.imshow(combined[0,0], aspect = 'equal')
        
        f.colorbar(lol, ax=axc, orientation='vertical', fraction = 0.05)
        plt.show()
        '''
        if threshold:
            capsout[capsout>threshold] = 1
            capsout[capsout<threshold] = 0
        '''
        return True #inputdata, capsout.detach(), recon.detach(), labeldata
    
    def pred_arrays(self, name, threshold=False):
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
    
    def getinandlabel(self, name, image_dir, label_dir):
        inputdata = np.load(os.path.join(image_dir, name))
        labeldata = np.load(os.path.join(label_dir, name))
        #self.inn = inputdata
        inputdata = inputdata.astype(float)
        labeldata = labeldata.astype(float)
        #print(inputdata.shape)
        #print(labeldata.shape)
        #self.inn = inputdata
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
        return inputdata, labeldata
    
    def thresh(self, threshold):
        c = torch.tensor(self.caps.detach())
        c[c>threshold] = 1
        c[c<threshold] = 0
        return 1-self.loss_fn1(c, self.labeldata)
    
    def calc_unc(self, capsout, l, u):
        #capsout from model, thresh from othresh dict, alpha float (0,1)
        #c1 = torch.tensor(capsout.detach())
        c2 = np.array(capsout.detach())
        
        '''
        c1[c1>thresh] = 1
        c1[c1<thresh] = 0
        
        secthresh = alpha*thresh
        
        c2[c2>secthresh] = 1
        c2[c2<secthresh] = 0
        
        
        c1 = np.ma.masked_greater(c1,thresh)
        c1 = np.ma.getmaskarray(c1) * np.ma.getdata(c2)
        '''
        
        #secthresh = alpha*thresh
        maxx = c2.max()
        la = maxx * l
        ua = maxx * u
        
        c2 = np.ma.masked_outside(c2, la, ua)
        #c2 = np.ma.masked_less(c2,0.01)
        #c2 = np.ma.getmaskarray(c2) * np.ma.getdata(c2)
        

        
        return float(c2.mean()), float(c2.sum()), float(c2.std())
        
        
        #print(c1.sum(), c2.sum())
        
        #return float(np.std(c2))
        #return float(torch.sum(c2-c1)), float(torch.sum(c1)), float(np.std(np.array(c2-c1)))
    
    
    def deploy(self, thresholds=False, alpha=None): 
        '''
        thresholds has to be a optimized thresholds dictionary
        '''
        starttime = time.time()
        
        model.eval()
        data_dir = '/media/arjun/VascLab EVO/projects/oct_ca_seg/actual final data'
        image_dir = os.path.join(data_dir, 'images')
        label_dir = os.path.join(data_dir, 'labels')
        
        self.col_losses1 = []
        self.col_losses2 = []
        self.col_losses3 = []
        self.col_lossestotal = []
        
        self.softdicepairs = {}
        self.harddicepairs = {}
        self.accpairs = {}
        self.senspairs = {}
        self.specpairs = {}
        
        self.othresh = thresholds #dict of optimized threshs from delpoyBAYES
        self.alpha =alpha #factor of uncertainty you wanna investigate
        
        self.uncpairs = {}
        
        for i, sample in enumerate(self.deployloader):
            imtime=time.time()
            inputdata = sample['deployin']
            labeldata = sample['deploylabel']
            name = sample['name'][0]
            #inputdata, labeldata = self.getinandlabel(name, image_dir, label_dir)
            
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
                       
            if self.othresh is not None:
                #if threhsold = float [0,1] this will execute
                c = torch.tensor(self.caps.detach())
                t=self.othresh[name]
                
                #unc = self.calc_unc(c, t, self.alpha)
                unc = self.calc_unc(c, 0.05, self.alpha)
                self.uncpairs[name]=unc
                '''
                c[c>t] = 1
                c[c<t] = 0
                total = (c == labeldata).sum()
                self.senspairs[name] = float(sens(c,labeldata))
                self.specpairs[name] = float(spec(c,labeldata))
                self.accpairs[name] = int(total) / np.prod(labeldata.size()[2:4])
                '''
            #print(loss1.data)
            #self.dicepairs[name] = np.array(loss1.data)[0].astype(float)
            
            #print(name, t, time.time()-imtime)
            
            #print(name, 'DONE')
        self.endtime = time.time()-starttime

        return capsout, recon
        '''
        sys.stdout.write('Mean ' + str(1-np.mean(self.col_losses1)) + '\n' + \
                         'Std ' + str(np.std(self.col_losses1)) + '\n' + \
                         'Min ' +str(1-np.max(self.col_losses1)) + '\n' + \
                         'Max ' + str(1-np.min(self.col_losses1)) + '\n')
        '''





data_dir = '/media/arjun/VascLab EVO/projects/oct_ca_seg/actual final data'
 
 
#path to whichever model you want. usually will live in a ehckpoint
checkpoint = torch.load('/media/arjun/VascLab EVO/projects/oct_ca_seg/runsaves/Final1-pawsey/checkpoints/checkpoint.pt')

model = m.CapsNet(o.opt)
model.load_state_dict(checkpoint['model_state_dict'])
model.to('cuda')

#this should be the testsamples from your loaded model
testnames = os.listdir('/media/arjun/VascLab EVO/projects/oct_ca_seg/runsaves/Final1-pawsey/testsamples')

#this probs wont exist yet!!!
dicenames = list(np.load('/media/arjun/VascLab EVO/projects/oct_ca_seg/runsaves/Final1-pawsey/analysis/diceordered.npy'))

#load opimized threshold from deployBAYES run If no bayes then ignore this(youll get an error here)
f = open("/media/arjun/VascLab EVO/projects/oct_ca_seg/runsaves/Final1-pawsey/analysis/optimizedthresholds.json")
othresh = json.load(f)
f.close()


deploydata = DeployOCTDataset(data_dir, testnames)
uncd = UncMeasure(o.opt, model, testnames, deploydata)

uncd.deploy(othresh, alpha=0.95) 
'''
othresholds has to be optimized threshold dict. So can only run this once
deployBAYES has been run and you have a saved dict.
'''

f = open("/media/arjun/VascLab EVO/projects/oct_ca_seg/runsaves/Final1-pawsey/analysis/hardicepairs.json")
harddice = json.load(f)
f.close()


unc = []
hardd = []
thresh = []
for k,v in uncd.uncpairs.items():
    unc.append(uncd.uncpairs[k])
    hardd.append(harddice[k])
    thresh.append(othresh[k])


