#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 16:58:24 2019

@author: arjunbalaji
"""
from options import OptionsB

o = OptionsB()
o.parse()

if o.opt.save:
    o.save()    

if o.opt.comet:
    from comet_ml import Experiment
    
import sys
import os
import numpy
import torch
import dataset
import time
import model
import train1 as train
import test1 as test
import warnings


warnings.simplefilter('ignore')
start_time = time.time()

if o.opt.comet:
    experiment = Experiment(api_key="aSb5hPnLFt1wjOyjJfTJy4fkJ",
                            project_name="general", workspace="arjunsbalaji")
else:
    experiment = None
    sys.stdout.write('No comet logging' + '\n')
    
if o.opt.loadcheckpoint is not None:
    checkpoint = torch.load(o.opt.loadcheckpoint)
else:
    checkpoint = None
    
data = dataset.OCTDataset(o.opt.dataroot,
                          start_size=o.opt.start_size,
                          cropped_size=o.opt.c_size,
                          transform=o.opt.transforms,
                          input_images = [0,1,2])

#this and beat sum(120) are to use 120 long data set!
#data = torch.utils.data.Subset(data, range(120))

beat = [8708,900,2403]
#beat = [90,10,20]

traindata, valdata, testdata = torch.utils.data.random_split(data, beat)

octmodel = model.CapsNet(o.opt)
octmodel.to(o.opt.device)

if o.opt.loadcheckpoint:
    octmodel.load_state_dict(checkpoint['model_state_dict'])
    
setsize={'train':range(70),
         'val':range(20),
         'test':range(30)}

if o.opt.train:
    sys.stdout.write('Starting Training... ' + '\n' + '\n')
    Trainer = train.Train(o.opt, octmodel, traindata, valdata, setsize['train'], setsize['val'], experiment, checkpoint)
    Trainer.train()
    sys.stdout.write('Training completed in: ' + str(Trainer.traintime)+ 'secs.' +'\n' +'\n')
    
    
if o.opt.test:
    sys.stdout.write('Starting Testing... ' + str(time.time()-start_time) + 'secs so far.' + '\n')
    Tester = test.Test(o.opt, octmodel, testdata, setsize['test'], experiment)
    Tester.test()
    sys.stdout.write('Testing completed in: ' + str(Tester.testtime)+ 'secs.' +'\n')
    
checkpointpath = os.path.join(o.opt.runsaves_dir, o.opt.name, 'checkpoints')


if o.opt.save:
    torch.save({'model_state_dict' : Trainer.model.state_dict(),
                'optimizer_state_dict':Trainer.optimizer.state_dict(),
                'scheduler_state_dict': Trainer.scheduler.state_dict(),
                'epoch': o.opt.epochs,
                'loss' : Trainer.loss.data}, 
                os.path.join(checkpointpath, 'checkpoint.pt'))
    
sys.stdout.write('Completed in: ' + str(time.time() - start_time)+ 'secs.' +'\n')
#"""torch.save(Trainer.model.state_dict(), checkpointpath + '/pytorchmodel.pt')
#torch.save(Trainer.optimizer.state_dict(), checkpointpath + '/optimizer.pt')
#torch.save(Trainer.scheduler.state_dict(), checkpointpath + '/scheduler.pt')
#analysispath = os.path.join(o.opt.runsaves_dir, o.opt.name, 'analysis')"""


