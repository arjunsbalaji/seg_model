#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 16:58:24 2019

@author: arjunbalaji
"""

from comet_ml import Experiment
import sys
import os
import numpy
import torch
from options import Options
import dataset
import time
import model
import train1
import test1
start_time = time.time()

experiment = Experiment(api_key="aSb5hPnLFt1wjOyjJfTJy4fkJ",
                        project_name="general", workspace="arjunsbalaji")

o = Options()
o.parse()
#o.save

data = dataset.OCTDataset(o.opt.dataroot,
                          start_size=o.opt.start_size,
                          cropped_size=o.opt.c_size,
                          transform=o.opt.transforms)


traindata, valdata, testdata = torch.utils.data.random_split(data, [1100,39,400])

octmodel = model.CapsNet(o.opt)
octmodel.to(o.opt.device)

if o.opt.train:
    sys.stdout.write('Starting Training... ' + '\n')
    Trainer = train1.Train(o.opt, octmodel, traindata, valdata, experiment)
    Trainer.train()
    sys.stdout.write('Training completed in: ' + str(Trainer.traintime)+ 'secs.' +'\n' +'\n')
    
    
if o.opt.test:
    sys.stdout.write('Starting Testing... ' + str(time.time()-start_time) + 'secs so far.' + '\n')
    Tester = test1.Test(o.opt, octmodel, testdata, experiment)
    Tester.test()
    sys.stdout.write('Testing completed in: ' + str(Tester.testtime)+ 'secs.' +'\n')
    
checkpointpath = os.path.join(o.opt.runsaves_dir, o.opt.name, 'checkpoints')

if o.opt.save:
    torch.save(Trainer.model.state_dict(), checkpointpath + '/pytorchmodel.pt')
    torch.save(Trainer.optimizer.state_dict(), checkpointpath + '/optimizer.pt')
    #torch.save(Trainer.scheduler.state_dict(), checkpointpath + '/scheduler.pt')
    
    analysispath = os.path.join(o.opt.runsaves_dir, o.opt.name, 'analysis')