#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 16:58:24 2019

@author: arjunbalaji
"""
from options import Options

o = Options()
o.parse()

if o.opt.save:
    o.save()    

import sys, os, torch, dataset, time, warnings, model
from pathlib import Path
import train1 as train
import test1 as test
import classification as c
import jutils as j

warnings.simplefilter('ignore')

start_time = time.time()
    
if o.opt.loadcheckpoint is not None:
    checkpoint = torch.load(o.opt.loadcheckpoint)
else:
    checkpoint = None

dataroot = Path(o.opt.dataroot)
total_names = os.listdir(dataroot/'images')

#for each mainA,B,C,D,E change these accordingly for 5 fold cv
train_names, val_names = dataset.get_subnamelists(total_names, 0., 0.2)
test_thresholds = [0.5, 0.75, 0.90, 0.95, 0.975]



traindata = dataset.OCTDataset(o.opt.dataroot,
                          name_list = train_names,
                          start_size=o.opt.start_size,
                          cropped_size=o.opt.c_size,
                          transform=o.opt.transforms,
                          input_images = [0,1,2])

valdata = dataset.OCTDataset(o.opt.dataroot,
                          name_list = val_names,
                          start_size=o.opt.start_size,
                          cropped_size=o.opt.c_size,
                          transform=False,
                          input_images = [0,1,2])


#this and beat sum(120) are to use 120 long data set!
#data = torch.utils.data.Subset(data, range(120))

#beat = [8708,900,2403]
#beat = [90,10,20]

#traindata, valdata, testdata = torch.utils.data.random_split(data, beat)



octmodel = model.CapsNet(o.opt)
octmodel.to(o.opt.device)

if o.opt.loadcheckpoint:
    octmodel.load_state_dict(checkpoint['model_state_dict'])


#these are only used im home testing, when randomsubset samplers are on in test and train!!! 
setsize={'train':range(100),
         'val':range(20)}




if o.opt.train:
    sys.stdout.write('Starting Training... ' + '\n' + '\n')
    Trainer = train.Train(o.opt, octmodel, traindata, valdata, setsize['train'], setsize['val'], checkpoint)
    Trainer.train()
    sys.stdout.write('Training completed in: ' + str(Trainer.traintime)+ 'secs.' +'\n' +'\n')
    

if o.opt.test:
    sys.stdout.write('Starting Testing... ' + str(time.time()-start_time) + 'secs so far.' + '\n' + '\n')
    Tester = test.Test(o.opt, octmodel, valdata, setsize['val'], test_thresholds)
    Tester.test()
    sys.stdout.write('Testing completed in: ' + str(Tester.testtime)+ 'secs.' +'\n')


checkpointpath = os.path.join(o.opt.runsaves_dir, o.opt.name, 'checkpoints')


if o.opt.save:
    torch.save({'model_state_dict' : Trainer.model.state_dict(),
                'optimizer_state_dict':Trainer.optimizer.state_dict(),
                'scheduler_state_dict': Trainer.scheduler.state_dict(),
                'epoch': o.opt.epochs,
                'loss' : Trainer.loss.data}, 
                os.path.join(checkpointpath, 'segcheckpoint.pt'))
    
sys.stdout.write('OCTSEG Completed in: ' + str(time.time() - start_time)+ 'secs.' +'\n' +'\n')

if o.opt.classify:
    sys.stdout.write('Starting Classify... ' + '\n')
    
    transfer_model_dict= torch.load(os.path.join(checkpointpath, 'segcheckpoint.pt'))['model_state_dict']
    
    labels = j.jsonloaddict(os.path.join(o.opt.runsaves_dir, o.opt.name,'analysis'), 'dicedata') #dicedata from now on
    labels = j.dict_to_difficulty(labels, 0.96)
    images_dir = os.path.join(o.opt.dataroot, 'images')
    
    classifydataset = c.OCTClassificationDataset(images_dir,
                                                        labels,
                                                        o.opt.start_size,
                                                        o.opt.c_size,
                                                        transform=True)
    
    cappy = c.ClassifyCapsNet(o.opt)
    cappy = cappy.to(o.opt.device)
    
    cappy.load_state_dict(transfer_model_dict, strict=False)
    
    Classifier = c.Classify(o.opt, cappy, classifydataset)
    Classifier.classify()    
    sys.stdout.write('total: ' + str(Classifier.total) + '  |   hard: ' + str(Classifier.hard) +'\n'+'\n')

    if o.opt.save:
        torch.save({'model_state_dict' : Classifier.model.state_dict(),
                    'optimizer_state_dict':Classifier.optim.state_dict(),
                    'epoch': o.opt.cl_e,
                    'loss' : Classifier.lossi.data}, 
                    os.path.join(checkpointpath, 'classifycheckpoint.pt'))
        
    sys.stdout.write('OCTSEG Completed in: ' + str(Classifier.endtime)+ 'secs.' +'\n')
       

