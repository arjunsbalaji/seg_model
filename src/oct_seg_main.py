#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 22:46:48 2019

@author: arjun
"""

#CT_seg_main



#import oct_train
#import oct_test
#import os
import sys
import numpy as np
#import shutil
import time
#import oct_dataset as octdata
total_start_time = time.time()

'''
'home'
    main_data_dir = '/media/arjun/VascLab EVO/projects/oct_ca_seg/data_10'
    save_spot = os.path.join('/media/arjun/VascLab EVO/projects/oct_ca_seg/run_saves', run_name)
'pawsey'
    main_data_dir = '/scratch/pawsey0271/abalaji/projects/oct_ca_seg/train_data'
    save_spot = os.path.join('/scratch/pawsey0271/abalaji/projects/oct_ca_seg/run_saves', run_name)
    warnings.simplefilter('ignore')
'laptop'
    main_data_dir = '/home/arjunbalaji/Documents/Projects/oct_ca_seg/data'
    save_spot = os.path.join('/home/arjunbalaji/Documents/Projects/oct_ca_seg/run_saves', run_name)
    warnings.simplefilter('ignore')
'''

model_args = {'raw size': (256, 256),
              'cropped size': (256,256), # should be tuple. should match random crop arg
              'input channels': 3,
              'prim maps':4,
              'prim dims':16,
              '1 maps': 8,
              '1 dims': 24,
              '2 maps': 16,
              '2 dims': 32,              
              '3 maps': 24,
              '3 dims': 48,
              'final 1 maps': 2,
              'final 1 dims': 32,
              'final 2 maps': 1,
              'final 2 dims': 16,              
              'recon channels': 1}

args = {'location': 'pawsey',
        'model_args': model_args,
        'train': True, #False, # for resuming training #path to checkpoints folder in models run_save
        'load_checkpoint': False,#False, # for resuming training #path to checkpoints folder in models run_sav>>>>>>> 7f53909735dd2b4d3c19f361fa52defbe356f286
        'test': True,
        'load_model': False,# False or path to model. Note that this is only for testing. if you want to load a model to train, you MUST load a whole checkpoint.
        'display_text':True,
        'show_percentage': 10,
        'save_analysis':True, #True,
        'transforms': True, #must be set to true!
        'epochs': 350,
        'batch_size': 10, #int
        'uptype': 'deconv', #upsample or deconv
        'init_lr':0.0001,
        'scheduler_gamma': 0.8,
        'scheduler_step': 100,
        'loss1_alpha': 0.1,
        'loss2_alpha': 1,
        'loss3_alpha': 0.05,
        'checkpoint_save': True}#True}

run_name =  args['location'] + '--lr-' + str(args['init_lr']) + '--trans-' + str(args['transforms']) + '-' + time.asctime().replace(' ', '-')

sys.stdout.write('Run started at ' + time.asctime() + '\n') 
sys.stdout.write('Run name is: ' + run_name + '\n')
if args['train']:
    import train
    sys.stdout.write('-----------Training Model-----------' + '\n')
    #train.train(args, run_name)
    trainer = train.Train(args, run_name)
    trainer.train()
    model = trainer.model_placeholder
    
if args['test']:
    import test
    sys.stdout.write('-----------Testing Model-----------' + '\n')
    #test.test(args, run_name)
    
    tester = test.Test(args, run_name, None)
    tester.test()
    dice_mean = np.mean(tester.collection_of_losses1)
    dice_max = np.max(tester.collection_of_losses1)
    dice_min = np.min(tester.collection_of_losses1)
    dice_std = np.std(tester.collection_of_losses1)
    
    sys.stdout.write('Dice Metrics: mean=' + str(1 - dice_mean) + ' min=' + str(1 - dice_max) + ' max=' + str(1 - dice_min) + ' std=' + str(dice_std) + '\n')
total_end_time = time.time()

sys.stdout.write('Total Completion Time : ' + str(total_end_time-total_start_time) + ' secs')
