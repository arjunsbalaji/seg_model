#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 16:36:11 2019

@author: arjunbalaji
"""

import argparse
import os
import torch
import numpy as np
import time 

name = time.asctime().replace(' ' , '-')

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--location', type=str, default='pawsey', help='home / pawsey / laptop')
        self.parser.add_argument('--dataroot', default='/media/arjun/VascLab EVO/projects/oct_ca_seg/actual final data', help='path to point clouds and labels. this is figured out depending on location')
        self.parser.add_argument('--name', type=str, default='bangs', help='name of the experiment.')
        
        self.parser.add_argument('--runsaves_dir', type=str, default='/media/arjun/VascLab EVO/projects/oct_ca_seg/runsaves', help='models are saved here. this is figured out depending on location')
        self.parser.add_argument('--save', type=bool, default=True, help='Whether to save checkpoints and analysis')
        self.parser.add_argument('--comet', type=bool, default=False, help='Whether to log on comet.')
        
        self.parser.add_argument('--loadcheckpoint', type=str, default=None, help='load a training checkpoint? give path')
        
        
        self.parser.add_argument('--train', type=bool, default=True, help='True to train, False to not.')
        self.parser.add_argument('--val', type=bool, default=True, help='True to validate, False to not.')
        self.parser.add_argument('--test', type=bool, default=True, help='True to test, False to not.')
        
        self.parser.add_argument('--epochs', type=int, default=10, help='number of training epochs. Test epochs is always 1')
        self.parser.add_argument('--batch_size', type=int, default=20, help='input batch size')
        
        self.parser.add_argument('--uptype', type=str, default='deconv', help='upsample or deconv')
        self.parser.add_argument('--transforms', type=bool, default=True, help='Whether to use transforms on data. False for testing.')
        self.parser.add_argument('--nThreads', default=8, type=int, help='# threads for loading data')
                                 
        
        self.parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
        self.parser.add_argument('--display_id', type=int, default=200, help='window id of the web display')
        

        self.parser.add_argument('--start_size', type=tuple, default=(224,224), help='resize initial image to this size')
        self.parser.add_argument('--c_size', type=tuple, default=(224,224), help='cropped size ')
        self.parser.add_argument('--inputchannels', type=int, default=3, help='number of input channels (image, df, lg) = 3')
        self.parser.add_argument('--primmaps', type=int, default=1, help='#primary maps')
        self.parser.add_argument('--primdims', type=int, default=4, help='#primary capsule vector dimensions')                                                  
        self.parser.add_argument('--maps1', type=int, default=2, help='1st layer maps')
        self.parser.add_argument('--dims1', type=int, default=8, help='1st layer dims')
        self.parser.add_argument('--maps2', type=int, default=4, help='2nd layer maps')
        self.parser.add_argument('--dims2', type=int, default=16, help='2nd layer dims')
        self.parser.add_argument('--maps3', type=int, default=8, help='3rd layer maps')
        self.parser.add_argument('--dims3', type=int, default=16, help='3rd layer dims')
        self.parser.add_argument('--f1maps', type=int, default=2, help='f1 layer maps')                                                  
        self.parser.add_argument('--f1dims', type=int, default=24, help='f1 layer dims')
        self.parser.add_argument('--f2maps', type=int, default=1, help='f2 layer maps')
        self.parser.add_argument('--f2dims', type=int, default=16, help='f2 layer dims')
        self.parser.add_argument('--reconchannels', type=int, default=1, help='recon channels out')
        
        self.parser.add_argument('--activation', type=str, default='relu', help='activation function: relu, elu')
        self.parser.add_argument('--normalization', type=str, default='batch', help='normalization function: batch, instance')
        
        
        #only lr is being used rn. Should change other sched params to patience etc...
        self.parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
        self.parser.add_argument('--sgamma', type=float, default=0.8, help='scheduler gamma')
        self.parser.add_argument('--sstep', type=int, default=50, help='scheduler step')
        self.parser.add_argument('--la', type=float, default=0.1, help='loss 1 coefficient')
        self.parser.add_argument('--lb', type=float, default=1, help='loss 2 coefficient')
        self.parser.add_argument('--lc', type=float, default=0.05, help='loss 3 coefficient')
        
        self.parser.add_argument('--threshes', type=str, default='[0.5, 0.75, 0.90, 0.95, 0.975]', help='threshs in test')
        self.parser.add_argument('--logging', type=bool, default=True, help='create gpu mem logs. turn save on to save.')
        
        self.parser.add_argument('--classify', type=bool, default=True, help='True to classify, False to not. Only works if train&save==True')
        self.parser.add_argument('--cl_e', type=int, default=5, help='classification epochs')
        
        
        
        self.parser.add_argument('--verbose', type=int, default=True, help='verbosity; explanation goes here')
        self.initialized = True
    
    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        
        self.opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(1)
            np.random.seed(1)
        else:
            torch.manual_seed(1)
            np.random.seed(1)
            
        self.opt.name = self.opt.name + '-' + self.opt.location + '-' + name
        
        if self.opt.location == 'home':
            self.opt.dataroot = '/media/arjun/VascLab EVO/projects/oct_ca_seg/actual final data'
            self.opt.runsaves_dir = '/media/arjun/VascLab EVO/projects/oct_ca_seg/runsaves'
        elif self.opt.location == 'pawsey':
            self.opt.dataroot = '/group/pawsey0271/abalaji/projects/oct_ca_seg/actual final data'  
            self.opt.runsaves_dir = '/group/pawsey0271/abalaji/projects/oct_ca_seg/run_saves'
        elif self.opt.location == 'laptop':
            self.opt.dataroot ='/media/arjunbalaji/Arjun1TB/VascLabData/OCT MACHINA DATA/train_data' 
            self.opt.runsaves_dir = '/home/arjunbalaji/Documents/Projects/oct_ca_seg/run_saves'
        
        args = vars(self.opt)
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')
        
        return self.opt
    
    def save(self):
        if not self.initialized:
            self.parse()

        args = vars(self.opt)       
        expr_dir =  os.path.join(self.opt.runsaves_dir, self.opt.name)
        os.mkdir(expr_dir)
        os.mkdir(os.path.join(expr_dir, 'analysis'))
        os.mkdir(os.path.join(expr_dir, 'checkpoints'))
        
        file_name = os.path.join(expr_dir, 'analysis', 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return True

class OptionsA():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--location', type=str, default='pawsey', help='home / pawsey / laptop')
        self.parser.add_argument('--dataroot', default='/media/arjun/VascLab EVO/projects/oct_ca_seg/actual final data', help='path to point clouds and labels. this is figured out depending on location')
        self.parser.add_argument('--name', type=str, default='Final1', help='name of the experiment.')
        
        self.parser.add_argument('--runsaves_dir', type=str, default='/media/arjun/VascLab EVO/projects/oct_ca_seg/run_saves', help='models are saved here. this is figured out depending on location')
        self.parser.add_argument('--save', type=bool, default=True, help='Whether to save checkpoints and analysis')
        self.parser.add_argument('--comet', type=bool, default=False, help='Whether to log on comet.')
        
        self.parser.add_argument('--loadcheckpoint', type=str, default=None, help='load a training checkpoint? give path')
        
        self.parser.add_argument('--train', type=bool, default=True, help='True to train, False to not.')
        self.parser.add_argument('--val', type=bool, default=True, help='True to validate, False to not.')
        self.parser.add_argument('--test', type=bool, default=True, help='True to test, False to not.')
        
        self.parser.add_argument('--epochs', type=int, default=50, help='number of training epochs. Test epochs is always 1')
        self.parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        
        self.parser.add_argument('--uptype', type=str, default='deconv', help='upsample or deconv')
        self.parser.add_argument('--transforms', type=bool, default=True, help='Whether to use transforms on data. False for testing.')
        self.parser.add_argument('--nThreads', default=8, type=int, help='# threads for loading data')
                                 
        
        self.parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
        self.parser.add_argument('--display_id', type=int, default=200, help='window id of the web display')
        

        self.parser.add_argument('--start_size', type=tuple, default=(256,256), help='resize initial image to this size')
        self.parser.add_argument('--c_size', type=tuple, default=(256,256), help='cropped size ')
        self.parser.add_argument('--inputchannels', type=int, default=3, help='number of input channels (image, df, lg) = 3')
        self.parser.add_argument('--primmaps', type=int, default=4, help='#primary maps')
        self.parser.add_argument('--primdims', type=int, default=16, help='#primary capsule vector dimensions')                                                  
        self.parser.add_argument('--maps1', type=int, default=8, help='1st layer maps')
        self.parser.add_argument('--dims1', type=int, default=24, help='1st layer dims')
        self.parser.add_argument('--maps2', type=int, default=16, help='2nd layer maps')
        self.parser.add_argument('--dims2', type=int, default=32, help='2nd layer dims')
        self.parser.add_argument('--maps3', type=int, default=24, help='3rd layer maps')
        self.parser.add_argument('--dims3', type=int, default=48, help='3rd layer dims')
        self.parser.add_argument('--f1maps', type=int, default=2, help='f1 layer maps')                                                  
        self.parser.add_argument('--f1dims', type=int, default=32, help='f1 layer dims')
        self.parser.add_argument('--f2maps', type=int, default=1, help='f2 layer maps')
        self.parser.add_argument('--f2dims', type=int, default=16, help='f2 layer dims')
        self.parser.add_argument('--reconchannels', type=int, default=1, help='recon channels out')
        
        self.parser.add_argument('--activation', type=str, default='relu', help='activation function: relu, elu')
        self.parser.add_argument('--normalization', type=str, default='batch', help='normalization function: batch, instance')

        self.parser.add_argument('--lr', type=float, default=0.0008, help='learning rate')
        self.parser.add_argument('--sgamma', type=float, default=0.8, help='scheduler gamma')
        self.parser.add_argument('--sstep', type=int, default=50, help='scheduler step')
        self.parser.add_argument('--la', type=float, default=0.1, help='loss 1 coefficient')
        self.parser.add_argument('--lb', type=float, default=1, help='loss 2 coefficient')
        self.parser.add_argument('--lc', type=float, default=0.05, help='loss 3 coefficient')
        
        self.parser.add_argument('--logging', type=bool, default=True, help='create gpu mem logs. turn save on to save.')
        
        self.parser.add_argument('--verbose', type=int, default=True, help='verbosity; explanation goes here')
        self.initialized = True
    
    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        
        self.opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(2)
            np.random.seed(2)
        else:
            torch.manual_seed(2)
            np.random.seed(2)
            
        self.opt.name = self.opt.name + '-' + self.opt.location #+ '-' + name
        
        if self.opt.location == 'home':
            self.opt.dataroot = '/media/arjun/VascLab EVO/projects/oct_ca_seg/actual final data'
            self.opt.runsaves_dir = '/media/arjun/VascLab EVO/projects/oct_ca_seg/run_saves'
        elif self.opt.location == 'pawsey':
            self.opt.dataroot = '/group/pawsey0271/abalaji/projects/oct_ca_seg/actual final data'  
            self.opt.runsaves_dir = '/group/pawsey0271/abalaji/projects/oct_ca_seg/run_saves'
        elif self.opt.location == 'laptop':
            self.opt.dataroot ='/media/arjunbalaji/Arjun1TB/VascLabData/OCT MACHINA DATA/train_data' 
            self.opt.runsaves_dir = '/home/arjunbalaji/Documents/Projects/oct_ca_seg/run_saves'
        
        args = vars(self.opt)
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')
        
        return self.opt
    
    def save(self):
        if not self.initialized:
            self.parse()

        args = vars(self.opt)       
        expr_dir =  os.path.join(self.opt.runsaves_dir, self.opt.name)
        os.mkdir(expr_dir)
        os.mkdir(os.path.join(expr_dir, 'analysis'))
        os.mkdir(os.path.join(expr_dir, 'checkpoints'))
        
        file_name = os.path.join(expr_dir, 'analysis', 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return True


class OptionsB():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--location', type=str, default='pawsey', help='home / pawsey / laptop')
        self.parser.add_argument('--dataroot', default='/media/arjun/VascLab EVO/projects/oct_ca_seg/actual final data', help='path to point clouds and labels. this is figured out depending on location')
        self.parser.add_argument('--name', type=str, default='de_e20_s3', help='name of the experiment.')
        
        self.parser.add_argument('--runsaves_dir', type=str, default='/media/arjun/VascLab EVO/projects/oct_ca_seg/run_saves', help='models are saved here. this is figured out depending on location')
        self.parser.add_argument('--save', type=bool, default=True, help='Whether to save checkpoints and analysis')
        self.parser.add_argument('--comet', type=bool, default=False, help='Whether to log on comet.')
        
        self.parser.add_argument('--loadcheckpoint', type=str, default=None, help='load a training checkpoint? give path')
        
        self.parser.add_argument('--train', type=bool, default=True, help='True to train, False to not.')
        self.parser.add_argument('--val', type=bool, default=True, help='True to validate, False to not.')
        self.parser.add_argument('--test', type=bool, default=True, help='True to test, False to not.')
        
        self.parser.add_argument('--epochs', type=int, default=20, help='number of training epochs. Test epochs is always 1')
        self.parser.add_argument('--batch_size', type=int, default=2, help='input batch size')
        
        self.parser.add_argument('--uptype', type=str, default='deconv', help='upsample or deconv')
        self.parser.add_argument('--transforms', type=bool, default=True, help='Whether to use transforms on data. False for testing.')
        self.parser.add_argument('--nThreads', default=8, type=int, help='# threads for loading data')
                                 
        
        self.parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
        self.parser.add_argument('--display_id', type=int, default=200, help='window id of the web display')
        

        self.parser.add_argument('--start_size', type=tuple, default=(256,256), help='resize initial image to this size')
        self.parser.add_argument('--c_size', type=tuple, default=(256,256), help='cropped size ')
        self.parser.add_argument('--inputchannels', type=int, default=3, help='number of input channels (image, df, lg) = 3')
        self.parser.add_argument('--primmaps', type=int, default=4, help='#primary maps')
        self.parser.add_argument('--primdims', type=int, default=16, help='#primary capsule vector dimensions')                                                  
        self.parser.add_argument('--maps1', type=int, default=8, help='1st layer maps')
        self.parser.add_argument('--dims1', type=int, default=24, help='1st layer dims')
        self.parser.add_argument('--maps2', type=int, default=16, help='2nd layer maps')
        self.parser.add_argument('--dims2', type=int, default=32, help='2nd layer dims')
        self.parser.add_argument('--maps3', type=int, default=24, help='3rd layer maps')
        self.parser.add_argument('--dims3', type=int, default=48, help='3rd layer dims')
        self.parser.add_argument('--f1maps', type=int, default=2, help='f1 layer maps')                                                  
        self.parser.add_argument('--f1dims', type=int, default=32, help='f1 layer dims')
        self.parser.add_argument('--f2maps', type=int, default=1, help='f2 layer maps')
        self.parser.add_argument('--f2dims', type=int, default=16, help='f2 layer dims')
        self.parser.add_argument('--reconchannels', type=int, default=1, help='recon channels out')
        
        self.parser.add_argument('--activation', type=str, default='relu', help='activation function: relu, elu')
        self.parser.add_argument('--normalization', type=str, default='batch', help='normalization function: batch, instance')

        self.parser.add_argument('--lr', type=float, default=0.0008, help='learning rate')
        self.parser.add_argument('--sgamma', type=float, default=0.8, help='scheduler gamma')
        self.parser.add_argument('--sstep', type=int, default=50, help='scheduler step')
        self.parser.add_argument('--la', type=float, default=0.1, help='loss 1 coefficient')
        self.parser.add_argument('--lb', type=float, default=1, help='loss 2 coefficient')
        self.parser.add_argument('--lc', type=float, default=0.05, help='loss 3 coefficient')
        
        self.parser.add_argument('--logging', type=bool, default=True, help='create gpu mem logs. turn save on to save.')
        
        self.parser.add_argument('--verbose', type=int, default=True, help='verbosity; explanation goes here')
        self.initialized = True
    
    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        
        self.opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(3)
            np.random.seed(3)
        else:
            torch.manual_seed(3)
            np.random.seed(3)
            
        self.opt.name = self.opt.name + '-' + self.opt.location + '-' + name
        
        if self.opt.location == 'home':
            self.opt.dataroot = '/media/arjun/VascLab EVO/projects/oct_ca_seg/actual final data'
            self.opt.runsaves_dir = '/media/arjun/VascLab EVO/projects/oct_ca_seg/run_saves'
        elif self.opt.location == 'pawsey':
            self.opt.dataroot = '/scratch/pawsey0271/abalaji/projects/oct_ca_seg/actual final data'  
            self.opt.runsaves_dir = '/scratch/pawsey0271/abalaji/projects/oct_ca_seg/run_saves'
        elif self.opt.location == 'laptop':
            self.opt.dataroot ='/media/arjunbalaji/Arjun1TB/VascLabData/OCT MACHINA DATA/train_data' 
            self.opt.runsaves_dir = '/home/arjunbalaji/Documents/Projects/oct_ca_seg/run_saves'
        
        args = vars(self.opt)
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')
        
        return self.opt
    
    def save(self):
        if not self.initialized:
            self.parse()

        args = vars(self.opt)       
        expr_dir =  os.path.join(self.opt.runsaves_dir, self.opt.name)
        os.mkdir(expr_dir)
        os.mkdir(os.path.join(expr_dir, 'analysis'))
        os.mkdir(os.path.join(expr_dir, 'checkpoints'))
        
        file_name = os.path.join(expr_dir, 'analysis', 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return True
    
class OptionsHome():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--location', type=str, default='home', help='home / pawsey / laptop')
        self.parser.add_argument('--dataroot', default='/media/arjun/VascLab EVO/projects/oct_ca_seg/actual final data', help='path to point clouds and labels. this is figured out depending on location')
        self.parser.add_argument('--name', type=str, default='aaa', help='name of the experiment.')
        
        self.parser.add_argument('--runsaves_dir', type=str, default='/media/arjun/VascLab EVO/projects/oct_ca_seg/run_saves', help='models are saved here. this is figured out depending on location')
        self.parser.add_argument('--save', type=bool, default=True, help='Whether to save checkpoints and analysis')
        self.parser.add_argument('--comet', type=bool, default=False, help='Whether to log on comet.')
        
        self.parser.add_argument('--loadcheckpoint', type=str, default=None, help='load a training checkpoint? give path')
        
        self.parser.add_argument('--train', type=bool, default=True, help='True to train, False to not.')
        self.parser.add_argument('--val', type=bool, default=True, help='True to validate, False to not.')
        self.parser.add_argument('--test', type=bool, default=True, help='True to test, False to not.')
        
        self.parser.add_argument('--epochs', type=int, default=1, help='number of training epochs. Test epochs is always 1')
        self.parser.add_argument('--batch_size', type=int, default=2, help='input batch size')
        
        self.parser.add_argument('--uptype', type=str, default='deconv', help='upsample or deconv')
        self.parser.add_argument('--transforms', type=bool, default=True, help='Whether to use transforms on data. False for testing.')
        self.parser.add_argument('--nThreads', default=8, type=int, help='# threads for loading data')
                                 
        
        self.parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
        self.parser.add_argument('--display_id', type=int, default=200, help='window id of the web display')
        

        self.parser.add_argument('--start_size', type=tuple, default=(128,128), help='resize initial image to this size')
        self.parser.add_argument('--c_size', type=tuple, default=(128,128), help='cropped size ')
        self.parser.add_argument('--inputchannels', type=int, default=3, help='number of input channels (image, df, lg) = 3')
        self.parser.add_argument('--primmaps', type=int, default=1, help='#primary maps')
        self.parser.add_argument('--primdims', type=int, default=4, help='#primary capsule vector dimensions')                                                  
        self.parser.add_argument('--maps1', type=int, default=2, help='1st layer maps')
        self.parser.add_argument('--dims1', type=int, default=16, help='1st layer dims')
        self.parser.add_argument('--maps2', type=int, default=4, help='2nd layer maps')
        self.parser.add_argument('--dims2', type=int, default=16, help='2nd layer dims')
        self.parser.add_argument('--maps3', type=int, default=8, help='3rd layer maps')
        self.parser.add_argument('--dims3', type=int, default=16, help='3rd layer dims')
        self.parser.add_argument('--f1maps', type=int, default=2, help='f1 layer maps')                                                  
        self.parser.add_argument('--f1dims', type=int, default=24, help='f1 layer dims')
        self.parser.add_argument('--f2maps', type=int, default=1, help='f2 layer maps')
        self.parser.add_argument('--f2dims', type=int, default=16, help='f2 layer dims')
        self.parser.add_argument('--reconchannels', type=int, default=1, help='recon channels out')
        
        self.parser.add_argument('--activation', type=str, default='relu', help='activation function: relu, elu')
        self.parser.add_argument('--normalization', type=str, default='batch', help='normalization function: batch, instance')

        self.parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
        self.parser.add_argument('--sgamma', type=float, default=0.8, help='scheduler gamma')
        self.parser.add_argument('--sstep', type=int, default=50, help='scheduler step')
        
        self.parser.add_argument('--la', type=float, default=0.1, help='loss 1 coefficient')
        self.parser.add_argument('--lb', type=float, default=1, help='loss 2 coefficient')
        self.parser.add_argument('--lc', type=float, default=0.05, help='loss 3 coefficient')
        
        
        self.parser.add_argument('--threshes', type=str, default='[0.5, 0.75, 0.90, 0.95, 0.975]', help='threshs in test')
        
        self.parser.add_argument('--classify', type=bool, default=True, help='True to classify, False to not. Only works if train&save==True')
        self.parser.add_argument('--cl_e', type=int, default=1, help='classification epochs')
        
        
        
        self.parser.add_argument('--logging', type=bool, default=True, help='create gpu mem logs. turn save on to save.')
        self.parser.add_argument('--verbose', type=int, default=True, help='verbosity; explanation goes here')
        self.initialized = True
    
    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        
        self.opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(4)
            np.random.seed(4)
        else:
            torch.manual_seed(4)
            np.random.seed(4)
            
        self.opt.name = self.opt.name + '-' + self.opt.location + '-' + name
        
        if self.opt.location == 'home':
            self.opt.dataroot = '/media/arjun/VascLab EVO/projects/oct_ca_seg/actual final data'
            self.opt.runsaves_dir = '/media/arjun/VascLab EVO/projects/oct_ca_seg/runsaves'
        elif self.opt.location == 'pawsey':
            self.opt.dataroot = '/scratch/pawsey0271/abalaji/projects/oct_ca_seg/actual final data'  
            self.opt.runsaves_dir = '/scratch/pawsey0271/abalaji/projects/oct_ca_seg/run_saves'
        elif self.opt.location == 'laptop':
            self.opt.dataroot ='/media/arjunbalaji/Arjun1TB/VascLabData/OCT MACHINA DATA/train_data' 
            self.opt.runsaves_dir = '/home/arjunbalaji/Documents/Projects/oct_ca_seg/run_saves'
        
        args = vars(self.opt)
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')
        
        return self.opt
    
    def save(self):
        if not self.initialized:
            self.parse()

        args = vars(self.opt)       
        expr_dir =  os.path.join(self.opt.runsaves_dir, self.opt.name)
        os.mkdir(expr_dir)
        os.mkdir(os.path.join(expr_dir, 'analysis'))
        os.mkdir(os.path.join(expr_dir, 'checkpoints'))
        
        file_name = os.path.join(expr_dir, 'analysis', 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return True
