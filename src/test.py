#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 11:58:54 2019

@author: arjunbalaji
"""

import utils as utils
import oct_dataset as octdata
import torch
from torch.utils.data import DataLoader
import time
import numpy as np
import os 
import sys
import shutil
from matplotlib import pyplot
import warnings

torch.manual_seed(7)

start_time = time.time()

class Test(object):
    def __init__(self, args, run_name):
        self.args = args
        self.run_name = run_name
        
        self.cuda_device = torch.device('cuda:0' if torch.cuda.is_available () else 'cpu')
        
        if args['location'] == 'home':    
            self.main_data_dir = '/media/arjun/VascLab EVO/projects/oct_ca_seg/TESTDATA'
            self.save_spot = os.path.join('/media/arjun/VascLab EVO/projects/oct_ca_seg/run_saves', run_name)
        elif args['location'] == 'pawsey':    
            self.main_data_dir = '/scratch/pawsey0271/abalaji/projects/oct_ca_seg/test_data'
            self.save_spot = os.path.join('/scratch/pawsey0271/abalaji/projects/oct_ca_seg/run_saves', run_name)
            warnings.simplefilter('ignore')
        elif args['location'] == 'laptop':
            self.main_data_dir = '/home/arjunbalaji/Documents/Projects/oct_ca_seg/1_data_test'
            self.save_spot = os.path.join('/home/arjunbalaji/Documents/Projects/oct_ca_seg/run_saves', run_name)
            warnings.simplefilter('ignore')
    
        self.data = octdata.OCTDataset(main_data_dir = self.main_data_dir,
                             start_size = args['model_args']['raw size'],
                             input_shape=args['model_args']['cropped size'],
                             transform = args['transforms'])
        self.total_epoch = args['epochs']
        self.batch_size = args['batch_size']
    
        #set up the loader object. increasing batchsize will increase memory usage.
        self.loader = DataLoader(self.data,
                        batch_size = self.batch_size,
                        shuffle = False)
    
        self.model_placeholder = utils.CapsNet(batch_size=self.batch_size,
                                               args=args,
                                          model_args = args['model_args'],
                                          uptype = args['uptype'])
        
        if args['load_model']:
            loaded_model = torch.load(args['load_model'])
            self.model_placeholder.load_state_dict(loaded_model)
            del loaded_model
            
        self.model_placeholder.to(self.cuda_device)
        self.model_placeholder.eval()
    
        self.loss_fn1 = utils.Dice_Loss()
        #loss_fn2 = torch.nn.BCEWithLogitsLoss()
        self.loss_fn2 = torch.nn.BCELoss()
        self.loss_fn3 = torch.nn.MSELoss()


        #this part figures out the illustrations
        self.total_images = len(self.data)
        
        batches_to_finish = self.total_images // self.batch_size
        
        show_percentage = args['show_percentage']
        self.show_chunks = batches_to_finish * show_percentage // 100
        
        
        ###############################################################################
        #initial print statements 
        #print('Data directory:', data_dir)
        #print('Model name:', model_name)
        #print('Total epochs:', total_epoch)
        #print('Total images:', total_images)
        sys.stdout.write('Total epochs:' + ' ' + str(self.total_epoch) + '\n' )
        sys.stdout.write('Total images:' + ' ' + str(self.total_images) + '\n' )
        
        self.collection_of_losses1 = []
        #collection_of_losses1 = collection_of_losses1.to(cuda_device)
        
        self.collection_of_losses2 = []
        #collection_of_losses2 = collection_of_losses2.to(cuda_device)
        self.collection_of_losses3 = []
        

    def test(self): 
        saved_pictures = torch.tensor([])
        saved_pictures = saved_pictures.to(self.cuda_device)
        for epoch in range(self.total_epoch):
            
            show_progress = 0
            sys.stdout.write('\n')
            
            for i, sample in enumerate(self.loader):
                sample_start_time = time.time()
                
                input_data = sample['input']
                input_data = input_data.float()
                #input_data = input_data.unsqueeze(1)
                input_data = input_data.to(self.cuda_device)
                input_data = input_data
                
                label_data = sample['label']
                label_data = label_data.float()
                label_data = label_data.to(self.cuda_device)
                label_data = label_data
                label_data = torch.unsqueeze(label_data, 1)
                
                caps_out, reconstruct = self.model_placeholder(input_data)
                
                #label_data = torch.randint(0,2,(1, 3, 128, 128))
                #print('input size -', input_data.size()) 
                #print(pred.size(), 'pred size')
                #print('label size -', label_data.size())
                
                lumen_masked = input_data[:,0,:,:] * label_data
                
                loss1 = self.loss_fn1(caps_out, label_data) #this is for my custom dice loss
                loss2 = self.loss_fn2(caps_out, label_data.float())
                loss3 = self.loss_fn3(reconstruct, lumen_masked)
                
                self.collection_of_losses1 += [float(loss1.data)]
                self.collection_of_losses2 += [float(loss2.data)]
                self.collection_of_losses3 += [float(loss3.data)]
                
                if i >= show_progress and self.args['display_text']:    
                    
                    time_left = (time.time() - sample_start_time) * ((self.total_epoch * self.total_images) - ((epoch + 1) * self.batch_size * (i+1)))
                    #note 1-dice loss to get back actual dice similarity coefficient
                    nth_image = epoch * self.total_images + i
                    sys.stdout.write('Epoch ' + str(epoch + 1) + ' ')
                    sys.stdout.write('| ' + str( (i * self.batch_size)  + 1) + ' ')
                    sys.stdout.write('| ' + 'DSM = ' + str(1 - self.collection_of_losses1[nth_image]) + ' ')
                    sys.stdout.write('| ' + 'BCE loss = '+ str(self.collection_of_losses2[nth_image]) + ' ')
                    sys.stdout.write('| ' + 'R loss = ' + str(self.collection_of_losses3[nth_image]) + ' ')
                    sys.stdout.write('| ' + 'Time remaining = ' +  str(np.round(time_left, 0)) + ' secs' + '\n')
                    
                    #pad_out = torch.nn.ZeroPad2d((0,0,2,2))
                    saved_pictures = torch.cat((saved_pictures,
                                                torch.cat((input_data.data[:,0].unsqueeze(0),
                                                           caps_out.data,
                                                           label_data.data), 1)))
                    
                    #saved_pictures = torch.cat((saved_pictures, images_to_save))
                    show_progress += self.show_chunks
                
                #print(pred.squeeze().size(), loss1.data, loss2.data)
        
        if self.args['save_analysis']:
            np.save(self.save_spot + '/test_DICE.npy', np.array(self.collection_of_losses1))
            np.save(self.save_spot + '/test_BCE.npy', np.array(self.collection_of_losses2))
            np.save(self.save_spot + '/test_MSERecon.npy', np.array(self.collection_of_losses3))
            np.save(self.save_spot + '/test_pics.npy', saved_pictures.cpu().numpy())
    
        end_time = time.time()
        
        sys.stdout.write('Completion Time: ' + str(end_time - start_time) + ' secs' + '\n' + '\n' + '\n')
                



'''
def test(args, run_name):
    cuda_device = torch.device('cuda:0' if torch.cuda.is_available () else 'cpu')
    
    if args['location'] == 'home':    
        main_data_dir = '/media/arjun/VascLab EVO/projects/oct_ca_seg/TESTDATA'
        save_spot = os.path.join('/media/arjun/VascLab EVO/projects/oct_ca_seg/run_saves', run_name)
    elif args['location'] == 'pawsey':    
        main_data_dir = '/scratch/pawsey0271/abalaji/projects/oct_ca_seg/test_data'
        save_spot = os.path.join('/scratch/pawsey0271/abalaji/projects/oct_ca_seg/run_saves', run_name)
        warnings.simplefilter('ignore')
    elif args['location'] == 'laptop':
        main_data_dir = '/home/arjunbalaji/Documents/Projects/oct_ca_seg/1_data_test'
        save_spot = os.path.join('/home/arjunbalaji/Documents/Projects/oct_ca_seg/run_saves', run_name)
        warnings.simplefilter('ignore')

    data = octdata.OCTDataset(main_data_dir = main_data_dir,
                         start_size = args['model_args']['start_size'],
                         transform = args['transforms'])
    total_epoch = args['epochs']
    batch_size = args['batch_size']

    #set up the loader object. increasing batchsize will increase memory usage.
    loader = DataLoader(data,
                    batch_size = batch_size,
                    shuffle = False)

    model_placeholder = utils.CapsNet(batch_size=batch_size,
                                      model_args = args['model_args'],
                                      uptype = args['uptype'])
    
    if args['load_model']:
        loaded_model = torch.load(args['load_model'])
        model_placeholder.load_state_dict(loaded_model)
        del loaded_model
        
    model_placeholder.to(cuda_device)
    model_placeholder.eval()

    loss_fn1 = utils.Dice_Loss()
    #loss_fn2 = torch.nn.BCEWithLogitsLoss()
    loss_fn2 = torch.nn.BCELoss()
    loss_fn3 = torch.nn.MSELoss()


    ###############################################################################
    #this part figures out the illustrations
    total_images = len(data)
    
    batches_to_finish = total_images // batch_size
    
    show_percentage = args['show_percentage']
    show_chunks = batches_to_finish * show_percentage // 100
    
    
    ###############################################################################
    #initial print statements 
    #print('Data directory:', data_dir)
    #print('Model name:', model_name)
    #print('Total epochs:', total_epoch)
    #print('Total images:', total_images)
    sys.stdout.write('Total epochs:' + ' ' + str(total_epoch) + '\n' )
    sys.stdout.write('Total images:' + ' ' + str(total_images) + '\n' )
    
    collection_of_losses1 = []
    #collection_of_losses1 = collection_of_losses1.to(cuda_device)
    
    collection_of_losses2 = []
    #collection_of_losses2 = collection_of_losses2.to(cuda_device)
    collection_of_losses3 = []
    
    saved_pictures = torch.tensor([])
    saved_pictures = saved_pictures.to(cuda_device)
    
    for epoch in range(total_epoch):
        
        show_progress = 0
        sys.stdout.write('\n')
        
        for i, sample in enumerate(loader):
            sample_start_time = time.time()
            
            input_data = sample['input']
            input_data = input_data.float()
            #input_data = input_data.unsqueeze(1)
            input_data = input_data.to(cuda_device)
            input_data = input_data
            
            label_data = sample['label']
            label_data = label_data.float()
            label_data = label_data.to(cuda_device)
            label_data = label_data
            label_data = torch.unsqueeze(label_data, 1)
            
            caps_out, reconstruct = model_placeholder(input_data)
            
            #label_data = torch.randint(0,2,(1, 3, 128, 128))
            #print('input size -', input_data.size()) 
            #print(pred.size(), 'pred size')
            #print('label size -', label_data.size())
            
            lumen_masked = input_data[:,0,:,:] * label_data
            
            loss1 = loss_fn1(caps_out, label_data) #this is for my custom dice loss
            loss2 = loss_fn2(caps_out, label_data.float())
            loss3 = loss_fn3(reconstruct, lumen_masked)
            
            collection_of_losses1 += [float(loss1.data)]
            collection_of_losses2 += [float(loss2.data)]
            collection_of_losses3 += [float(loss3.data)]
            
            if i >= show_progress and args['display_text']:    
                
                time_left = (time.time() - sample_start_time) * ((total_epoch * total_images) - ((epoch + 1) * batch_size * (i+1)))
                #note 1-dice loss to get back actual dice similarity coefficient
                nth_image = epoch * total_images + i
                sys.stdout.write('Epoch ' + str(epoch + 1) + ' ')
                sys.stdout.write('| ' + str( (i * batch_size)  + 1) + ' ')
                sys.stdout.write('| ' + 'DSM = ' + str(1 - collection_of_losses1[nth_image]) + ' ')
                sys.stdout.write('| ' + 'BCE loss = '+ str(collection_of_losses2[nth_image]) + ' ')
                sys.stdout.write('| ' + 'R loss = ' + str(collection_of_losses3[nth_image]) + ' ')
                sys.stdout.write('| ' + 'Time remaining = ' +  str(np.round(time_left, 0)) + ' secs' + '\n')
                
                #pad_out = torch.nn.ZeroPad2d((0,0,2,2))
                saved_pictures = torch.cat((saved_pictures,
                                            torch.cat((input_data.data[:,0].unsqueeze(0),
                                                       caps_out.data,
                                                       label_data.data), 1)))
                
                #saved_pictures = torch.cat((saved_pictures, images_to_save))
                show_progress += show_chunks
            
            #print(pred.squeeze().size(), loss1.data, loss2.data)
    
    if args['save_analysis']:
        np.save(save_spot + '/test_DICE.npy', np.array(collection_of_losses1))
        np.save(save_spot + '/test_BCE.npy', np.array(collection_of_losses2))
        np.save(save_spot + '/test_MSERecon.npy', np.array(collection_of_losses3))
        np.save(save_spot + '/test_pics.npy', saved_pictures.cpu().numpy())

    end_time = time.time()
    
    sys.stdout.write('Completion Time: ' + str(end_time - start_time) + ' secs' + '\n' + '\n' + '\n')
 '''           
    
    