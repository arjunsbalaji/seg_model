#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 15:06:11 2018

@author: arjunbalaji
"""
#oct_segboi
# 7 takes in long grad as an input too

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
import json 

torch.manual_seed(7)
np.random.seed(7)

start_time = time.time()

class Train(object):
    def __init__(self, args, run_name):
        self.args = args
        self.run_name = run_name
        
        self.cuda_device = torch.device('cuda:0' if torch.cuda.is_available () else 'cpu')
    
        if args['location'] == 'home':    
            #self.main_data_dir = '/media/arjun/VascLab EVO/projects/oct_ca_seg/data_10'
            self.main_data_dir = '/media/arjun/Arjun1TB/OCT MACHINA DATA/test_data'
            self.save_spot = os.path.join('/media/arjun/VascLab EVO/projects/oct_ca_seg/run_saves', run_name)
        elif args['location'] == 'pawsey':    
            self.main_data_dir = '/scratch/pawsey0271/abalaji/projects/oct_ca_seg/train_data'
            self.save_spot = os.path.join('/scratch/pawsey0271/abalaji/projects/oct_ca_seg/run_saves', run_name)
            warnings.simplefilter('ignore')
        elif args['location'] == 'laptop':
            self.main_data_dir = '/home/arjunbalaji/Documents/Projects/oct_ca_seg/1_data_train'
            self.save_spot = os.path.join('/home/arjunbalaji/Documents/Projects/oct_ca_seg/run_saves', run_name)
            warnings.simplefilter('ignore')
    
        if not os.path.exists(self.save_spot):
            os.mkdir(self.save_spot)
        else:
            os.rmdir(os.path.exists(self.save_spot))
            os.mkdir(self.save_spot)
    
        self.data = octdata.OCTDataset(main_data_dir = self.main_data_dir,
                                       start_size = args['model_args']['raw size'],
                                       cropped_size=args['model_args']['cropped size'],
                                       transform = args['transforms'])
        
        self.total_epoch = args['epochs']
        self.batch_size = args['batch_size']
    
        #set up the loader object. increasing batchsize will increase memory usage.
        self.loader = DataLoader(self.data,
                            batch_size = args['batch_size'],
                            shuffle = False)
    
        self.model_placeholder = utils.CapsNet(batch_size=args['batch_size'],
                                               args=args,
                                               model_args = args['model_args'],
                                               uptype = args['uptype'])
        if args['load_checkpoint']:
            loaded_model = torch.load(os.path.join(self.args['load_checkpoint'], 'pytorchmodel.pt'))
            self.model_placeholder.load_state_dict(loaded_model)
            #del loaded_model
        
        self.model_placeholder.to(self.cuda_device)
        self.model_placeholder.train()
    
        self.loss_fn1 = utils.Dice_Loss() 
        #loss_fn2 = torch.nn.BCEWithLogitsLoss()
        self.loss_fn2 = torch.nn.BCELoss()
        self.loss_fn3 = torch.nn.MSELoss()

        ###############################################################################
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
        ###############################################################################
        
        self.time_prediction_list = []
        
        #adam optimizer seems to work well. note the american spelling with a 'z'.
        self.optimizer = torch.optim.Adam(self.model_placeholder.parameters(),
                                     lr=args['init_lr'])
        if args['load_checkpoint']:
            loaded_optimzer = torch.load(os.path.join(self.args['load_checkpoint'], 'optimizer.pt'))
            self.optimizer.load_state_dict(loaded_optimzer)
            del loaded_optimzer
            
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                    step_size = args['scheduler_step'],
                                                    gamma = args['scheduler_gamma'])
        if args['load_checkpoint']:
            loaded_scheduler = torch.load(os.path.join(self.args['load_checkpoint'], 'scheduler.pt'))
            self.scheduler.load_state_dict(loaded_scheduler)
            del loaded_scheduler
        
        self.collection_of_losses1 = []
        #collection_of_losses1 = collection_of_losses1.to(cuda_device)
        
        self.collection_of_losses2 = []
        #collection_of_losses2 = collection_of_losses2.to(cuda_device)
        self.collection_of_losses3 = []
        
        saved_pictures = torch.tensor([])
        saved_pictures = saved_pictures.to(self.cuda_device)
        
    def train(self):
        saved_pictures = torch.tensor([])
        saved_pictures = saved_pictures.to(self.cuda_device)
        
        for epoch in range(self.total_epoch):
        
            show_progress = 0
            self.scheduler.step()
            sys.stdout.write('\n')
            sys.stdout.write('Learning rate for epoch ' + str(epoch + 1) + ' is ' + str(self.scheduler.get_lr()) + '\n')
        
            for i, sample in enumerate(self.loader):
                sample_start_time = time.time()
                self.sample = sample
                input_data = sample['input']
                input_data = input_data.float()
                #input_data = input_data.unsqueeze(1)
                input_data = input_data.to(self.cuda_device)
                #input_data = input_data
                #print(input_data.size(), 'i')
                
                label_data = sample['label']
                label_data = label_data.float()
                label_data = label_data.to(self.cuda_device)
                
                #print(label_data.size(), 'l1')
                label_data = label_data.squeeze()
                
                #make the label size correct depending on batch size!
                if len(label_data.size()) == 2:
                    #print(label_data.size(), 'l2')
                    label_data = torch.unsqueeze(label_data, 0)
                    #print(label_data.size(), 'l3')
                    label_data = torch.unsqueeze(label_data, 1)
                else:
                    label_data = torch.unsqueeze(label_data, 1)
                #print(label_data.size())
                caps_out, reconstruct = self.model_placeholder(input_data)
                
                #print(caps_out.size(), reconstruct.size())
                #label_data = torch.randint(0,2,(1, 3, 128, 128))
                #print('input size -', input_data.size()) 
                #print(pred.size(), 'pred size')
                #print('label size -', label_data.size())
                
                lumen_masked = (input_data[:,0,:,:].unsqueeze(1)) * label_data
                
                #print(lumen_masked.size())
                
                
                self.optimizer.zero_grad()
                
                #pred = oct_unet(input_data)
                
                loss1 = self.loss_fn1(caps_out, label_data) #this is for my custom dice loss
                loss2 = self.loss_fn2(caps_out, label_data.float())
                loss3 = self.loss_fn3(reconstruct, lumen_masked)
                
                (self.args['loss1_alpha'] * loss1 + self.args['loss2_alpha'] * loss2 + self.args['loss3_alpha'] * loss3).backward()
                #loss2.backward()
                
                self.optimizer.step()
                
                self.collection_of_losses1 += [float(loss1.data)]
                self.collection_of_losses2 += [float(loss2.data)]
                self.collection_of_losses3 += [float(loss3.data)]
                
                #collection_of_losses2 = torch.cat((collection_of_losses2,
                 #                                  loss2.data))
                
            
                if i >= show_progress and self.args['display_text']:    
                    
                    time_left = (time.time() - sample_start_time) * ((self.total_epoch * self.total_images) - ((epoch + 1) * self.batch_size * (i+1)))
                    self.time_prediction_list.append(time_left)
                    #note 1-dice loss to get back actual dice similarity coefficient
                    nth_image = int(epoch * self.total_images / self.batch_size + i)
                    sys.stdout.write('Epoch ' + str(epoch + 1) + ' ')
                    sys.stdout.write('| ' + str( (i * self.batch_size)  + 1) + ' ')
                    sys.stdout.write('| ' + 'DSM = ' + str(1 - self.collection_of_losses1[nth_image]) + ' ')
                    sys.stdout.write('| ' + 'BCE loss = '+ str(self.collection_of_losses2[nth_image]) + ' ')
                    sys.stdout.write('| ' + 'R loss = ' + str(self.collection_of_losses3[nth_image]) + ' ')
                    sys.stdout.write('| ' + 'Time remaining = ' +  str(np.round(time_left, 0)) + ' secs' + '\n')
                    
                    #pad_out = torch.nn.ZeroPad2d((0,0,2,2))
                    
                    #save the first sample per batch
                    input_to_save = input_data.data[0,0,:,:].unsqueeze(0).unsqueeze(0)
                    caps_to_save = caps_out.data[0,:,:,:].unsqueeze(0)
                    label_to_save = label_data.data[0,:,:,:].unsqueeze(0)
                    reconc_to_save = reconstruct.data[0,:,:,:].unsqueeze(0)
                    
                    saved_pictures = torch.cat((saved_pictures,
                                                torch.cat((input_to_save,
                                                           caps_to_save,
                                                           label_to_save,
                                                           reconc_to_save), 1)))
                    
                    #saved_pictures = torch.cat((saved_pictures, images_to_save))
                    show_progress += self.show_chunks
                break
                #print(pred.squeeze().size(), loss1.data, loss2.data)
        #with open(os.path.join(save_spot, 'run_name.txt'), "w") as text_file:
        #    text_file.write(run_name)
            
        if self.args['save_analysis']:
            analysis_spot = os.path.join(self.save_spot, 'analysis')
            os.mkdir(analysis_spot)
            np.save(analysis_spot + '/DICE.npy', np.array(self.collection_of_losses1))
            np.save(analysis_spot + '/BCE.npy', np.array(self.collection_of_losses2))
            np.save(analysis_spot + '/MSERecon.npy', np.array(self.collection_of_losses3))
            np.save(analysis_spot + '/pics.npy', saved_pictures.cpu().numpy())
            temp_args = self.args.copy()
            temp_args['transforms'] = str(self.args['transforms'])
            with open(os.path.join(analysis_spot, 'args.json'), 'w') as fp:
                json.dump(temp_args, fp, indent = 4)


        state_spot = os.path.join(self.save_spot, 'checkpoint')
        os.mkdir(state_spot)
        if self.args['checkpoint_save']:
            torch.save(self.model_placeholder.state_dict(), state_spot + '/pytorchmodel.pt')
            torch.save(self.optimizer.state_dict(), state_spot + '/optimizer.pt')
            torch.save(self.scheduler.state_dict(), state_spot + '/scheduler.pt')
        else:
            torch.save(self.model_placeholder.state_dict(), state_spot + '/pytorchmodel.pt')
    
        end_time = time.time()
        
        sys.stdout.write('Completion Time: ' + str(end_time - start_time) + ' secs' + '\n' + '\n' + '\n')
            
            
        
        
        
        
        
        
        
        
        
        
        
        
'''
def train(args, run_name):
    

    cuda_device = torch.device('cuda:0' if torch.cuda.is_available () else 'cpu')

    if args['location'] == 'home':    
        main_data_dir = '/media/arjun/VascLab EVO/projects/oct_ca_seg/data_100'
        save_spot = os.path.join('/media/arjun/VascLab EVO/projects/oct_ca_seg/run_saves', run_name)
    elif args['location'] == 'pawsey':    
        main_data_dir = '/scratch/pawsey0271/abalaji/projects/oct_ca_seg/train_data'
        save_spot = os.path.join('/scratch/pawsey0271/abalaji/projects/oct_ca_seg/run_saves', run_name)
        warnings.simplefilter('ignore')
    elif args['location'] == 'laptop':
        main_data_dir = '/home/arjunbalaji/Documents/Projects/oct_ca_seg/1_data_train'
        save_spot = os.path.join('/home/arjunbalaji/Documents/Projects/oct_ca_seg/run_saves', run_name)
        warnings.simplefilter('ignore')

    if not os.path.exists(save_spot):
        os.mkdir(save_spot)
    else:
        os.rmdir(os.path.exists(save_spot))
        os.mkdir(save_spot)

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
    if args['load_checkpoint']:
        loaded_model = torch.load(args['load_model'])
        model_placeholder.load_state_dict(loaded_model)
        del loaded_model
    
    model_placeholder.to(cuda_device)
    model_placeholder.train()

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
    ###############################################################################
    
    time_prediction_list = []
    
    #adam optimizer seems to work well. note the american spelling with a 'z'.
    optimizer = torch.optim.Adam(model_placeholder.parameters(),
                                 lr=args['init_lr'])
    if args['load_checkpoint']:
        loaded_optimzer = torch.load(os.path.join(save_spot, 'checkpoint', 'optimizer.pt'))
        model_placeholder.load_state_dict(loaded_optimzer)
        del loaded_optimzer
        
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size = args['scheduler_step'],
                                                gamma = args['scheduler_gamma'])
    if args['load_checkpoint']:
        loaded_scheduler = torch.load(os.path.join(save_spot, 'checkpoint', 'scheduler.pt'))
        model_placeholder.load_state_dict(loaded_scheduler)
        del loaded_scheduler
    
    collection_of_losses1 = []
    #collection_of_losses1 = collection_of_losses1.to(cuda_device)
    
    collection_of_losses2 = []
    #collection_of_losses2 = collection_of_losses2.to(cuda_device)
    collection_of_losses3 = []
    
    saved_pictures = torch.tensor([])
    saved_pictures = saved_pictures.to(cuda_device)
    
    for epoch in range(total_epoch):
        
        show_progress = 0
        scheduler.step()
        sys.stdout.write('\n')
        sys.stdout.write('Learning rate for epoch ' + str(epoch + 1) + ' is ' + str(scheduler.get_lr()) + '\n')
        
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
            break
            #label_data = torch.randint(0,2,(1, 3, 128, 128))
            #print('input size -', input_data.size()) 
            #print(pred.size(), 'pred size')
            #print('label size -', label_data.size())
            
            lumen_masked = input_data[:,0,:,:] * label_data
            
            
            
            optimizer.zero_grad()
            
            #pred = oct_unet(input_data)
            
            loss1 = loss_fn1(caps_out, label_data) #this is for my custom dice loss
            loss2 = loss_fn2(caps_out, label_data.float())
            loss3 = loss_fn3(reconstruct, lumen_masked)
            
            (args['loss1_alpha'] * loss1 + args['loss2_alpha'] * loss2 + args['loss3_alpha'] * loss3).backward()
            #loss2.backward()
            
            optimizer.step()
            
            collection_of_losses1 += [float(loss1.data)]
            collection_of_losses2 += [float(loss2.data)]
            collection_of_losses3 += [float(loss3.data)]
            
            #collection_of_losses2 = torch.cat((collection_of_losses2,
             #                                  loss2.data))
            
        
            if i >= show_progress and args['display_text']:    
                
                time_left = (time.time() - sample_start_time) * ((total_epoch * total_images) - ((epoch + 1) * batch_size * (i+1)))
                time_prediction_list.append(time_left)
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
    #with open(os.path.join(save_spot, 'run_name.txt'), "w") as text_file:
    #    text_file.write(run_name)
        
    if args['save_analysis']:
        analysis_spot = os.path.join(save_spot, 'analysis')
        os.mkdir(analysis_spot)
        np.save(analysis_spot + '/DICE.npy', np.array(collection_of_losses1))
        np.save(analysis_spot + '/BCE.npy', np.array(collection_of_losses2))
        np.save(analysis_spot + '/MSERecon.npy', np.array(collection_of_losses3))
        np.save(analysis_spot + '/pics.npy', saved_pictures.cpu().numpy())
        temp_args = args.copy()
        temp_args['transforms'] = str(args['transforms'])
        with open(os.path.join(analysis_spot, 'args.json'), 'w') as fp:
            json.dump(temp_args, fp, indent = 4)

    if args['checkpoint_save']:
        state_spot = os.path.join(save_spot, 'checkpoint')
        os.mkdir(state_spot)
        torch.save(model_placeholder.state_dict(), state_spot + '/pytorchmodel.pt')
        torch.save(optimizer.state_dict(), state_spot + '/optimizer.pt')
        torch.save(scheduler.state_dict(), state_spot + '/scheduler.pt')
    else:
        torch.save(model_placeholder.state_dict(), save_spot + '/pytorchmodel.pt')

    end_time = time.time()
    
    sys.stdout.write('Completion Time: ' + str(end_time - start_time) + ' secs' + '\n' + '\n' + '\n')
'''