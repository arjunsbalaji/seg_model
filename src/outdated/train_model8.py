#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 15:06:11 2018

@author: arjunbalaji
"""
#oct_segboi
# 7 takes in long grad as an input too

import capsnet_utils8 as cn
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

modelname = 'capsnet'
run_location = 'home' # 'home' #or 'pawsey'  o'r 'laptop
uptype = 'upsample'

run_name = modelname + uptype + '-' + run_location + '-' + time.asctime().replace(' ', '-')



cuda_device = torch.device('cuda:0' if torch.cuda.is_available () else 'cpu')

if run_location == 'home':    
    main_data_dir = '/media/arjun/VascLab EVO/projects/oct_ca_seg/data_100'
    save_spot = os.path.join('/media/arjun/VascLab EVO/projects/oct_ca_seg/run_saves', run_name)
elif run_location == 'pawsey':    
    main_data_dir = '/scratch/pawsey0271/abalaji/projects/oct_ca_seg/train_data'
    save_spot = os.path.join('/scratch/pawsey0271/abalaji/projects/oct_ca_seg/run_saves', run_name)
    warnings.simplefilter('ignore')
elif run_location == 'laptop':
    main_data_dir = '/home/arjunbalaji/Documents/Projects/oct_ca_seg/data'
    save_spot = os.path.join('/home/arjunbalaji/Documents/Projects/oct_ca_seg/run_saves', run_name)
    warnings.simplefilter('ignore')

if not os.path.exists(save_spot):
    os.mkdir(save_spot)
else:
    os.rmdir(os.path.exists(save_spot))
    os.mkdir(save_spot)

data = cn.OCTDataset(main_data_dir = main_data_dir,
                     start_size = (380, 512),
                     transform = None)
total_epoch = 1
batch_size = 1

#set up the loader object. increasing batchsize will increase memory usage.
loader = DataLoader(data,
                    batch_size = batch_size,
                    shuffle = False)

model_placeholder = cn.CapsNet(batch_size=batch_size,
                               uptype = uptype)
model_placeholder.to(cuda_device)
model_placeholder.train()

loss_fn1 = cn.Dice_Loss()
#loss_fn2 = torch.nn.BCEWithLogitsLoss()
loss_fn2 = torch.nn.BCELoss()
loss_fn3 = torch.nn.MSELoss()


###############################################################################
#this part figures out the illustrations
total_images = len(data)

batches_to_finish = total_images // batch_size

show_percentage = 10
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
optimizer = torch.optim.Adam(model_placeholder.parameters(), lr=0.0001)#, momentum=0.5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.3)

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
    sys.stdout.write('Learning rate from now = ' + str(scheduler.get_lr()) + '\n')
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
        
        
        
        optimizer.zero_grad()
        
        #pred = oct_unet(input_data)
        
        loss1 = loss_fn1(caps_out, label_data) #this is for my custom dice loss
        loss2 = loss_fn2(caps_out, label_data.float())
        loss3 = loss_fn3(reconstruct, lumen_masked)
        
        (0.05 * loss1 + loss2 + 0.01 * loss3).backward()
        #loss2.backward()
        
        optimizer.step()
        
        collection_of_losses1 += [float(loss1.data)]
        collection_of_losses2 += [float(loss2.data)]
        collection_of_losses3 += [float(loss3.data)]
        
        #collection_of_losses2 = torch.cat((collection_of_losses2,
         #                                  loss2.data))
        
    
        if i >= show_progress:    
            
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

np.save(save_spot + '/DICE.npy', np.array(collection_of_losses1))
np.save(save_spot + '/BCE.npy', np.array(collection_of_losses2))
np.save(save_spot + '/MSERecon.npy', np.array(collection_of_losses3))

torch.save(model_placeholder.state_dict(), save_spot + '/pytorchmodel.pt')


np.save(save_spot + '/pics.npy', saved_pictures.cpu().numpy())

end_time = time.time()

sys.stdout.write('Completion Time: ' + str(end_time - start_time) + ' secs')
