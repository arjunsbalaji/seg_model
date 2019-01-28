#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 11:14:58 2019

@author: arjunbalaji
"""

#OCT dataset
import numpy as np
import os
import torch 
import sys
from torch.utils.data import Dataset
from sklearn import preprocessing
from skimage.transform import resize
from torchvision import transforms
import matplotlib.pyplot as plt

###############################################################################

def get_image(main_data_dir, name, image_type):
    this_data_path = os.path.join(main_data_dir, image_type)
    return np.genfromtxt(os.path.join(this_data_path, name), delimiter = ',')

###############################################################################
    
class RandomCrop(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        
        
    def __call__(self, sample):
        input_data = sample['input']
        label_data = sample['label']
        
        _, h, w = input_data.size()
        
        new_h, new_w = self.output_size
        
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        
        input_data = input_data[:, top:top + new_h, left:left + new_w]
        label_data = label_data[top:top + new_h, left:left + new_w]
        
        return {'input': input_data,
                'label': label_data,
                'case_name': sample['case_name']}
###############################################################################
        
#dataset class
class OCTDataset(Dataset):
    """
    First we create a dataset that will encapsulate our data. It has 3 special 
    functions which will be explained as they go. We will pass this dataset object
    to the torch dataloader object later which will make training easier.
    """
    def __init__ (self,
                  main_data_dir,
                  start_size,
                  input_shape,
                  transform = None):
        self.main_data_dir = main_data_dir
        self.start_size = start_size
        self.transform = transform
        self.input_shape = input_shape
        
        #iterate through the 2d images and get all their names
        name_list = []
        for im in os.listdir(os.path.join(self.main_data_dir, 'OG_IMAGES')):
            filename = os.fsdecode(im)
            name_list.append(filename)
            
        self.name_list = name_list
        self.pcrop = np.random.rand()
        self. phflip = np.random.rand()
        self.pvflip = np.random.rand()
        self.pafftrans = np.random.rand()
        
    def transformation(self, input_data, label):
        _, h, w = input_data.size()
        hnew, wnew = self.input_shape
        label = label.unsqueeze(0)
        combined = torch.cat((input_data, label), 0)
        combined = transforms.functional.to_pil_image(combined)
        #label = transforms.functional.to_pil_image(label)
        #sys.stdout.write('after pil')
        #random crop of startsize

            #label = transforms.functional.crop(label, i, left, hnew, wnew)
        #sys.stdout.write('after crop')    
        if self.phflip > 0.5:
            combined = transforms.functional.hflip(combined)
            #label = transforms.functional.hflip(label)
        #sys.stdout.write('after hflip')
        if self.pvflip > 0.5:
            combined = transforms.functional.vflip(combined)
            #label = transforms.functional.vflip(label)
        #sys.stdout.write('after vflip')
        '''
        if self.pafftrans > 0.01:
            angle = np.random.randint(-175, 175)
            #shear = np.random.randint(-175, 175)
            combined = transforms.functional.affine(combined,
                                                    translate = (0,0), 
                                                    angle=angle,
                                                    scale = 1,
                                                    shear=0)
            '''
        if self.pcrop > 0:
            i = np.random.randint(0, h - hnew)
            left = np.random.randint(0, w - wnew)
            combined = transforms.functional.crop(combined, i, left, hnew, wnew)
            
        combined = transforms.functional.to_tensor(combined)
        #label = transforms.functional.to_tensor(label)
        #sys.stdout.write('after done')
        
        #input_data = transforms.functional.normalize(input_data, [0,0,0], [1,1,1])
        return combined[:-1,:,:], combined[-1].unsqueeze(0)
        
    def visualise(self, idx):
        
        sample = self.__getitem__(idx)
        #print(sample['input'].size())
        #print(sample['label'].size())
        input_data = sample['input'].cpu().numpy()[0,:,:]
        label_data = sample['label'].cpu().numpy()[0,:,:]
        
        f, (axin, axl) = plt.subplots(1,2, sharey=True)
        f.subplots_adjust(hspace=0.3)
        plt.tight_layout()
        
        #plot image
        image = axin.imshow(input_data,
                            aspect = 'equal')
        f.colorbar(image, ax=axin, orientation='vertical', fraction = 0.05)
        
        axl.imshow(label_data,
                   aspect = 'equal')
        plt.show()
        
        
    def __getitem__(self, idx):
        """This function will allow us to index the data object and it will 
        return a sample."""
        name = self.name_list[idx]
        
        #label
        #TL = get_image(name, 'TL')
        #FL = get_image(name, 'FL')
        #ILT = get_image(name, 'ILT')      
        #print(self.main_data_dir, name, 'FILLED_OBJECTIVE')
        label = get_image(self.main_data_dir, name, 'FILLED_OBJECTIVE')[:,135:895]
        
        #this bit is hacky, but YOLO its to make the capsnet output shape match
        # the label shape 
        
        #label = label[2:-2]
        #label = label[:][:-1]
        #image data and filters

        image = get_image(self.main_data_dir, name, 'OG_IMAGES')[:,135:895]
        double_filter = get_image(self.main_data_dir, name, 'DOUBLE_FILTER')[:,135:895]
        long_grad = get_image(self.main_data_dir, name, 'LONG_GRAD')[:,135:895]
        
        
        if not self.transform:
            image = resize(image, output_shape = self.start_size)
            double_filter = resize(double_filter, output_shape = self.start_size)
            long_grad = resize(long_grad, output_shape = self.start_size)
        
            label = resize(label, output_shape = self.start_size)
            
            image = preprocessing.scale(image)
        #og = preprocessing.MinMaxScaler(og)

        #print(image.shape)
        sample = {'input': torch.cat((torch.tensor(image, dtype=torch.float32).unsqueeze(0),
                                      torch.tensor(double_filter, dtype=torch.float32).unsqueeze(0),
                                      torch.tensor(long_grad, dtype=torch.float32).unsqueeze(0))),
                  'label': torch.tensor(label, dtype=torch.float32),
                  'case_name': name}
        
        
        if self.transform:
            #sample = self.transform(sample)
            input_data, label_data = self.transformation(sample['input'], sample['label'])
            sample = {'input': input_data,
                      'label': label_data,
                      'case_name': name}

        return sample
    
    def __len__(self):    
        """This function is mandated by Pytorch and allows us to see how many 
        data points we have in our dataset"""
        return len(self.name_list)

###############################################################################
        