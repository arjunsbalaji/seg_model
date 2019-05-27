#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 11:53:22 2019

@author: arjun
"""

import os
import numpy as np
import shutil
import matplotlib.pyplot as plt
import skimage.transform as skitransforms

#350GB is wayyy too much and itsbecause the images are 760x1024
#so we are gonna downsize them to the central 512x 512

data_dir = '/media/arjun/VascLab EVO/projects/oct_ca_seg/new final data'
dest_dir = '/media/arjun/VascLab EVO/projects/oct_ca_seg/actual final data'

imagesdir = os.path.join(data_dir, 'images')
labelsdir = os.path.join(data_dir, 'labels')

for name in  sorted(os.listdir(imagesdir)):
    image = np.load(os.path.join(imagesdir, name))
    label = np.load(os.path.join(labelsdir, name))
    
    image = image.astype(float)
    label = label.astype(float)
    
    
    image = np.transpose(image, (1, 2, 0))
    label = np.transpose(label, (1, 2, 0))
    
    #print(image.shape, label.shape)
    
    image = image[:, 132:892,:]
    label = label[:, 132:892,:]
    
    #print(image.shape, label.shape)
    
    image = skitransforms.resize(image, output_shape=(512, 512))
    label = skitransforms.resize(label, output_shape=(512, 512))
    
    #print(image.shape, label.shape)
    
    image = np.transpose(image, (2, 0, 1))
    label = np.transpose(label, (2, 0, 1))
    
    #print(image.shape, label.shape)
    
    np.save(os.path.join(dest_dir, 'images', name), image)
    np.save(os.path.join(dest_dir, 'labels', name), label)
    print(name)
    