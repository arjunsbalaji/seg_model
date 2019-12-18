#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 15:47:16 2019

@author: arjun
"""

#new oct data formatting images, labels
import os
import numpy as np
import shutil

main_dir = '/media/arjun/Arjun1TB/OCT MACHINA DATA/test_data'

images_dir = os.path.join(main_dir, 'OG_IMAGES')
lg_dir = os.path.join(main_dir, 'LONG_GRAD')
df_dir = os.path.join(main_dir, 'DOUBLE_FILTER')

fo_dir = os.path.join(main_dir, 'FILLED_OBJECTIVE')

new_images_dir = os.path.join(main_dir, 'images')
new_labels_dir = os.path.join(main_dir, 'labels')

if os.path.exists(new_images_dir):
    shutil.rmtree(new_images_dir)
    
if os.path.exists(new_labels_dir):
    shutil.rmtree(new_labels_dir)
    
os.mkdir(new_images_dir)
os.mkdir(new_labels_dir)



for name in os.listdir(images_dir):
    print('Doing: ', name)
    image_array = np.genfromtxt(os.path.join(images_dir, name), delimiter = ',')
    lg_array = np.genfromtxt(os.path.join(lg_dir, name), delimiter = ',')
    df_array = np.genfromtxt(os.path.join(df_dir, name), delimiter = ',')
    fo_array = np.genfromtxt(os.path.join(fo_dir, name), delimiter = ',')
    
    
    image_array = np.expand_dims(image_array, 0)
    lg_array = np.expand_dims(lg_array, 0)
    df_array = np.expand_dims(df_array, 0)
    fo_array = np.expand_dims(fo_array, 0)
    
    images = np.concatenate([image_array, df_array, lg_array], 0)
    
    np.save(os.path.join(new_images_dir, name[:-4]), images)
    np.save(os.path.join(new_labels_dir, name[:-4]), fo_array)
