#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 11:01:14 2019

@author: arjun
"""

import os
import numpy as np
import shutil

main_dir = '/media/arjun/Arjun1TB/VascLabData/octdataextra'
dest_dir = '/media/arjun/VascLab EVO/projects/oct_ca_seg/new_data'

new_images_dir = os.path.join(dest_dir, 'images')
new_labels_dir = os.path.join(dest_dir, 'labels')

if os.path.exists(new_images_dir):
    shutil.rmtree(new_images_dir)
    
if os.path.exists(new_labels_dir):
    shutil.rmtree(new_labels_dir)
    
os.mkdir(new_images_dir)
os.mkdir(new_labels_dir)

i = 0

for case in os.listdir(main_dir):
    images_dir = os.path.join(main_dir, case, 'GRAY')
    lg_dir = os.path.join(main_dir, case, 'LONG')
    df_dir = os.path.join(main_dir, case, 'CONV')

    fo_dir = os.path.join(main_dir, case, 'OBJECTIVE')
    
    lengths = [len(os.listdir(images_dir)),
               len(os.listdir(lg_dir)),
               len(os.listdir(df_dir)),
               len(os.listdir(fo_dir))]
    
    if max(lengths) != min(lengths):
        print('This case has an issue with how much data it has! Skipped.')
        continue 
    
    print('Doing case:', case)
    for array in os.listdir(images_dir):
        image = np.genfromtxt(os.path.join(images_dir, array), delimiter= ',')
        lg = np.genfromtxt(os.path.join(lg_dir, array), delimiter= ',')
        df = np.genfromtxt(os.path.join(df_dir, array), delimiter= ',')
        
        #need to get rid of nans in df
        where = np.isnan(df)
        df[where] = 0
        
        fo = np.genfromtxt(os.path.join(fo_dir, array), delimiter= ',')
        
        image = np.expand_dims(image, 0)
        lg = np.expand_dims(lg, 0)
        df = np.expand_dims(df, 0)
        
        image = np.concatenate((image, df, lg), 0)
        
        fo = np.expand_dims(fo, 0)
        
        name = str(i).zfill(7) + '.npy'
        
        np.save(os.path.join(new_images_dir,name), image)
        np.save(os.path.join(new_labels_dir, name), fo)
        print(i, '/', lengths[0])
        i += 1

