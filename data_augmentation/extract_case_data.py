#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 11:01:14 2019

@author: arjun
"""

import os
import numpy as np
import shutil

#main_dir = '/media/arjun/Arjun1TB/VascLabData/octdataextra'
#main_dir = '/run/user/1000/gvfs/dav:host=unidrive.uwa.edu.au,ssl=true,prefix=%2Fstudents/irds/ECM-V-002/Machine Learning/OCT_Data/Lumen_Segmentation_Data'
main_dir = '/media/arjun/VascLab EVO/projects/oct_ca_seg/new final data/lachy data'
dest_dir = '/media/arjun/VascLab EVO/projects/oct_ca_seg/new final data'

new_images_dir = os.path.join(dest_dir, 'images')
new_labels_dir = os.path.join(dest_dir, 'labels')

if os.path.exists(new_images_dir):
    shutil.rmtree(new_images_dir)
    
if os.path.exists(new_labels_dir):
    shutil.rmtree(new_labels_dir)
    
os.mkdir(new_images_dir)
os.mkdir(new_labels_dir)



a = os.listdir(main_dir)
folders  = [x for x in a if len(x)==3]


for name in folders:

    casedir = os.path.join(main_dir, name)
    folders  = [x for x in a if len(x)==3]

    
    sets = [x for x in os.listdir(casedir) if x[0]=='I']

    for gather in sets:
        #mprint(gather)
        setdir = os.path.join(casedir, gather)
        
        #print(setdir)
        for case in os.listdir(setdir):
            images_dir = os.path.join(setdir, 'GRAY')
            lg_dir = os.path.join(setdir, 'LONG')
            df_dir = os.path.join(setdir, 'CONV')
        
            fo_dir = os.path.join(setdir, 'OBJECTIVE')
            
            lengths = [len(os.listdir(images_dir)),
                       len(os.listdir(lg_dir)),
                       len(os.listdir(df_dir)),
                       len(os.listdir(fo_dir))]
            
            if max(lengths) != min(lengths):
                print('There are unequal image type set sizes! Case Skipped.')
                continue 
            
            print('Doing case:', gather)
            i = 0
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

