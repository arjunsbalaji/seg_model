#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 13:56:23 2019

@author: arjun
"""

#rename raw data and make into neat data for oct

import os 
import shutil
import numpy as np

this_file_location = os.getcwd()

os.chdir('..')
os.chdir('..')
os.chdir('./data_raw')

raw_data_dir = os.getcwd()

raw_image_data_dir = os.path.join(raw_data_dir, 'OG_IMAGES')
raw_fo_data_dir = os.path.join(raw_data_dir, 'DOUBLE_FILTER')
raw_df_data_dir = os.path.join(raw_data_dir, 'FILLED_OBJECTIVE')

i = 0
"""
for name in sorted(os.listdir(raw_image_data_dir)):
    name_path = os.path.join(raw_image_data_dir, name)
    os.rename(name_path, 
              os.path.join(raw_image_data_dir, str(i).zfill(5)+'.csv'))
    i += 1
"""
for name in sorted(os.listdir(raw_df_data_dir)):
    name_path = os.path.join(raw_df_data_dir, name)
    os.rename(name_path, 
              os.path.join(raw_df_data_dir, str(i).zfill(5)+'.csv'))
    i += 1