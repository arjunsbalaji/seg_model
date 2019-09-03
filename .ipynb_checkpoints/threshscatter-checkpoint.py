#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 19:42:28 2019

@author: arjun
"""
#this sheet is for Final1!!! it uses the outputs of deploBAYES
import os 
import numpy as np
import matplotlib.pyplot as plt
import json 

data_dir = '/media/arjun/VascLab EVO/projects/oct_ca_seg/actual final data'

dicenames = list(np.load('/media/arjun/VascLab EVO/projects/oct_ca_seg/runsaves/Final1-pawsey/analysis/diceordered.npy'))
testnames = os.listdir('/media/arjun/VascLab EVO/projects/oct_ca_seg/runsaves/Final1-pawsey/testsamples')

#load opimized threshold from deployBAYES run If no bayes then ignore this(youll get an error here)
f = open("/media/arjun/VascLab EVO/projects/oct_ca_seg/runsaves/Final1-pawsey/analysis/optimizedthresholds.json")
othresh = json.load(f)
f.close()

f = open("/media/arjun/VascLab EVO/projects/oct_ca_seg/runsaves/Final1-pawsey/analysis/hardicepairs.json")
harddice = json.load(f)
f.close()

f = open("/media/arjun/VascLab EVO/projects/oct_ca_seg/runsaves/Final1-pawsey/analysis/accpairs.json")
acc = json.load(f)
f.close()

f = open("/media/arjun/VascLab EVO/projects/oct_ca_seg/runsaves/Final1-pawsey/analysis/senspairs.json")
sens = json.load(f)
f.close()

f = open("/media/arjun/VascLab EVO/projects/oct_ca_seg/runsaves/Final1-pawsey/analysis/specpairs.json")
spec = json.load(f)
f.close()

f = open("/media/arjun/VascLab EVO/projects/oct_ca_seg/runsaves/Final1-pawsey/analysis/softdicepairs.json")
softdice = json.load(f)
f.close()


