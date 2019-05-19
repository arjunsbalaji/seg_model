#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 16:58:24 2019

@author: arjunbalaji
"""

import sys
import os
import numpy
import torch
from options import Options
import dataset
import time
import model

o = Options()
o.parse()
#o.save

data = dataset.OCTDataset(o.opt.dataroot,
                          start_size=o.opt.start_size,
                          cropped_size=o.opt.c_size,
                          transform=o.opt.transforms)


#traindata, valdata, testdata = torch.utils.data.random_split([a,b,c])

octmodel = model.CapsNet(o.opt)