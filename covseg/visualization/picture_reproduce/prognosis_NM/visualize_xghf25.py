import random
import Tool_Functions.Functions as Functions
import numpy as np
import math
import random

rescaled = np.load('/home/zhoul0a/Desktop/prognosis_project/rescaled_ct_array/xghf-25_2020-07-29.npy')
rescaled[0, 0, 308] = -0.5
Functions.image_save(np.clip(rescaled[:, :, 308], -0.5, 0.5), '/home/zhoul0a/Desktop/transfer/03.png', gray=True, high_resolution=True)

enhanced = np.load('/home/zhoul0a/Desktop/prognosis_project/rescaled_ct_enhanced/progonosis_enhanced/xghf-25_2020-07-29.npz')['array']


Functions.image_save(np.clip(enhanced[:, :, 308], -0.1, 0.1), '/home/zhoul0a/Desktop/transfer/00.png', gray=True, high_resolution=True)

negative = 0.1 - np.clip(enhanced[:, :, 308], -0.1, 0)
negative = negative * 10
negative = negative * negative
negative = negative * negative
negative = negative * negative
Functions.image_save(negative, '/home/zhoul0a/Desktop/transfer/01.png', gray=True, high_resolution=True)

Functions.image_save(np.clip(enhanced[:, :, 308], 0, 0.1), '/home/zhoul0a/Desktop/transfer/02.png', gray=True, high_resolution=True)
