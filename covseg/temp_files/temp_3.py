import random
import Tool_Functions.Functions as Functions
import numpy as np
import math
import random

patient_id = 'xghf-11_2020-07-31'
z = 320

rescaled = np.load('/home/zhoul0a/Desktop/prognosis_project/rescaled_ct_array/' + patient_id + '.npy')
rescaled[0, 0, z] = -0.5

Functions.image_save(np.clip(rescaled[:, :, z], -0.5, 0.5), '/home/zhoul0a/Desktop/transfer/19_healthy_rescaled.png', gray=True, high_resolution=True)
# Functions.image_save(np.clip(rescaled[:, :, 308], -0.5, 0.5), '/home/zhoul0a/Desktop/transfer/03.png', gray=True, high_resolution=True)

enhanced = np.load('/home/zhoul0a/Desktop/prognosis_project/rescaled_ct_enhanced/progonosis_enhanced/' + patient_id + '.npz')['array']


# Functions.image_save(np.clip(enhanced[:, :, 308], -0.1, 0.1), '/home/zhoul0a/Desktop/transfer/00.png', gray=True, high_resolution=True)

positive = np.array(np.clip(enhanced[:, :, z], 0, 0.1) * 10)


positive = positive * positive

Functions.image_save(positive, '/home/zhoul0a/Desktop/transfer/20_healthy_positive.png', gray=True, high_resolution=True)

negative = - (np.clip(enhanced[:, :, z], -0.1, -0.0))
negative = negative * 10

negative = negative * negative
negative = negative * negative
negative = negative * negative

Functions.image_save(negative, '/home/zhoul0a/Desktop/transfer/21_healthy_negative.png', gray=True, high_resolution=True)
exit()
Functions.image_save(negative, '/home/zhoul0a/Desktop/transfer/01.png', gray=True, high_resolution=True)

Functions.image_save(np.clip(enhanced[:, :, z], 0, 0.1), '/home/zhoul0a/Desktop/transfer/02.png', gray=True, high_resolution=True)
