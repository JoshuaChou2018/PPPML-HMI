import numpy as np

import Tool_Functions.Functions as Functions

import os

array_name_list = os.listdir('/home/zhoul0a/Desktop/COVID-19 delta/rescaled_arrays/')

for array_name in array_name_list:
    print(array_name)
    if os.path.exists('/home/zhoul0a/Desktop/COVID-19 delta/visualization/lung_250/' + array_name[:-4] + '.png'):
        print("processed")
        continue

    rescaled_ct = np.load('/home/zhoul0a/Desktop/COVID-19 delta/rescaled_arrays/' + array_name)

    lung = np.load('/home/zhoul0a/Desktop/COVID-19 delta/masks/lung_mask/' + array_name[:-4] + '.npz')['array']

    Functions.image_save(Functions.merge_image_with_mask(np.clip((rescaled_ct[:, :, 250] + 0.5), 0, 1), lung[:, :, 250], show=False),
                         '/home/zhoul0a/Desktop/COVID-19 delta/visualization/lung_250/' + array_name[:-4], high_resolution=True)

