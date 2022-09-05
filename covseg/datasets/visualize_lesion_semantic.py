import numpy as np

import Tool_Functions.Functions as Functions

import os

array_name_list = os.listdir('/home/zhoul0a/Desktop/COVID-19 delta/rescaled_arrays/')

for array_name in ['P0140557_20210807.npy']:
    print(array_name)
    if os.path.exists('/home/zhoul0a/Desktop/COVID-19 delta/visualization/lesion_mid_z/' + array_name[:-4] + '.png'):
        print("processed")
        continue

    rescaled_ct = np.load('/home/zhoul0a/Desktop/COVID-19 delta/rescaled_arrays/' + array_name)

    lesion = np.load('/home/zhoul0a/Desktop/COVID-19 delta/masks/lesions/' + array_name[:-4] + '.npz')['array']

    total_lesion = np.sum(lesion)
    if total_lesion < 100:
        mid_z = 0
    else:
        z_loc = np.where(lesion > 0.5)
        mid_z = int(np.median(z_loc[2]))

    print(mid_z)
    print(total_lesion)

    Functions.image_save(Functions.merge_image_with_mask(np.clip((rescaled_ct[:, :, mid_z] + 0.5), 0, 1), lesion[:, :, mid_z], show=False),
                         '/home/zhoul0a/Desktop/COVID-19 delta/visualization/lesion_mid_z/' + array_name[:-4], high_resolution=True)

