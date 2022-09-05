import os
import Tool_Functions.Functions as Functions
import numpy as np

file_name_list = os.listdir('/home/zhoul0a/Desktop/prognosis_project/rescaled_ct_array')

for name in file_name_list:
    array = np.load('/home/zhoul0a/Desktop/prognosis_project/rescaled_ct_array/' + name)
    Functions.save_np_array('/home/zhoul0a/Desktop/prognosis_project/rescaled_ct_compressed', name[:-4], array, compress=True)

