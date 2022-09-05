import random
import Tool_Functions.Functions as Functions
import numpy as np
import math
import random

a = np.load('/home/zhoul0a/Desktop/prognosis_project/scale_free_property/adjacency_strict/air_way/Z/frequency_area.npy')
b = np.load('/home/zhoul0a/Desktop/prognosis_project/scale_free_property/adjacency_strict/air_way/X/frequency_area.npy')
c = np.load('/home/zhoul0a/Desktop/prognosis_project/scale_free_property/adjacency_strict/air_way/Y/frequency_area.npy')
d = a + b + c
d = d * (420667 / np.sum(d))
Functions.scale_free_check(np.arange(1, 200), d[1: 200], cache=20, remove_front_cache=1, label='airway regions', pic_save_path='/home/zhoul0a/Desktop/transfer/Scale_free_airway.png')

d = np.load('/home/zhoul0a/Desktop/prognosis_project/scale_free_property/adjacency_strict/blood_vessel/Y/frequency_area.npy')
d = d * (1594446 / np.sum(d))
Functions.scale_free_check(np.arange(1, 500), d[1: 500], cache=50, remove_front_cache=1, label='blood vessel regions', pic_save_path='/home/zhoul0a/Desktop/transfer/Scale_free_blood.png')



exit()


# here prove log log plot can make good estimate for yita
yita = -1.75
prob_list = []
scale = np.arange(1, 100001)
for i in range(0, 100000):
    prob = math.pow(i + 1, yita)
    prob_list.append(prob)
prob_list = np.array(prob_list)
prob_list = prob_list / np.sum(prob_list)

cdf_list = np.zeros((100000,), 'float32')
cdf_list[0] = prob_list[0]
for i in range(1, 100000):
    cdf_list[i] = prob_list[i] + cdf_list[i - 1]

freq_list = np.zeros((100000,), 'float32')


def one_thread(pid_id):
    return_list = np.zeros((100000,), 'float32')
    for i in range(pid_id, 1000000, 40):
        t = random.uniform(0, 1)
        for j in range(100000):
            if t < cdf_list[j]:
                return_list[j] += 1
                break
    return return_list


item_list = Functions.func_parallel(one_thread, np.arange(40), leave_cpu_num=6)
for item in item_list:
    freq_list = freq_list + item

Functions.scale_free_check_cdf(scale, freq_list, log_p_min=-5, remove_front_point=10)
Functions.scale_free_check(scale[0: 200], freq_list[0: 200], cache=20, remove_front_cache=1)
exit()






array = np.load('/home/zhoul0a/Desktop/prognosis_project/rescaled_ct_enhanced/progonosis_enhanced/xghf-25_2020-07-29.npz')['array']
Functions.image_show(np.clip(array[:, :, 308], -0.05, 0.20), gray=True)

exit()


import Tool_Functions.Functions as Functions

Functions.read_in_mha('/home/zhoul0a/Desktop/其它肺炎/Immune_Pneumonia/raw_data/CIP07/2020-03-10/Data/ground_truth/CIP.mha')
exit()
import math

import numpy as np

freq = np.zeros([1008, ], 'float32')


def prob(k):
    #return math.pow(k, -2.5)
    return (-math.log(k) + k - 1) / (k*k*k*math.log(k)*math.log(k))


total_p = 0
for i in range(8, 1008):
    total_p += prob(i)
print(total_p)


cum = np.zeros([1008, ], 'float32')
current_cum = 0
for i in range(8, 1008):
    cum[i] = (prob(i) + current_cum) / total_p
    current_cum += prob(i)

for i in range(1000000):
    r = random.uniform(0, 1)
    for j in range(8, 1008):
        if r < cum[j]:
            freq[j] += 1
            break

b = np.arange(8, 1008)
Functions.scale_free_check_cdf(b, freq[8::], show=True)
