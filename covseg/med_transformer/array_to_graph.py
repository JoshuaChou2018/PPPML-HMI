"""
array can be any dimension in float32 format
input: data array in float32, mask array indicating where to convert to graph, interval_list
output: adjacency matrix, list for node merge size, list for node surface size, list for node signal,
array indicating node index
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import Tool_Functions.Functions as Functions
import format_convert.dcm_np_converter as converter
import analysis.connect_region_detect as get_connect_region


interval_list_lung_ct = [-np.inf] + list(np.arange(-950, 400, 30))
hash_list = []

eps = 0.00001


def cast_array_to_graph(data_array, mask_array=None, interval_list=None):
    """

    :param data_array: like freq rescaled CT scan
    :param mask_array: freq binary array has same shape with data_array, indicating where to convert to graph
    :param interval_list: voxel in data_array will be converted into int according to the interval_list
    interval_list[0] < data_array <= interval_list[1]    cast to 0, ...
    interval_list[-1] < data_array <= np.inf             cast to len(interval_list)
    :return: adjacency matrix, list for node merge size, list for node surface size, list for node signal,
    array indicating node index
    adjacent is the boundary length for two nodes
    node merge size is how many voxel in the node, which is an approximated number
    node index is from 0 to node_number -1; node index -1 means not in the mask array
    """
    if interval_list is None:
        interval_list = interval_list_lung_ct
    pass


def search_interval(input_flow):
    # input_flow = (pid, parallel_count, data_array, interval_list)
    pid = input_flow[0]
    parallel_count = input_flow[1]
    data_array = input_flow[2]
    interval_list = input_flow[3]

    data_array_hashed_sub = np.zeros(np.shape(data_array), 'float32')
    intervals = len(interval_list)
    for i in range(pid, intervals - 1, parallel_count):
        print(i)
        loc_greater = data_array > interval_list[i]
        loc_lower = data_array <= interval_list[i + 1]
        data_array_hashed_sub[np.where(loc_greater * loc_lower > 0.5)] = hash_list[i]
    return data_array_hashed_sub


def cast_into_certain_interval(data_array, interval_list=None, parallel_count=10):
    """

    :param data_array:
    :param interval_list:
    :param parallel_count:
    :return: data_array with hash values based on the intervals, hash_list
    """
    if interval_list is None:
        interval_list = interval_list_lung_ct
    global hash_list
    for i in range(len(interval_list)):
        hash_list.append(hash(str(interval_list[i])))
    data_array_hashed = np.zeros(np.shape(data_array), 'float32')

    parallel_input = []
    for i in range(parallel_count):
        parallel_input.append((i, parallel_count, np.array(data_array), list(interval_list)))
    sub_array_list = Functions.func_parallel(search_interval, parallel_input)
    data_array_hashed[np.where(interval_list[-1] < data_array)] = hash_list[-1]
    for data_array_hashed_sub in sub_array_list:
        data_array_hashed = data_array_hashed + data_array_hashed_sub
    return data_array_hashed, hash_list


class SurfaceDetect(nn.Module):
    def __init__(self):
        super(SurfaceDetect, self).__init__()
        super().__init__()
        kernel = [[[[[0, 0, 0],
                     [0, 1, 0],
                     [0, 0, 0]],
                    [[0, 1, 0],
                     [1, -6, 1],
                     [0, 1, 0]],
                    [[0, 0, 0],
                     [0, 1, 0],
                     [0, 0, 0]]]]]
        kernel = torch.tensor(kernel).float()
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        x = func.conv3d(x, self.weight, padding=1)
        return x


class DimensionError(Exception):
    def __init__(self, array):
        self.shape = np.shape(array)
        self.dimension = len(self.shape)

    def __str__(self):
        print("invalid dimension of", self.dimension, ", array has shape", self.shape)


def surface_convolution(id_mask_same_interval):
    """
    :param id_mask_same_interval: in shape [x, y, z] or [batch_size, x, y, z]. x >= 3, y >= 3, z >= 3
    each of the id_mask_same_interval contains
    :return: array of the surface: 0 for inner voxels, node id for inner surface
    """
    convolution_layer = SurfaceDetect().cuda()
    if torch.cuda.device_count() > 1:
        convolution_layer = nn.DataParallel(convolution_layer)
    shape = np.shape(id_mask_same_interval)
    if len(shape) == 3:
        array = torch.from_numpy(id_mask_same_interval).unsqueeze(0).unsqueeze(0)
    elif len(shape) == 4:
        array = torch.from_numpy(id_mask_same_interval).unsqueeze(1)
    else:
        raise DimensionError(id_mask_same_interval)

    # now the array in shape [batch_size, 1, x, y, z]
    array_gpu = array.cuda()
    surface = convolution_layer(array_gpu)
    surface_mask = surface.clone().detach() < 0
    surface_id = array_gpu * surface_mask.float()
    surface_id = surface_id.to('cpu')
    surface_id = surface_id.numpy()

    if len(shape) == 3:
        return surface_id[0, 0, :, :, :]  # [x, y, z]
    else:
        return surface_id[:, 0, :, :, :]  # [batch_size, x, y, z]


def single_thread_abstract_interval_1(input_flow):
    data_array = input_flow[0]
    interval_list = input_flow[1]
    pointer = input_flow[2]
    return data_array >= interval_list[pointer]


def single_thread_abstract_interval_2(input_flow):
    data_array = input_flow[0]
    interval_list = input_flow[1]
    pointer = input_flow[2]
    return data_array < interval_list[pointer + 1]


def interval_connected_analysis(data_array, mask_array=None, interval_list=None, parallel_limit=200, leave_cpu=1):
    """
    the speed bottleneck is in the ram reading speed, for CT with shape [512, 512, 512] and 130 intervals,
    will take about 100G ram. reduce the parallel_limit to reduce the ram demand.
    :param data_array:
    :param mask_array: binary mask
    :param interval_list:
    :param parallel_limit:
    :param leave_cpu:
    :return: None
    """
    if interval_list is None:
        interval_list = interval_list_lung_ct
    if mask_array is None:
        mask_array = np.ones(np.shape(data_array), 'float32')
    node_id_array = np.zeros(np.shape(data_array), 'float32')
    list_node_merge_size = []
    number_intervals = len(interval_list)
    interval_list.append(np.inf)  # add freq new end to the original interval

    existing_nodes = 0
    bounding_box = Functions.get_bounding_box(mask_array)  # freq list [(x_min, x_max), (y_min, y_max), ...]

    x_min, x_max = bounding_box[0]
    y_min, y_max = bounding_box[1]
    z_min, z_max = bounding_box[2]
    shape = np.shape(data_array)
    x_min = max(x_min - 1, 0)
    x_max = min(x_max + 2, shape[0])
    y_min = max(y_min - 1, 0)
    y_max = min(y_max + 2, shape[1])
    z_min = max(z_min - 1, 0)
    z_max = min(z_max + 2, shape[2])  # ensure the rim of the cube is of zeros
    print("x_min, x_max, y_min, y_max, z_min, z_max", x_min, x_max, y_min, y_max, z_min, z_max)

    mask_array = mask_array[x_min: x_max, y_min: y_max, z_min: z_max]

    for i in range(0, number_intervals, parallel_limit):
        end_point = min(number_intervals - i, parallel_limit)
        interval_mask_list_lower = []
        interval_mask_list_upper = []
        print("abstracting mask within interval", i, 'to', i + end_point)
        for j in range(end_point):
            interval_mask_list_lower.append(data_array[x_min: x_max, y_min: y_max, z_min: z_max]
                                            >= interval_list[j + i])
            interval_mask_list_upper.append(data_array[x_min: x_max, y_min: y_max, z_min: z_max]
                                            < interval_list[j + i + 1])

        interval_mask_list = []
        for j in range(end_point):
            interval_mask_list.append((interval_mask_list_lower[j] * interval_mask_list_upper[j] * mask_array, j))

        print("connect analysis for interval", i, 'to', i + end_point)
        return_batch = Functions.func_parallel(get_connect_region.get_connected_regions_light, interval_mask_list,
                                               leave_cpu_num=leave_cpu)
        # return_batch is [id_loc_dict,  ...]

        for j in range(end_point):
            id_loc_dict = return_batch[j]
            num_nodes = np.max(list(id_loc_dict.keys()))
            print("existing nodes:", existing_nodes)
            print("interval", i + j, "has", num_nodes, "nodes")
            # here existing_nodes = len(list_node_merge_size)
            for node_id in range(1, num_nodes + 1):
                merge_size = len(id_loc_dict[node_id])
                list_node_merge_size.append(len(id_loc_dict[node_id]))  # record volume for nodes in the item
                existing_nodes += 1
                for loc in id_loc_dict[node_id]:
                    node_id_array[loc[0] + x_min, loc[1] + y_min, loc[2] + z_min] = existing_nodes
                    # merge_size_array[loc[0] + x_min, loc[1] + y_min, loc[2] + z_min] = merge_size

    return node_id_array, list_node_merge_size


def get_graph(node_id_array, mask_array=None):
    if mask_array is None:
        mask_array = np.ones(np.shape(node_id_array), 'float32')
    adjacent_dict = {}  # key: node id; value: [(adjacent_id, connection_edges_counts), ...]
    pass


ar = converter.dcm_to_spatial_rescaled('/home/zhoul0a/Desktop/pulmonary nodules/ct_and_gt/fjj-124/2020-03-10/Data/raw_data/')

mask = np.load('/home/zhoul0a/Desktop/pulmonary nodules/basic_tissue/fjj-124_2020-03-10.npz')['array']
mask = np.array(mask[:, :, :, 0] + mask[:, :, :, 1] > 0.5, 'float32')

interval_connected_analysis(ar, mask)

Functions.save_np_array('/home/zhoul0a/Desktop/pulmonary nodules/temp/', 'id_array2.npy', node_id_array)
# Functions.save_np_array('/home/zhoul0a/Desktop/pulmonary nodules/temp/', 'merge_size2.npy', merge_size_array)
Functions.pickle_save_object('/home/zhoul0a/Desktop/pulmonary nodules/temp/merge_list.pickle', list_node_merge_size)
exit()

mask = np.load('/home/zhoul0a/Desktop/pulmonary nodules/basic_tissue/fjj-124_2020-03-10.npz')['array']
mask = np.array(mask[:, :, :, 0] + mask[:, :, :, 1] > 0.5, 'float32')

Functions.image_show(ar[:, :, 256])
Functions.image_show(mask[:, :, 256])
ar_hash = cast_into_certain_interval(ar)
Functions.image_show(ar_hash[:, :, 256])
exit()
ar = np.array([[1, 3], [3, 6]])
ar[np.where(ar > 2)] = 50
print(ar)
exit()