import torch
import Tool_Functions.Functions as Functions
import numpy as np
from models.Unet_2D.test import *
import os
from sample_manager.sample_slicer_multi_classes import slice_one_sample

os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'

array_info = {
        "resolution": (1, 1, 1),
        "data_channel": 1,
        "enhanced_channel": 2,
        "window": (-1, 0, 1),
        "positive_semantic_channel": None,  # prediction phase this should be None
        "output_channels": 2,  # output_channels is 2: positive and negative
        "mute_output": True,  # if you want to see prediction details, set is as False
        "wrong_scan": None,
        "init_features": 16
    }


def stage_one_heat_map(file_name, show=False):
    print("processing", file_name)
    array = np.load('/home/zhoul0a/Desktop/Breast_Cancer_MICCAI/new/rescaled_array_hayida/' + file_name)[
        'array']

    array_info["enhanced_channel"] = 2

    rescaled_ct = array[:, :, :, 0: 3]

    gt_array = array[:, :, :, 3]

    locations = np.where(gt_array > 0.5)
    tumor_center = [int(np.median(locations[0])), int(np.median(locations[1])), int(np.median(locations[2]))]

    direction_list = ['X', 'Y', 'Z']
    view = 0
    for direction in direction_list:
        print(direction)
        check_point_path = '/home/zhoul0a/Desktop/Breast_Cancer_MICCAI/new/check_points/hayida/stage_one/' + direction + \
                           '/' + str(int(file_name.split('_')[0][-1]) % 5) + '_saved_model.pth'

        model = load_model(check_point_path, array_info)

        sample = slice_one_sample(rescaled_ct, array_info['resolution'], tumor_center[view], direction,
                                    array_info['data_channel'],
                                    array_info['enhanced_channel'], array_info['window'])
        if view == 0:
            predict, heat_map = heat_map_segment(sample, gt_array[tumor_center[view], :, :], model, 1)
        elif view == 1:
            predict, heat_map = heat_map_segment(sample, gt_array[:, tumor_center[view], :], model, 1)
        else:
            predict, heat_map = heat_map_segment(sample, gt_array[:, :, tumor_center[view]], model, 1)
        Functions.save_np_array('/home/zhoul0a/Desktop/Breast_Cancer_MICCAI/new/visualization/heat_maps/stage_one/',
                                file_name[:-4] + '_sample_' + direction, sample, compress=False)
        Functions.save_np_array('/home/zhoul0a/Desktop/Breast_Cancer_MICCAI/new/visualization/heat_maps/stage_one/',
                                file_name[:-4] + '_heatmap_' + direction, heat_map, compress=False)
        if show:
            print(np.shape(sample), np.shape(predict), np.shape(heat_map))

            Functions.merge_image_with_mask(sample[:, :, 1], np.array(predict[:, :] > 0.15, 'float32'))
            for i in range(5):
                Functions.array_stat(heat_map[:, :, i])
                Functions.image_show(heat_map[:, :, i])
        view += 1


def stage_two_heat_map(file_name, show=False):
    print("processing", file_name)
    array = np.load('/home/zhoul0a/Desktop/Breast_Cancer_MICCAI/new/enhanced_arrays/' + file_name)[
        'array']

    array_info["enhanced_channel"] = 3

    rescaled_ct = array[:, :, :, 0: 4]

    gt_array = array[:, :, :, 4]

    locations = np.where(gt_array > 0.5)
    tumor_center = [int(np.median(locations[0])), int(np.median(locations[1])), int(np.median(locations[2]))]

    direction_list = ['X', 'Y', 'Z']
    view = 0
    for direction in direction_list:
        print(direction)
        check_point_path = '/home/zhoul0a/Desktop/Breast_Cancer_MICCAI/new/check_points/hayida/stage_two/' + direction + \
                           '/' + str(int(file_name.split('_')[0][-1]) % 5) + '_saved_model.pth'

        model = load_model(check_point_path, array_info)

        sample = slice_one_sample(rescaled_ct, array_info['resolution'], tumor_center[view], direction,
                                    array_info['data_channel'],
                                    array_info['enhanced_channel'], array_info['window'])
        if view == 0:
            predict, heat_map = heat_map_segment(sample, gt_array[tumor_center[view], :, :], model, 1)
        elif view == 1:
            predict, heat_map = heat_map_segment(sample, gt_array[:, tumor_center[view], :], model, 1)
        else:
            predict, heat_map = heat_map_segment(sample, gt_array[:, :, tumor_center[view]], model, 1)

        Functions.save_np_array('/home/zhoul0a/Desktop/Breast_Cancer_MICCAI/new/visualization/heat_maps/stage_two/',
                                file_name[:-4] + '_sample_' + direction, sample, compress=False)
        Functions.save_np_array('/home/zhoul0a/Desktop/Breast_Cancer_MICCAI/new/visualization/heat_maps/stage_two/',
                                file_name[:-4] + '_heatmap_' + direction, heat_map, compress=False)

        if show:
            print(np.shape(sample), np.shape(predict), np.shape(heat_map))
            Functions.merge_image_with_mask(sample[:, :, 1], np.array(predict[:, :] > 0.15, 'float32'))
            for i in range(6):
                Functions.array_stat(heat_map[:, :, i])
                Functions.image_show(heat_map[:, :, i])
        view += 1


def stage_one_cam_map(file_name, show=False, version=2, direction_list=None, feature='bottle'):

    if direction_list is None:
        direction_list = ['Z']
    print("processing", file_name)
    array = np.load('/home/zhoul0a/Desktop/Breast_Cancer_MICCAI/new/rescaled_array_hayida/' + file_name)[
        'array']

    array_info["enhanced_channel"] = 2

    rescaled_ct = array[:, :, :, 0: 3]

    gt_array = array[:, :, :, 3]

    locations = np.where(gt_array > 0.5)
    tumor_center = [int(np.median(locations[0])), int(np.median(locations[1])), int(np.median(locations[2]))]

    view = -1
    for direction in direction_list:
        print(direction)
        if direction == 'X':
            view = 0
        if direction == 'Y':
            view = 1
        if direction == 'Z':
            view = 2
        assert view >= 0
        check_point_path = '/home/zhoul0a/Desktop/Breast_Cancer_MICCAI/new/check_points/hayida/stage_one/' + direction + \
                           '/' + str(int(file_name.split('_')[0][-1]) % 5) + '_saved_model.pth'

        model = load_model(check_point_path, array_info, feature_type=feature, gpu=False)

        sample = slice_one_sample(rescaled_ct, array_info['resolution'], tumor_center[view], direction,
                                    array_info['data_channel'],
                                    array_info['enhanced_channel'], array_info['window'])
        if view == 0:
            predict, heat_map = heat_map_cam(sample, gt_array[tumor_center[view], :, :], model, 1, version=version)
        elif view == 1:
            predict, heat_map = heat_map_cam(sample, gt_array[:, tumor_center[view], :], model, 1, version=version)
        else:
            predict, heat_map = heat_map_cam(sample, gt_array[:, :, tumor_center[view]], model, 1, version=version)

        Functions.save_np_array('/home/zhoul0a/Desktop/Breast_Cancer_MICCAI/new/visualization/cam_maps/stage_one_' + feature + '_version' + str(version),
                                file_name[:-4] + '_sample_' + direction, sample, compress=False)
        Functions.save_np_array('/home/zhoul0a/Desktop/Breast_Cancer_MICCAI/new/visualization/cam_maps/stage_one_' + feature + '_version' + str(version),
                                file_name[:-4] + '_heatmap_' + direction, heat_map, compress=False)

        if show:
            print(np.shape(sample), np.shape(predict), np.shape(heat_map))

            Functions.merge_image_with_mask(sample[:, :, 1], np.array(predict[:, :] > 0.15, 'float32'))

            Functions.array_stat(heat_map)
            Functions.image_show(heat_map)
        view += 1


def stage_two_cam_map(file_name, show=False, version=2, direction_list=None, feature='bottle'):
    if direction_list is None:
        direction_list = ['Z']
    print("processing", file_name)
    array = np.load('/home/zhoul0a/Desktop/Breast_Cancer_MICCAI/new/enhanced_arrays/' + file_name)[
        'array']

    array_info["enhanced_channel"] = 3

    rescaled_ct = array[:, :, :, 0: 4]

    gt_array = array[:, :, :, 4]

    locations = np.where(gt_array > 0.5)
    tumor_center = [int(np.median(locations[0])), int(np.median(locations[1])), int(np.median(locations[2]))]

    view = -1
    for direction in direction_list:
        print(direction)
        if direction == 'X':
            view = 0
        if direction == 'Y':
            view = 1
        if direction == 'Z':
            view = 2
        assert view >= 0
        check_point_path = '/home/zhoul0a/Desktop/Breast_Cancer_MICCAI/new/check_points/hayida/stage_two/' + direction + \
                           '/' + str(int(file_name.split('_')[0][-1]) % 5) + '_saved_model.pth'

        model = load_model(check_point_path, array_info, feature_type=feature, gpu=False)

        sample = slice_one_sample(rescaled_ct, array_info['resolution'], tumor_center[view], direction,
                                  array_info['data_channel'],
                                  array_info['enhanced_channel'], array_info['window'])
        if view == 0:
            predict, heat_map = heat_map_cam(sample, gt_array[tumor_center[view], :, :], model, 1, version=version)
        elif view == 1:
            predict, heat_map = heat_map_cam(sample, gt_array[:, tumor_center[view], :], model, 1, version=version)
        else:
            predict, heat_map = heat_map_cam(sample, gt_array[:, :, tumor_center[view]], model, 1, version=version)

        Functions.save_np_array('/home/zhoul0a/Desktop/Breast_Cancer_MICCAI/new/visualization/cam_maps/stage_two_' + feature + '_version' + str(version),
                                file_name[:-4] + '_sample_' + direction, sample, compress=False)
        Functions.save_np_array('/home/zhoul0a/Desktop/Breast_Cancer_MICCAI/new/visualization/cam_maps/stage_two_' + feature + '_version' + str(version),
                                file_name[:-4] + '_heatmap_' + direction, heat_map, compress=False)

        if show:
            print(np.shape(sample), np.shape(predict), np.shape(heat_map))

            Functions.merge_image_with_mask(sample[:, :, 1], np.array(predict[:, :] > 0.15, 'float32'))

            Functions.array_stat(heat_map)
            Functions.image_show(heat_map)


def get_cam_picture(file_name, stage='one'):
    top_dict = '/home/zhoul0a/Desktop/Breast_Cancer_MICCAI/new/visualization/cam_maps/stage_' + stage + '/'
    if file_name[-4] == '.':
        file_name = file_name[:-4]
    heat_map = np.load(top_dict + file_name + '_heatmap_Z.npy')
    sample = np.load(top_dict + file_name + '_sample_Z.npy')
    print(np.shape(sample))
    for i in range(5):
        Functions.image_show(sample[:, :, i])
    gt = sample[:, :, 4]
    highest_contrast = sample[:, :, 1]
    Functions.merge_image_with_mask(highest_contrast, gt)


fn_list = os.listdir('/home/zhoul0a/Desktop/Breast_Cancer_MICCAI/new/enhanced_arrays/')
processed = 0
for fn in fn_list:
    print(len(fn_list) - processed, 'left')
    if os.path.exists('/home/zhoul0a/Desktop/Breast_Cancer_MICCAI/new/visualization/cam_maps/stage_one/' + fn[:-4] +
                      '_heatmap_Z.npy'):
        processed += 1
        continue
    stage_one_cam_map(fn, show=False, version=4, feature='bottle')
    stage_one_cam_map(fn, show=False, version=4, feature='last_cnn')
    processed += 1
