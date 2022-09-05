import cv2
from models.Unet_2D.test import *
import os
from sample_manager.sample_slicer_multi_classes import slice_one_sample

os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'

array_info = {
        "resolution": (1, 1, 1),
        "data_channel": 1,
        "enhanced_channel": 0,
        "window": (-1, 0, 1),
        "positive_semantic_channel": None,  # prediction phase this should be None
        "output_channels": 2,  # output_channels is 2: positive and negative
        "mute_output": True,  # if you want to see prediction details, set is as False
        "wrong_scan": None,
        "init_features": 16
    }


def get_cam_map(rescaled_ct, gt_array, model_top_dict, array_info_spec=None, show=False, version=4,
                direction_list=None, feature='bottle', save_dict=None, patient_id=None):
    # feature can be 'bottle' or 'last_cnn'
    if not len(np.shape(rescaled_ct)) == 4:
        assert len(np.shape(rescaled_ct)) == 3
        new_shape = [np.shape(rescaled_ct)[0], np.shape(rescaled_ct)[1], np.shape(rescaled_ct)[2], 1]
        rescaled_ct = np.reshape(rescaled_ct, new_shape)
    if direction_list is None:
        direction_list = ['Z']

    locations = np.where(gt_array > 0.5)
    semantic_center = [int(np.median(locations[0])), int(np.median(locations[1])), int(np.median(locations[2]))]

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
        if not model_top_dict[-1] == '/':
            model_top_dict = model_top_dict + '/'
        model_names = os.listdir(model_top_dict + direction)
        check_point_path = model_top_dict + direction + '/' + model_names[0]
        if array_info_spec is not None:
            model = load_model(check_point_path, array_info_spec, feature_type=feature, gpu=False)
        else:
            model = load_model(check_point_path, array_info, feature_type=feature, gpu=False)

        if array_info_spec is not None:
            sample = slice_one_sample(rescaled_ct, array_info_spec['resolution'], semantic_center[view], direction,
                                      array_info_spec['data_channel'],
                                      array_info_spec['enhanced_channel'], array_info_spec['window'])
        else:
            sample = slice_one_sample(rescaled_ct, array_info['resolution'], semantic_center[view], direction,
                                      array_info['data_channel'],
                                      array_info['enhanced_channel'], array_info['window'])

        if view == 0:
            predict, heat_map = heat_map_cam(sample, gt_array[semantic_center[view], :, :], model, 1, version=version)
        elif view == 1:
            predict, heat_map = heat_map_cam(sample, gt_array[:, semantic_center[view], :], model, 1, version=version)
        else:
            predict, heat_map = heat_map_cam(sample, gt_array[:, :, semantic_center[view]], model, 1, version=version)

        if not save_dict[-1] == '/':
            save_dict = save_dict + '/'

        if save_dict is not None:
            assert type(patient_id) is str

            Functions.save_np_array(save_dict + feature + '_version' + str(version),
                                    str(patient_id) + '_sample_' + direction, sample, compress=False)
            Functions.save_np_array(save_dict + feature + '_version' + str(version),
                                    str(patient_id) + '_heatmap_' + direction, heat_map, compress=False)

        if show:
            print(np.shape(sample), np.shape(predict), np.shape(heat_map))

            Functions.merge_image_with_mask(sample[:, :, 1], np.array(predict[:, :] > 0.66, 'float32'))

            Functions.array_stat(heat_map)
            Functions.image_show(heat_map)
        view += 1


def get_heat_map(cam_map, target_shape=None):
    # input freq numpy array with shape (freq, b)
    min_value, max_value = np.min(cam_map), np.max(cam_map)
    cam_map = (cam_map - min_value) / (max_value + 0.00001) * 255
    cam_map = np.array(cam_map, 'int32')
    if target_shape is not None:
        assert len(target_shape) == 2

        cam_map = cv2.resize(np.array(cam_map, 'float32'), target_shape)  # must in float to resize
    colored_cam = cv2.normalize(cam_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    colored_cam = cv2.applyColorMap(colored_cam, cv2.COLORMAP_JET)

    return_image = np.zeros(np.shape(colored_cam), 'int32')
    return_image[:, :, 0] = colored_cam[:, :, 2]
    return_image[:, :, 1] = colored_cam[:, :, 1]
    return_image[:, :, 2] = colored_cam[:, :, 0]

    return return_image / 255


def merge_with_heat_map(data_image, cam_map, signal_rescale=False):
    """

    :param signal_rescale: 0-1 rescale of data_image
    :param data_image: freq numpy array with shape (freq, b) or (freq, b, 3)
    :param cam_map: freq numpy array with shape (c, d)
    :return: merged image with shape (freq, b, 3), in float32, min 0 max 1.0
    """
    shape_image = np.shape(data_image)
    if not shape_image == np.shape(cam_map):
        heat_map = get_heat_map(cam_map, target_shape=(shape_image[0], shape_image[1]))
    else:
        heat_map = get_heat_map(cam_map, target_shape=None)
    if signal_rescale:
        min_value, max_value = np.min(data_image), np.max(data_image)
        data_image = (data_image - min_value) / (max_value + 0.00001)
    cam_map = cv2.resize(np.array(cam_map, 'float32'), (shape_image[0], shape_image[1]))  # must in float to resize
    weight_map = cam_map / (np.max(cam_map) + 0.00001)
    weight_map_image = 1 - weight_map
    return_image = np.zeros((shape_image[0], shape_image[1] * 2, 3), 'float32')
    if len(shape_image) == 2:
        return_image[:, 0: shape_image[1], 0] = data_image
        return_image[:, 0: shape_image[1], 1] = data_image
        return_image[:, 0: shape_image[1], 2] = data_image
    else:
        return_image[:, 0: shape_image[1], :] = data_image

    return_image[:, shape_image[1]::, 0] = \
        weight_map_image * return_image[:, 0: shape_image[1], 0] + weight_map * heat_map[:, :, 0]
    return_image[:, shape_image[1]::, 1] = \
        weight_map_image * return_image[:, 0: shape_image[1], 1] + weight_map * heat_map[:, :, 1]
    return_image[:, shape_image[1]::, 2] = \
        weight_map_image * return_image[:, 0: shape_image[1], 2] + weight_map * heat_map[:, :, 2]
    return return_image


if __name__ == '__main__':
    import numpy as np
    import Tool_Functions.Functions as Functions
    data = np.load('/home/zhoul0a/Desktop/vein_artery_identification/heat_cam/stage_one/last_cnn_version4/f031_sample_Z.npy')[:, :, 1]
    Functions.array_stat(data * 1600 - 600)
    data = np.clip(data + 0.5, 0, 1)
    cam_image = np.load('/home/zhoul0a/Desktop/vein_artery_identification/heat_cam/stage_one/last_cnn_version4/f031_heatmap_Z.npy')
    print(np.shape(cam_image))
    merged_image = merge_with_heat_map(data, cam_image)
    Functions.image_show(merged_image)
    exit()
    from prediction.three_way_prediction import get_enhance_channel
    top_dict = '/home/zhoul0a/Desktop/vein_artery_identification/rescaled_ct/f03'
    tail = '_2020-03-10.npy'
    for i in [1, 2, 4, 5]:
        ct = np.load(top_dict + str(i) + tail)

        lungs = np.load('/home/zhoul0a/Desktop/vein_artery_identification/stage_one_prediction_blood/' + 'f03' +
                        str(i) + '_lung.npz')['array']

        blood_vessel_stage_one = np.load('/home/zhoul0a/Desktop/vein_artery_identification/stage_one_prediction_blood/'
                                         + 'f03' + str(i) + '_blood.npy')
        gt = np.array(blood_vessel_stage_one > 0.5, 'float32')
        model_dict = '/home/zhoul0a/Desktop/prognosis_project/check_points/blood_vessel_seg_stage_one'
        save_dict = '/home/zhoul0a/Desktop/vein_artery_identification/heat_cam/stage_one'
        array_info["enhanced_channel"] = 0
        get_cam_map(ct, gt, model_dict, save_dict=save_dict, patient_id='f03' + str(i), feature='last_cnn')

        model_dict = '/home/zhoul0a/Desktop/prognosis_project/check_points/blood_vessel_seg_stage_two'
        save_dict = '/home/zhoul0a/Desktop/vein_artery_identification/heat_cam/stage_two'
        new_input = np.zeros([512, 512, 512, 3], 'float32')
        new_input[:, :, :, 0] = ct
        ratio_high = 0.108  # for high recall enhance channel
        ratio_low = 0.043  # for high precision enhance channel
        enhanced_channel_one, enhanced_channel_two = get_enhance_channel(lungs, blood_vessel_stage_one, ratio_low,
                                                                         ratio_high)
        new_input[:, :, :, 1] = enhanced_channel_one
        new_input[:, :, :, 2] = enhanced_channel_two
        array_info["enhanced_channel"] = 2
        get_cam_map(new_input, gt, model_dict, save_dict=save_dict, patient_id='f03' + str(i), feature='last_cnn')
    exit()
