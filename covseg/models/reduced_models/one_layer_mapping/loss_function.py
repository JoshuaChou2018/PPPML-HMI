import torch
import torch.nn.functional as F


def cross_entropy_pixel_wise_multi_class(prediction, ground_truth, weight_array, balance_weight=(10, 1)):
    """
    all parameters should on GPU, with float32 data type.
    :param balance_weight: balance_weight for class one, two, three, etc
    :param prediction: [batch_size, class_num, height, width], NOT soft_maxed!
    :param ground_truth: [batch_size, class_num, height, width], each pixel with value [0, 1]
    :param weight_array: [batch_size, class_num, height, width], each pixel with value [0, inf)
    :return: freq float with value [0, inf)
    """
    prediction = prediction.float()
    ground_truth = ground_truth.float()

    softmax_then_log = torch.nn.LogSoftmax(dim=1)
    log_prediction_probability = -softmax_then_log(prediction)

    return_tensor = log_prediction_probability * ground_truth * weight_array * 1
    for i in range(len(balance_weight)):
        hyper_weight = balance_weight[i]
        return_tensor[:, i, :, :] = return_tensor[:, i, :, :] * hyper_weight
        loss = torch.sum(return_tensor)

    return loss