import torch
import numpy as np
import regularization
from regularization import SinLayerNet, LossByDistance
import os

params = {
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "n_epochs": 10,
    "batch_size": 5,
    "lr": 1e-4,
    "weight_decay": 10,
    "p": 2,
    "leave_one_out_pos": 16,
    "input_dir": r'/home/zhoul0a/Desktop/Lung_CAD_NMI/source_codes/models/reduced_models/input_array.npy',
    "gt_dir": r'/home/zhoul0a/Desktop/Lung_CAD_NMI/source_codes/models/reduced_models/gt_array.npy',
    "val_dir": r'/home/zhoul0a/Desktop/Lung_CAD_NMI/source_codes/models/reduced_models/input_array.npy',
    "val_gt": r'/home/zhoul0a/Desktop/Lung_CAD_NMI/source_codes/models/reduced_models/gt_array.npy',
    "save_model_path": r'/home/zhoul0a/Desktop/prognosis_project/check_points/reduced_models',
    "save_model_filename": r'singlelayermodel.pth',
    "permutation_times": 10000,
    "quantile": 0.05
}


def permutation_test(model, input, ground_truth, params):
    r"""
    置换检验，用训练好的model预测46个病人的数据，计算和ground truth的loss，看随机打乱预测值位置的loss小于预测出来的loss是否满足p<0.05
    """
    print("start permutation test")
    loss_list = []
    num_of_patients = np.shape(ground_truth)[0]
    pred = regularization.prediction(model, input, params)
    pred = torch.Tensor(pred).to(params["device"])
    ground_truth = torch.Tensor(ground_truth).to(params["device"])
    pre_loss = LossByDistance(pred, ground_truth).to(params["device"])
    loss_list.append(pre_loss.item())
    for e in range(params["permutation_times"] - 1):
        if e % 200 == 0:
            print(e, 'in', params["permutation_times"])
        loss = 0
        arr = np.arange(num_of_patients)
        np.random.shuffle(arr)
        for i in range(num_of_patients):
            # rand_pred = torch.Tensor(pred[arr[i]]).to(params["device"])
            # rand_gt = torch.Tensor(ground_truth[i]).to(params["device"])
            # loss = loss + LossByDistance(rand_pred, rand_gt).to(
            #     params["device"]).item()
            loss = loss + LossByDistance(pred[arr[i]], ground_truth[i]).to(
                params["device"]).item()
        loss_list.append(loss)
        # if e % 20 == 0:
        #     print('epoch: {}'.format(e + 1))
    loss_list.sort()
    pos = loss_list.index(pre_loss)
    print('{} is smaller than the original loss.'.format(pos))
    if pos / params["permutation_times"] < params["quantile"]:
        return True
    else:
        return False


def main():
    model = SinLayerNet().to(params["device"])
    regularization.train(model, params)
    model_dir = os.path.join(params["save_model_path"],
                             params["save_model_filename"])
    saved_model = torch.load(model_dir)

    input = np.load(params["input_dir"])
    gt = np.load(params["gt_dir"])
    state = permutation_test(saved_model, input, gt, params=params)
    if state:
        print("pass test!")
    else:
        print("not pass permutation test!")


if __name__ == '__main__':
    main()
    exit()