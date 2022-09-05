import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

params = {
    "device": None,
    "n_epochs": 3,
    "batch_size": 5,
    "lr": 1e-4,
    "weight_decay": 10,
    "p": 2,
    "leave_one_out_pos": 16,
    "input_dir": None,
    "gt_dir": None,
    "save_model_path": None,
    "save_model_filename": None
}


class Regularization(nn.Module):
    def __init__(self, model, weight_decay, p=2):
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay must be above 0! ")
            exit(0)
        self.model = model
        self.weight_decay = weight_decay
        self.p = p
        self.weight_list = self.get_weight(model)
        self.weight_info(self.weight_list)

    def to(self, device):
        r"""
        cpu or gpu
        """
        self.device = device
        super().to(device)
        return self

    def forward(self, model):
        self.weight_list = self.get_weight(model)
        reg_loss = self.regularization_loss(self.weight_list,
                                            self.weight_decay,
                                            p=self.p)
        return reg_loss

    def get_weight(self, model):
        r"""
        获得权重列表
        """
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self, weight_list, weight_decay, p=2):
        r"""
        正则化附加的张量范数
        """
        reg_loss = 0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg
        reg_loss = weight_decay * reg_loss
        return reg_loss

    def weight_info(self, weight_list):
        print("----------regularization weight----------")
        for name, w in weight_list:
            print(name)
        print("-----------------------------------------")


class SinLayerNet(nn.Module):
    def __init__(self):
        super(SinLayerNet, self).__init__()
        self.fc1 = nn.Linear(24, 9)  # 输入24个指标，输出9项指标

    def forward(self, data):
        data = self.fc1(data)
        return data


def LossByDistance(prediction, gt):
    r"""
    差的平方和表示距离
    """
    # prediction = prediction.float()
    # gt = gt.float()
    prediction = torch.tensor(prediction.float(), requires_grad=True)
    gt = torch.tensor(gt.float(), requires_grad=False)
    loss = torch.sum(torch.pow((prediction - gt), 2))
    return loss
    # dis = 0
    # num_of_patients = np.shape(output)[0]
    # num_of_index = np.shape(output)[1]
    # output = torch.FloatTensor(output)
    # gt = torch.FloatTensor(gt)
    # for i in range(num_of_patients):
    #     for j in range(num_of_index):
    #         dis = dis + (output[i][j] - gt[i][j])**2
    # return dis


def load_train_data(parameter):
    r"""
    分batch：思路是不动原数据，打散坐标，读数据的时候就直接从原数据中取对应的坐标
    """
    from numpy import random
    params = parameter
    input = np.load(params["input_dir"])
    num_of_patients = np.shape(input)[0]
    total_pos = np.arange(num_of_patients)
    train_pos = np.delete(
        total_pos,
        params["leave_one_out_pos"])  # 删掉值为params["leave_one_out_pos"]的元素
    random.shuffle(train_pos)
    train_pos = tuple(train_pos)
    batch = [
        train_pos[i:i + params["batch_size"]]
        for i in range(0, num_of_patients - 1, params["batch_size"])
    ]
    return batch


def train(model, parameters):
    global params
    params = parameters

    model.train()
    if params["weight_decay"] > 0:
        reg_loss = Regularization(model, params["weight_decay"],
                                  params["p"]).to(params["device"])
    else:
        print("no regularization")
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    inputs = np.load(params["input_dir"])
    gts = np.load(params["gt_dir"])
    for e in range(params["n_epochs"]):
        print("epoch:", e)
        batches = load_train_data(params)
        for _, batch in enumerate(batches):
            input = []
            gt = []
            for i in range(params["batch_size"]):
                input.append(inputs[batch[i]])
                gt.append(gts[batch[i]])
            input = torch.Tensor(input)
            gt = torch.Tensor(gt)
            input = input.to(params["device"])
            gt = gt.to(params["device"])
            out = model(input)
            loss = LossByDistance(out, gt).to(params["device"])
            print(loss)
            if params["weight_decay"] > 0:
                loss = loss + reg_loss(model)
                # print("regularized")
            # loss = loss.item()
            print("loss:", float(loss.cpu()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if not os.path.isdir(params["save_model_path"]):
        os.makedirs(params["save_model_path"])
    filename = os.path.join(params["save_model_path"],
                            params["save_model_filename"])
    torch.save(model, filename)  # 保存模型，以便之后用置换检验
    print("finish train")


def test(model, params):
    model.eval()
    test_input = [np.load(params["input_dir"])[params["leave_one_out_pos"]]]
    test_gt = [np.load(params["gt_dir"])[params["leave_one_out_pos"]]]
    with torch.no_grad():
        test_out = model(test_input)
        devia = LossByDistance(test_out, test_gt)
        return devia.item()


def prediction(model, input, params):
    model.eval()
    with torch.no_grad():
        input = torch.Tensor(input)
        input = input.to(params["device"])
        out = model(input)
        out = out.cpu().numpy()
        return out
    # prediction_list = []
    # with torch.no_grad():
    #     for i, sample in enumerate(input):
    #         input = torch.Tensor(input)
    #         input = input.to(params["device"])
    #         out = model(input)
    #         out = out.cpu().numpy()
    #         prediction_list.append(out)
    #     predictions = np.concatenate(prediction_list, axis_list=0)
    #     return predictions