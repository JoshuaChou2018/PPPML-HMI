#!/usr/bin/env python
import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import importlib
import random
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import copy
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from math import sqrt, exp
from scipy.special import erf
import covseg.models.Unet_2D.U_net_loss_function as unlf
from covseg.models.Unet_2D.dataset import RandomFlipWithWeight, RandomRotateWithWeight, ToTensorWithWeight, \
    WeightedTissueDataset
import covseg.models.Unet_2D.U_net_Model as unm
from copy import deepcopy
import torchvision.transforms
import collections
import time
import tenseal as ts
from torchvision.models import densenet121
import pandas as pd
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging
from Net import DenseNet
from covseg.Tool_Functions import Functions

def log_creater(final_log_file):

    # creat a log
    log = logging.getLogger('train_log')
    log.setLevel(logging.DEBUG)

    # FileHandler
    file = logging.FileHandler(final_log_file,'w')
    file.setLevel(logging.DEBUG)

    # StreamHandler
    stream = logging.StreamHandler()
    stream.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter(
        '[%(asctime)s][line: %(lineno)d] ==> [INFO] %(message)s')

    # setFormatter
    file.setFormatter(formatter)
    stream.setFormatter(formatter)

    # addHandler
    log.addHandler(file)
    log.addHandler(stream)

    log.info('creating {}'.format(final_log_file))
    return log

## For RAD Dataset

my_transform = transforms.Compose([
    transforms.Resize((299,299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class RADDataset(Dataset):
    def __init__(self, x=None, y=None, thickness = -1):
        self.x = x
        self.y = y
        self.thickness = thickness
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        img_path = self.x[index]
        
        if self.thickness!=-1:
            ct_array = np.load(f'my_rescaled_70x/{self.x[index]}.npy')
            ct_array = ct_array.reshape(ct_array.shape[0]//self.thickness, self.thickness, ct_array.shape[1], ct_array.shape[2])
            ct_array = ct_array.mean(axis = 1)
            
            #ct_array = np.array([Functions.rescale_2d_array(ct_array[:,:,i],(70,70)) for i in range(int(ct_array.shape[2]))])
            #ct_array = ct_array.transpose((1,2,0))
            
            new_data = np.zeros(shape=[70,70,70], dtype=np.float32)
            new_data[int((70-ct_array.shape[0])/2):int((70-ct_array.shape[0])/2)+ct_array.shape[0],:,:]=ct_array
            ct_array = new_data
        else:
            ct_array = np.load(f'my_rescaled/{self.x[index]}.npy')
        
        return torch.tensor(ct_array).unsqueeze(0), torch.tensor(self.y[index])
    
def read_user_data_RAD(index, args):
    id = index
    exp = args.exp
    df = pd.read_csv(f'{exp}.csv')
    user_params = deepcopy(args.params)
    check_point_dict = f"{user_params['checkpoint_dir']}/{index}/"
    user_params["checkpoint_dir"] = check_point_dict
    params = deepcopy(user_params)
    
    try:
        if not os.path.isdir(params["checkpoint_dir"]):
            os.system(f'mkdir -p {params["checkpoint_dir"]}')
            logger.info(f'mkdir {params["checkpoint_dir"]}')
    except:
        pass
        
    if True:
        client = str(index)
        X = list(df[df.client==client].NoteAcc_DEID)
        Y = list(df[df.client==client].binary_infection_to_lung_ratio)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=30)
        logger.info(f'client {client}, #Train: {len(Y_train)}, #Test: {len(Y_test)},  #Train posi: {sum(np.array(Y_train)>0)},  #Test posi: {sum(np.array(Y_test)>0)}')
        if client == '0':
            thickness = 1
        elif client == '1':
            thickness = 5
        elif client == '2':
            thickness = 10
        train_dataset = RADDataset(X_train, Y_train, thickness = thickness)
        test_dataset = RADDataset(X_test, Y_test, thickness = thickness)
        
    return id, train_dataset, test_dataset, user_params
    
# load COVID data for each user

def read_user_data_COVID(index, args):
    #index: user index, 对应COVID_client_labels中的label
    label = args.COVID_client_labels[index]
    id = index
    user_params = deepcopy(args.params)
    logger.info(f'>> reading data at client: {label}')

    user_params["test_id"] = args.test_id
    train_dict = f"{args.COVID_data_root}/{label}/training_samples/{args.direction}"
    user_params["train_data_dir"] = train_dict
    user_params["test_data_dir"] = train_dict
    check_point_dict = f"{args.COVID_checkpoint_root}/{label}/{args.direction}"
    user_params["checkpoint_dir"] = check_point_dict
    if user_params["balance_weights"] is None:
        user_params["balance_weights"] = [1000000000, 1]

    flag=False
    #other_ids=deepcopy(args.test_ids)
    other_ids=[0,1,2,3,4]
    other_ids.remove(args.test_id)
    params = deepcopy(user_params)

    while flag!=True and len(other_ids)>=0:
        try:
            if not os.path.isdir(params["checkpoint_dir"]):
                os.makedirs(params["checkpoint_dir"])
        except:
            pass

        train_transform = torchvision.transforms.Compose([
            ToTensorWithWeight(),
            RandomFlipWithWeight(),
            RandomRotateWithWeight()
        ])

        train_transform_no_rotate = torchvision.transforms.Compose([
            ToTensorWithWeight(),
            RandomFlipWithWeight(),
        ])

        test_transform = torchvision.transforms.Compose([
            ToTensorWithWeight()
        ])

        if params["no_rotate"] is True:
            train_dataset = WeightedTissueDataset(
                params["train_data_dir"],
                params["weight_dir"],
                transform=train_transform_no_rotate,
                channels=params["channels"],
                mode="train",
                test_id=params["test_id"],
                wrong_patient_id=params["wrong_patient_id"],
                default_weight=params["default_weight"]
            )
        else:
            train_dataset = WeightedTissueDataset(
                params["train_data_dir"],
                params["weight_dir"],
                transform=train_transform,
                channels=params["channels"],
                mode="train",
                test_id=params["test_id"],
                wrong_patient_id=params["wrong_patient_id"],
                default_weight=params["default_weight"]
            )

        test_dataset = WeightedTissueDataset(
            params["train_data_dir"],
            params["weight_dir"],
            transform=test_transform,
            channels=params["channels"],
            mode='test',
            test_id=params["test_id"],
            wrong_patient_id=params["wrong_patient_id"],
            default_weight=params["default_weight"]
        )

        logger.info("train:", params["train_data_dir"], len(train_dataset))
        logger.info("test:", params["test_data_dir"], len(test_dataset))

        if len(test_dataset)>0:
            flag=True
        else:
            old_id = params["test_id"]
            params["test_id"]=other_ids[0]
            other_ids.remove(params["test_id"])
            logger.info(f'>> Because test id: {old_id} has no data in this client, so we change the test id to: {params["test_id"]}, remaining choice: {other_ids}')
    return id, train_dataset, test_dataset, user_params

# User

class User:
    """
    Base class for users in federated learning.
    """

    def __init__(self, device, id, train_data, test_data, model, batch_size=0, learning_rate=0, beta=0, lamda=0,
                 local_epochs=0,user_params=None):

        self.device = device
        self.model = copy.deepcopy(model)
        self.id = id  # integer
        self.train_samples = len(train_data)
        self.test_samples = len(test_data)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta = beta
        self.lamda = lamda
        self.local_epochs = local_epochs
        self.trainloader = DataLoader(train_data, self.batch_size,shuffle=True)
        self.testloader = DataLoader(test_data, self.batch_size,shuffle=False)
        self.testloaderfull = DataLoader(test_data, self.test_samples)
        self.trainloaderfull = DataLoader(train_data, self.train_samples)
        self.iter_trainloader = iter(self.trainloader)
        self.iter_testloader = iter(self.testloader)
        self.user_params=user_params

        # those parameters are for persionalized federated learing.
        self.local_model = copy.deepcopy(list(self.model.parameters()))
        self.persionalized_model = copy.deepcopy(list(self.model.parameters()))
        self.persionalized_model_bar = copy.deepcopy(list(self.model.parameters()))

    def set_parameters(self, model):
        for old_param, new_param, local_param in zip(self.model.parameters(), model.parameters(), self.local_model):
            old_param.data = new_param.data.clone()
            local_param.data = new_param.data.clone()
        # self.local_weight_updated = copy.deepcopy(self.optimizer.param_groups[0]['params'])

    def get_parameters(self):
        for param in self.model.parameters():
            param.detach()
        return self.model.parameters()

    def clone_model_paramenter(self, param, clone_param):
        for param, clone_param in zip(param, clone_param):
            clone_param.data = param.data.clone()
        return clone_param

    def get_updated_parameters(self):
        return self.local_weight_updated

    def update_parameters(self, new_params):
        for param, new_param in zip(self.model.parameters(), new_params):
            param.data = new_param.data.clone()

    def get_grads(self):
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad.data)
        return grads

    def train_COVID(self):
        self.model.train()
        logger.info(f'--------------- Local train on client: {args.COVID_client_labels[self.numeric_id]} ---------------')
        params = self.user_params
        params[
            "saved_model_filename"] = f"rep{args.current_repeat_experiment_id}_globalepoch{args.current_global_epoch_id}_testid{str(args.test_id)}_saved_model.pth"
        resume = True
        if resume and params["best_f1"] is not None:  # direct give freq best_f1
            best_f1 = params["best_f1"]
        else:
            best_f1 = 0
        saved_model_path = os.path.join(params["checkpoint_dir"], 'current_' + params["saved_model_filename"])
        if resume and os.path.isfile(saved_model_path):
            logger.info(">> load saved model")
            data_dict = torch.load(saved_model_path)
            epoch_start = data_dict["epoch"]
            if type(self.model) == nn.DataParallel:
                self.model.module.load_state_dict(data_dict["state_dict"])
            else:
                self.model.load_state_dict(data_dict["state_dict"])
            self.optimizer.load_state_dict(data_dict["optimizer"])
            history = data_dict["history"]
            best_f1 = data_dict["best_f1"]
            self.phase_info = data_dict["phase_info"]
            logger.info("best_f1 is:", best_f1)
        else:  # this means we do not have freq checkpoint
            epoch_start = 0
            history = collections.defaultdict(list)
            self.phase_info = None
        logger.info("Going to train epochs [%d-%d]" % (epoch_start + 1, epoch_start + args.local_epochs))

        if self.phase_info is None:
            base = 1
            self.precision_phase = True  # if in precision phase, the model increase the precision
            flip_remaining = params[
                "flip_remaining:"]  # initially, model has high recall low precision. Thus, we initially
            # in the precision phase. When precision is high and recall is low, we flip to recall phase.
            # flip_remaining is the number times flip precision phase to recall phase

            self.fluctuate_phase = False
            # when flip_remaining is 0 and the model reached target_performance during recall phase, change to fluctuate
            #  phase during this phase, the recall fluctuate around the target_performance.

            fluctuate_epoch = 0

            current_phase = "precision"
        else:
            current_phase, base, flip_remaining, fluctuate_epoch = self.phase_info
            if current_phase == 'precision':
                self.precision_phase = True
            else:
                self.precision_phase = False
            if current_phase == 'fluctuate':
                self.fluctuate_phase = True
            else:
                self.fluctuate_phase = False
        logger.info("current_phase, base, flip_remaining, fluctuate_epoch:", current_phase, base, flip_remaining,
              fluctuate_epoch)

        previous_recall = 0
        precision_to_recall_count = 4

        for epoch in range(epoch_start + 1, epoch_start + 1 + args.local_epochs):
            logger.info(
                f"--------------- Local train on client: {args.COVID_client_labels[self.numeric_id]}, Local training epoch: {epoch} ---------------")
            logger.info("fluctuate_epoch:", fluctuate_epoch)
            if fluctuate_epoch > 50:
                break
            if precision_to_recall_count < 0:
                break
            self.model.train()
            for i, sample in enumerate(self.trainloader):
                current_batch_size, input_channel, width, height = sample["image"].shape
                image = sample["image"].to(self.device).float()
                label = torch.zeros([current_batch_size, 2, width, height])
                label[:, 0:1, :, :] = 1 - sample["label"]
                label[:, 1::, :, :] = sample["label"]
                label = label.to(self.device).float()
                weight = sample["weight"].to(self.device).float()
                pred = self.model(image)
                maximum_balance = params["balance_weights"][0]
                hyper_balance = base
                if hyper_balance > maximum_balance:
                    hyper_balance = maximum_balance
                if args.algorithm=='FedAvg':
                    loss = unlf.cross_entropy_pixel_wise_multi_class(pred, label, weight,
                                                                     (hyper_balance, 100 / hyper_balance))
                    # loss = unlf.cross_entropy_pixel_wise_2d_binary_with_pixel_weight(pred, sample["label"].to(params["device"]), weight)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    if i % 10 == 0:
                        logger.info("\tstep [%d/%d], loss=%.4f" % (i + 1, len(self.trainloader), loss), base,
                              (hyper_balance, 100 / hyper_balance))
                elif args.algorithm=='PerAvg' or args.algorithm == 'CSAPerAvg':
                    temp_model = copy.deepcopy(list(self.model.parameters()))
                    loss = unlf.cross_entropy_pixel_wise_multi_class(pred, label, weight,
                                                                     (hyper_balance, 100 / hyper_balance))
                    # loss = unlf.cross_entropy_pixel_wise_2d_binary_with_pixel_weight(pred, sample["label"].to(params["device"]), weight)
                    if i%2==0:
                        # step 1
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                    else:
                        tmp_optimizer=torch.optim.Adam(self.model.parameters(),lr=self.beta)
                        tmp_optimizer.zero_grad()
                        loss.backward()
                        # step 2
                        # restore the model parameters to the one before first update
                        for old_p, new_p in zip(self.model.parameters(), temp_model):
                            old_p.data = new_p.data.clone()
                        tmp_optimizer.step()
                        #self.optimizer.step(beta=self.beta)
                        # clone model to user model
                        self.clone_model_paramenter(self.model.parameters(), self.local_model)
                    if i % 10 == 0:
                        logger.info("\tstep [%d/%d], loss=%.4f" % (i + 1, len(self.trainloader), loss), base,
                              (hyper_balance, 100 / hyper_balance))

            logger.info("\tEvaluating")

            eval_vals_train = evaluate_COVID(self.model, self.testloader, params, self.device)

            logger.info("flip_remaining:", flip_remaining, "self.precision_phase:", self.precision_phase)

            if epoch >= 0 and not self.fluctuate_phase:  # recall will be very very high, start with precision phase

                    if eval_vals_train["precision"] < params["target_performance"] and self.precision_phase:
                        logger.info("precision phase, increase base to", base * 1.13)
                        base = base * 1.13
                    elif self.precision_phase:
                        self.precision_phase = False
                        logger.info("change to recall phase")

                    if eval_vals_train["recall"] < params["baseline_performance_recall"] and self.precision_phase:
                        logger.info("the recall is too low when we try to improve precision")
                        self.precision_phase = False
                        logger.info("change to recall phase")

                    if eval_vals_train["recall"] < params["target_performance"] and not self.precision_phase:
                        logger.info("recall phase, decrease base to", base / 1.15)
                        base = base / 1.15

                    elif not self.precision_phase:
                        if flip_remaining > 0:
                            flip_remaining -= 1
                            self.precision_phase = True
                            logger.info("change to precision phase")
                        else:
                            logger.info("change to fluctuate phase")
                            self.fluctuate_phase = True

            if self.fluctuate_phase:
                if eval_vals_train['recall'] > params["target_performance"] > previous_recall:
                    precision_to_recall_count -= 1
                    logger.info("precision to recall count:", precision_to_recall_count)
                if eval_vals_train["recall"] < params["target_performance"]:
                    logger.info("fluctuate phase, decrease base to", base / 1.025)
                    base = base / 1.025
                else:
                    logger.info("fluctuate phase, increase base to", base * 1.024)
                    base = base * 1.024
                fluctuate_epoch += 1
                previous_recall = eval_vals_train['recall']

            if self.precision_phase:
                current_phase = 'precision'
            elif self.fluctuate_phase:
                current_phase = 'fluctuate'
            else:
                current_phase = 'recall'

            self.phase_info = [current_phase, base, flip_remaining, fluctuate_epoch]

            for k, v in eval_vals_train.items():
                history[k + "_train"].append(v)
            logger.info("\tloss=%.4f, precision=%.4f, recall=%.4f, f1=%.4f"
                  % (
                      eval_vals_train["loss"], eval_vals_train["precision"], eval_vals_train["recall"],
                      eval_vals_train["f1"]))
            if eval_vals_train["f1"] > best_f1 and eval_vals_train["recall"] > params["target_performance"] - 0.01:
                logger.info(">> saving local model")
                best_f1 = eval_vals_train["f1"]
                save_checkpoint(epoch, self.model, self.optimizer, history, best_f1, params, phase_info=self.phase_info)
            save_checkpoint(epoch, self.model, self.optimizer, history, eval_vals_train["f1"], params, False,
                            phase_info=self.phase_info)
        logger.info("Training finished")
        logger.info("best_f1:", best_f1)

    def train_RAD(self):
        self.model.train()
        logger.info(f'--------------- Local train on client: {self.numeric_id} ---------------')
        params = self.user_params
        params[
            "saved_model_filename"] = f"rep{args.current_repeat_experiment_id}_globalepoch{args.current_global_epoch_id}_saved_model.pth"

        loss_criterion = torch.nn.CrossEntropyLoss()
        best_acc = 0
        for epoch in range(args.local_epochs):
            logger.info(
                f"--------------- Local train on client: {self.numeric_id}, Local training epoch: {epoch} ---------------")
            self.model.train()
            i=0
            for data, target in tqdm(self.trainloader):
                data = data.to(device=self.device)
                target = target.to(device=self.device)
                score = self.model(data)
                #logger.info(score,target)
                if args.algorithm == 'FedAvg':
                    loss = loss_criterion(score, target)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    if i % 10 == 0:
                        logger.info("\tstep [%d/%d], loss=%.4f" % (i + 1, len(self.trainloader), loss))
                elif args.algorithm == 'PerAvg' or args.algorithm == 'CSAPerAvg':
                    temp_model = copy.deepcopy(list(self.model.parameters()))
                    loss = loss_criterion(score, target)
                    if i % 2 == 0:
                        # step 1
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                    else:
                        tmp_optimizer=torch.optim.Adam(self.model.parameters(),lr=self.beta)
                        tmp_optimizer.zero_grad()
                        loss.backward()
                        # step 2
                        # restore the model parameters to the one before first update
                        for old_p, new_p in zip(self.model.parameters(), temp_model):
                            old_p.data = new_p.data.clone()
                        tmp_optimizer.step()
                        # self.optimizer.step(beta=self.beta)
                        # clone model to user model
                        self.clone_model_paramenter(self.model.parameters(), self.local_model)
                    if i % 10 == 0:
                        logger.info("\tstep [%d/%d], loss=%.4f" % (i + 1, len(self.trainloader), loss))
                i+=1

            logger.info("\tEvaluating")

            eval_vals_train = evaluate_RAD(self.model, self.testloader, params, self.device)

            logger.info("\tloss=%.4f"% (eval_vals_train["acc"]))
            if eval_vals_train["acc"] > best_acc:
                logger.info(">> saving local model")
                best_acc = eval_vals_train["acc"]
                save_checkpoint(epoch, self.model, self.optimizer, None, best_acc, params, phase_info=None)
            save_checkpoint(epoch, self.model, self.optimizer, None, eval_vals_train["acc"], params, False,
                            phase_info=None)
        logger.info("Training finished")
        logger.info(f"best_acc: {best_acc}")
        
    def train_COVID_onestep(self):
        self.model.train()
        #logger.info(f'--------------- Local train on client: {args.COVID_client_labels[self.numeric_id]} ---------------')
        params = self.user_params
        params[
            "saved_model_filename"] = f"rep{args.current_repeat_experiment_id}_globalepoch{args.current_global_epoch_id}_testid{str(args.test_id)}_onestepfinetune_saved_model.pth"
        resume = True
        if resume and params["best_f1"] is not None:  # direct give freq best_f1
            best_f1 = params["best_f1"]
        else:
            best_f1 = 0
        saved_model_path = os.path.join(params["checkpoint_dir"], 'current_' + params["saved_model_filename"])
        if resume and os.path.isfile(saved_model_path):
            #logger.info(">> load saved model")
            data_dict = torch.load(saved_model_path)
            epoch_start = data_dict["epoch"]
            if type(self.model) == nn.DataParallel:
                self.model.module.load_state_dict(data_dict["state_dict"])
            else:
                self.model.load_state_dict(data_dict["state_dict"])
            self.optimizer.load_state_dict(data_dict["optimizer"])
            history = data_dict["history"]
            best_f1 = data_dict["best_f1"]
            self.phase_info = data_dict["phase_info"]
            #logger.info("best_f1 is:", best_f1)
        else:  # this means we do not have freq checkpoint
            epoch_start = 0
            history = collections.defaultdict(list)
            self.phase_info = None
        #logger.info("Going to train epochs [%d-%d]" % (epoch_start + 1, epoch_start + params["n_epochs"]))

        if self.phase_info is None:
            base = 1
            self.precision_phase = True  # if in precision phase, the model increase the precision
            flip_remaining = params[
                "flip_remaining:"]  # initially, model has high recall low precision. Thus, we initially
            # in the precision phase. When precision is high and recall is low, we flip to recall phase.
            # flip_remaining is the number times flip precision phase to recall phase

            self.fluctuate_phase = False #TODO finetune的时候让模型直接进去fluctuate phase
            # when flip_remaining is 0 and the model reached target_performance during recall phase, change to fluctuate
            #  phase during this phase, the recall fluctuate around the target_performance.

            fluctuate_epoch = 0

            current_phase = "precision"
        else:
            current_phase, base, flip_remaining, fluctuate_epoch = self.phase_info
            if current_phase == 'precision':
                self.precision_phase = True
            else:
                self.precision_phase = False
            if current_phase == 'fluctuate':
                self.fluctuate_phase = True
            else:
                self.fluctuate_phase = False
        #logger.info("current_phase, base, flip_remaining, fluctuate_epoch:", current_phase, base, flip_remaining,fluctuate_epoch)

        previous_recall = 0
        precision_to_recall_count = 4

        for epoch in range(epoch_start + 1, epoch_start + 1 + args.onestepupdate):
            #logger.info(f"--------------- Local train on client: {args.COVID_client_labels[self.numeric_id]}, Local training epoch: {epoch} ---------------")
            #logger.info("fluctuate_epoch:", fluctuate_epoch)
            logger.info(f'onestep finetune: {epoch}/{args.onestepupdate} @ {args.COVID_client_labels[self.id]}')
            if fluctuate_epoch > 50:
                break
            if precision_to_recall_count < 0:
                break
            self.model.train()
            for i, sample in enumerate(self.trainloader):
                current_batch_size, input_channel, width, height = sample["image"].shape
                image = sample["image"].to(self.device).float()
                label = torch.zeros([current_batch_size, 2, width, height])
                label[:, 0:1, :, :] = 1 - sample["label"]
                label[:, 1::, :, :] = sample["label"]
                label = label.to(self.device).float()
                weight = sample["weight"].to(self.device).float()
                pred = self.model(image)
                maximum_balance = params["balance_weights"][0]
                hyper_balance = base
                if hyper_balance > maximum_balance:
                    hyper_balance = maximum_balance
                if args.algorithm == 'FedAvg':
                    loss = unlf.cross_entropy_pixel_wise_multi_class(pred, label, weight,
                                                                     (hyper_balance, 100 / hyper_balance))
                    # loss = unlf.cross_entropy_pixel_wise_2d_binary_with_pixel_weight(pred, sample["label"].to(params["device"]), weight)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    if i % 10 == 0:
                        pass
                        #logger.info("\tstep [%d/%d], loss=%.4f" % (i + 1, len(self.trainloader), loss), base,(hyper_balance, 100 / hyper_balance))
                elif args.algorithm == 'PerAvg' or args.algorithm == 'CSAPerAvg':
                    temp_model = copy.deepcopy(list(self.model.parameters()))
                    loss = unlf.cross_entropy_pixel_wise_multi_class(pred, label, weight,
                                                                     (hyper_balance, 100 / hyper_balance))
                    # loss = unlf.cross_entropy_pixel_wise_2d_binary_with_pixel_weight(pred, sample["label"].to(params["device"]), weight)
                    if i % 2 == 0:
                        # step 1
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                    else:
                        tmp_optimizer=torch.optim.Adam(self.model.parameters(),lr=self.beta)
                        tmp_optimizer.zero_grad()
                        loss.backward()
                        # step 2
                        # restore the model parameters to the one before first update
                        for old_p, new_p in zip(self.model.parameters(), temp_model):
                            old_p.data = new_p.data.clone()
                        tmp_optimizer.step()
                        # self.optimizer.step(beta=self.beta)
                        # clone model to user model
                        self.clone_model_paramenter(self.model.parameters(), self.local_model)
                    if i % 10 == 0:
                        pass
                        #logger.info("\tstep [%d/%d], loss=%.4f" % (i + 1, len(self.trainloader), loss), base,(hyper_balance, 100 / hyper_balance))

            #logger.info("\tEvaluating")

            eval_vals_train = evaluate_COVID(self.model, self.testloader, params, self.device)

            #logger.info("flip_remaining:", flip_remaining, "self.precision_phase:", self.precision_phase)

            # 对于finetune来说 30没有作用了
            if epoch >= 0 and not self.fluctuate_phase:  # recall will be very very high, start with precision phase

                if eval_vals_train["precision"] < params["target_performance"] and self.precision_phase:
                    #logger.info("precision phase, increase base to", base * 1.13)
                    base = base * 1.13
                elif self.precision_phase:
                    self.precision_phase = False
                    #logger.info("change to recall phase")

                if eval_vals_train["recall"] < params["baseline_performance_recall"] and self.precision_phase:
                    #logger.info("the recall is too low when we try to improve precision")
                    self.precision_phase = False
                    #logger.info("change to recall phase")

                if eval_vals_train["recall"] < params["target_performance"] and not self.precision_phase:
                    #logger.info("recall phase, decrease base to", base / 1.15)
                    base = base / 1.15

                elif not self.precision_phase:
                    if flip_remaining > 0:
                        flip_remaining -= 1
                        self.precision_phase = True
                        #logger.info("change to precision phase")
                    else:
                        #logger.info("change to fluctuate phase")
                        self.fluctuate_phase = True

            if self.fluctuate_phase:
                if eval_vals_train['recall'] > params["target_performance"] > previous_recall:
                    precision_to_recall_count -= 1
                    #logger.info("precision to recall count:", precision_to_recall_count)
                if eval_vals_train["recall"] < params["target_performance"]:
                    #logger.info("fluctuate phase, decrease base to", base / 1.025)
                    base = base / 1.025
                else:
                    #logger.info("fluctuate phase, increase base to", base * 1.024)
                    base = base * 1.024
                fluctuate_epoch += 1
                previous_recall = eval_vals_train['recall']

            if self.precision_phase:
                current_phase = 'precision'
            elif self.fluctuate_phase:
                current_phase = 'fluctuate'
            else:
                current_phase = 'recall'

            self.phase_info = [current_phase, base, flip_remaining, fluctuate_epoch]

            for k, v in eval_vals_train.items():
                history[k + "_train"].append(v)
            #logger.info("\tloss=%.4f, precision=%.4f, recall=%.4f, f1=%.4f"% (eval_vals_train["loss"], eval_vals_train["precision"], eval_vals_train["recall"],eval_vals_train["f1"]))
            if eval_vals_train["f1"] > best_f1 and eval_vals_train["recall"] > params["target_performance"] - 0.01:
                #logger.info(">> saving local model")
                best_f1 = eval_vals_train["f1"]
                save_checkpoint(epoch, self.model, self.optimizer, history, best_f1, params, phase_info=self.phase_info)
            save_checkpoint(epoch, self.model, self.optimizer, history, eval_vals_train["f1"], params, False,
                            phase_info=self.phase_info)

            logger.info(f'>> onestep finetune precision: {eval_vals_train["precision"]}, recall: {eval_vals_train["recall"]}, f1: {eval_vals_train["f1"]}')
        #logger.info("Training finished")
        #logger.info("best_f1:", best_f1)

    def train_RAD_onestep(self):
        self.model.train()
        params = self.user_params
        params[
            "saved_model_filename"] = f"rep{args.current_repeat_experiment_id}_globalepoch{args.current_global_epoch_id}_onestepfinetune_saved_model.pth"
        loss_criterion = torch.nn.CrossEntropyLoss()
        best_acc = 0
        for epoch in range(args.onestepupdate):
            logger.info(f'onestep finetune: {epoch}/{args.onestepupdate} @ {self.id}')
            self.model.train()
            i=0
            for data, target in tqdm(self.trainloader):
                data = data.to(device=self.device)
                target = target.to(device=self.device)
                score = self.model(data)
                #logger.info(score,target)
                if args.algorithm == 'FedAvg':
                    loss = loss_criterion(score, target)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                elif args.algorithm == 'PerAvg' or args.algorithm == 'CSAPerAvg':
                    temp_model = copy.deepcopy(list(self.model.parameters()))
                    loss = loss_criterion(score, target)
                    if i % 2 == 0:
                        # step 1
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                    else:
                        tmp_optimizer=torch.optim.Adam(self.model.parameters(),lr=self.beta)
                        tmp_optimizer.zero_grad()
                        loss.backward()
                        # step 2
                        # restore the model parameters to the one before first update
                        for old_p, new_p in zip(self.model.parameters(), temp_model):
                            old_p.data = new_p.data.clone()
                        tmp_optimizer.step()
                        # self.optimizer.step(beta=self.beta)
                        # clone model to user model
                        self.clone_model_paramenter(self.model.parameters(), self.local_model)
                i+=1

            eval_vals_train = evaluate_RAD(self.model, self.testloader, params, self.device)
            if eval_vals_train["acc"] > best_acc:
                best_acc = eval_vals_train["acc"]
                save_checkpoint(epoch, self.model, self.optimizer, None, best_acc, params, phase_info=None)
            save_checkpoint(epoch, self.model, self.optimizer, None, eval_vals_train["acc"], params, False,
                            phase_info=None)

            logger.info(f'>> onestep finetune acc: {eval_vals_train["acc"]}')

        
    def test(self):
        self.model.eval()
        test_acc = 0
        #if True:
        with torch.no_grad():
            for x, y in self.testloaderfull:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                # @loss += self.loss(output, y)
                # logger.info(self.id + ", Test Accuracy:", test_acc / y.shape[0] )
                # logger.info(self.id + ", Test Loss:", loss)
        return test_acc, y.shape[0]

    def test_COVID(self):
        self.model.eval()
        count=0
        with torch.no_grad():
            vals = {
                'loss': 0,
                'tp': 0,
                'fp': 0,
                'fn': 0,
                "tot_pixels": 0,
            }
            for i, sample in enumerate(self.testloader):
                current_batch_size, input_channel, width, height = sample["image"].shape
                vals["tot_pixels"] += current_batch_size * width * height
                image = sample["image"].to(self.device).float()
                label = sample["label"].to(self.device).float()
                pred = self.model(image)
                loss = unlf.cross_entropy_pixel_wise_2d_binary(pred, label,
                                                               balance_weights=args.params["balance_weights"]).item()
                # here we use not_voxel_weighted loss to illustrate the performance
                vals["loss"] += loss

                pred = (pred[:, 1, :, :] > pred[:, 0, :, :]).float().unsqueeze(1)
                pred_tp = pred * label
                tp = pred_tp.sum().float().item()
                vals["tp"] += tp
                vals["fp"] += pred.sum().float().item() - tp
                pred_fn = (1 - pred) * label
                vals["fn"] += pred_fn.sum().float().item()
                count+=image.shape[0]
            eps = 1e-6
            vals["loss"] = vals["loss"] / vals["tot_pixels"]

            beta = args.params['beta']  # number times recall is more important than precision

            vals["precision"] = (vals["tp"] + eps) / (vals["tp"] + vals["fp"] + eps)
            vals["recall"] = (vals["tp"] + eps) / (vals["tp"] + vals["fn"] + eps)
            vals["f1"] = (1 + beta * beta) * (vals["precision"] * vals["recall"] + eps) / \
                         (vals["precision"] * (beta * beta) + vals["recall"] + eps)
        return vals, count
    
    def test_RAD(self):
        self.model.eval()
        correct_output = 0
        total_output = 0
        with torch.no_grad():
            vals = {}
            for x, y in tqdm(self.testloader):
                x = x.to(device=self.device)
                y = y.to(device=self.device)
                score = self.model(x)
                _,predictions = score.max(1)
                correct_output += (y==predictions).sum()
                total_output += predictions.shape[0]
            test_acc = float(correct_output.item()/total_output)
            vals['acc'] = test_acc
        return vals, total_output

    def train_error_and_loss(self):
        self.model.eval()
        train_acc = 0
        loss = 0
        #count=0
        #if True:
        with torch.no_grad():
            for x, y in self.trainloaderfull:
                #logger.info(count)
                #count+=1
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                loss += self.loss(output, y)
                # logger.info(self.id + ", Train Accuracy:", train_acc)
                # logger.info(self.id + ", Train Loss:", loss)
        return train_acc, loss, self.train_samples

    def train_error_and_loss_COVID(self):
        self.model.eval()
        #count=0
        #if True:
        count=0
        with torch.no_grad():
            vals = {
                'loss': 0,
                'tp': 0,
                'fp': 0,
                'fn': 0,
                "tot_pixels": 0,
            }
            for i, sample in enumerate(self.trainloader):
                current_batch_size, input_channel, width, height = sample["image"].shape
                vals["tot_pixels"] += current_batch_size * width * height
                image = sample["image"].to(self.device).float()
                label = sample["label"].to(self.device).float()
                pred = self.model(image)
                loss = unlf.cross_entropy_pixel_wise_2d_binary(pred, label,
                                                               balance_weights=args.params["balance_weights"]).item()
                # here we use not_voxel_weighted loss to illustrate the performance
                vals["loss"] += loss

                pred = (pred[:, 1, :, :] > pred[:, 0, :, :]).float().unsqueeze(1)
                pred_tp = pred * label
                tp = pred_tp.sum().float().item()
                vals["tp"] += tp
                vals["fp"] += pred.sum().float().item() - tp
                pred_fn = (1 - pred) * label
                vals["fn"] += pred_fn.sum().float().item()
                count+=image.shape[0]
            eps = 1e-6
            vals["loss"] = vals["loss"] / vals["tot_pixels"]

            beta = args.params['beta']  # number times recall is more important than precision

            vals["precision"] = (vals["tp"] + eps) / (vals["tp"] + vals["fp"] + eps)
            vals["recall"] = (vals["tp"] + eps) / (vals["tp"] + vals["fn"] + eps)
            vals["f1"] = (1 + beta * beta) * (vals["precision"] * vals["recall"] + eps) / \
                         (vals["precision"] * (beta * beta) + vals["recall"] + eps)
        return vals['f1'], vals['loss'], count
    
    def train_error_and_loss_RAD(self):
        self.model.eval()
        #count=0
        #if True:
        count=0
        correct_output = 0
        total_output = 0
        loss_criterion = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            vals = {
                'loss': 0,
            }
            for data, target in tqdm(self.trainloader):
                data = data.to(device=self.device)
                target = target.to(device=self.device)
                score = self.model(data)
                self.optimizer.zero_grad()
                loss = loss_criterion(score, target).item()
                vals["loss"] += loss
                _,predictions = score.max(1)
                correct_output += (target==predictions).sum()
                total_output += predictions.shape[0]
            test_acc = float(correct_output.item()/total_output)
            vals['acc'] = test_acc
            
        return vals['acc'], vals['loss'], total_output

    def test_persionalized_model(self):
        self.model.eval()
        test_acc = 0
        self.update_parameters(self.persionalized_model_bar)
        #if True:
        with torch.no_grad():
            for x, y in self.testloaderfull:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                # @loss += self.loss(output, y)
                # logger.info(self.id + ", Test Accuracy:", test_acc / y.shape[0] )
                # logger.info(self.id + ", Test Loss:", loss)
        self.update_parameters(self.local_model)
        return test_acc, y.shape[0]

    def test_persionalized_model_COVID(self):
        self.model.eval()
        self.update_parameters(self.persionalized_model_bar)
        count = 0
        with torch.no_grad():
            vals = {
                'loss': 0,
                'tp': 0,
                'fp': 0,
                'fn': 0,
                "tot_pixels": 0,
            }
            for i, sample in enumerate(self.testloader):
                current_batch_size, input_channel, width, height = sample["image"].shape
                vals["tot_pixels"] += current_batch_size * width * height
                image = sample["image"].to(self.device).float()
                label = sample["label"].to(self.device).float()
                pred = self.model(image)
                loss = unlf.cross_entropy_pixel_wise_2d_binary(pred, label,
                                                               balance_weights=args.params["balance_weights"]).item()
                # here we use not_voxel_weighted loss to illustrate the performance
                vals["loss"] += loss

                pred = (pred[:, 1, :, :] > pred[:, 0, :, :]).float().unsqueeze(1)
                pred_tp = pred * label
                tp = pred_tp.sum().float().item()
                vals["tp"] += tp
                vals["fp"] += pred.sum().float().item() - tp
                pred_fn = (1 - pred) * label
                vals["fn"] += pred_fn.sum().float().item()
                count += image.shape[0]
            eps = 1e-6
            vals["loss"] = vals["loss"] / vals["tot_pixels"]

            beta = args.params['beta']  # number times recall is more important than precision

            vals["precision"] = (vals["tp"] + eps) / (vals["tp"] + vals["fp"] + eps)
            vals["recall"] = (vals["tp"] + eps) / (vals["tp"] + vals["fn"] + eps)
            vals["f1"] = (1 + beta * beta) * (vals["precision"] * vals["recall"] + eps) / \
                         (vals["precision"] * (beta * beta) + vals["recall"] + eps)
        self.update_parameters(self.local_model)
        return vals, count

    def train_error_and_loss_persionalized_model(self):
        self.model.eval()
        train_acc = 0
        loss = 0
        self.update_parameters(self.persionalized_model_bar)
        #if True:
        with torch.no_grad():
            for x, y in self.trainloaderfull:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                loss += self.loss(output, y)
                # logger.info(self.id + ", Train Accuracy:", train_acc)
                # logger.info(self.id + ", Train Loss:", loss)
        self.update_parameters(self.local_model)
        return train_acc, loss, self.train_samples

    def train_error_and_loss_persionalized_model_COVID(self):
        self.model.eval()
        # count=0
        # if True:
        count = 0
        self.update_parameters(self.persionalized_model_bar)
        with torch.no_grad():
            vals = {
                'loss': 0,
                'tp': 0,
                'fp': 0,
                'fn': 0,
                "tot_pixels": 0,
            }
            for i, sample in enumerate(self.trainloader):
                current_batch_size, input_channel, width, height = sample["image"].shape
                vals["tot_pixels"] += current_batch_size * width * height
                image = sample["image"].to(self.device).float()
                label = sample["label"].to(self.device).float()
                pred = self.model(image)
                loss = unlf.cross_entropy_pixel_wise_2d_binary(pred, label,
                                                               balance_weights=args.params["balance_weights"]).item()
                # here we use not_voxel_weighted loss to illustrate the performance
                vals["loss"] += loss

                pred = (pred[:, 1, :, :] > pred[:, 0, :, :]).float().unsqueeze(1)
                pred_tp = pred * label
                tp = pred_tp.sum().float().item()
                vals["tp"] += tp
                vals["fp"] += pred.sum().float().item() - tp
                pred_fn = (1 - pred) * label
                vals["fn"] += pred_fn.sum().float().item()
                count += image.shape[0]
            eps = 1e-6
            vals["loss"] = vals["loss"] / vals["tot_pixels"]

            beta = args.params['beta']  # number times recall is more important than precision

            vals["precision"] = (vals["tp"] + eps) / (vals["tp"] + vals["fp"] + eps)
            vals["recall"] = (vals["tp"] + eps) / (vals["tp"] + vals["fn"] + eps)
            vals["f1"] = (1 + beta * beta) * (vals["precision"] * vals["recall"] + eps) / \
                         (vals["precision"] * (beta * beta) + vals["recall"] + eps)
        self.update_parameters(self.local_model)
        return vals['f1'], vals['loss'], count

    def get_next_train_batch(self):
        try:
            # Samples a new batch for persionalizing
            (X, y) = next(self.iter_trainloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_trainloader = iter(self.trainloader)
            (X, y) = next(self.iter_trainloader)
        return (X.to(self.device), y.to(self.device))

    def get_next_test_batch(self):
        try:
            # Samples a new batch for persionalizing
            (X, y) = next(self.iter_testloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_testloader = iter(self.testloader)
            (X, y) = next(self.iter_testloader)
        return (X.to(self.device), y.to(self.device))

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "user_" + self.id + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        self.model = torch.load(os.path.join(model_path, "server" + ".pt"))

    def load_best_model(self):
        try:
            filename = self.user_params["saved_model_filename"]
            filename = os.path.join(self.user_params["checkpoint_dir"], filename)
            model_dict = torch.load(filename)['state_dict']
            self.model.load_state_dict(model_dict)
        except:
            filename = 'current_' + self.user_params["saved_model_filename"]
            filename = os.path.join(self.user_params["checkpoint_dir"], filename)
            model_dict = torch.load(filename)['state_dict']
            self.model.load_state_dict(model_dict)

    def load_best_model_onestepfinetune(self):
        try:
            filename = self.user_params["saved_model_filename"]
            filename = os.path.join(self.user_params["checkpoint_dir"], filename)
            model_dict = torch.load(filename)['state_dict']
            self.model.load_state_dict(model_dict)
        except:
            filename = 'current_' + self.user_params["saved_model_filename"]
            filename = os.path.join(self.user_params["checkpoint_dir"], filename)
            model_dict = torch.load(filename)['state_dict']
            self.model.load_state_dict(model_dict)

    @staticmethod
    def model_exists():
        return os.path.exists(os.path.join("models", "server" + ".pt"))

def save_checkpoint(epoch, model, optimizer, history, best_f1, params=None, best=True, phase_info=None):
    # self.phase_info = [current_phase, current_base, flip_remaining, fluctuate_epoch],
    # current_phase in {"recall", "precision", "fluctuate"}
    filename = params["saved_model_filename"]
    if not best:  # this means we store the current model
        filename = "current_" + params["saved_model_filename"]
    filename = os.path.join(params["checkpoint_dir"], filename)
    torch.save({
        'epoch': epoch,
        'state_dict': model.module.state_dict() if type(model) == nn.DataParallel else model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'history': history,
        'best_f1': best_f1,
        'phase_info': phase_info,
    }, filename)
    
def save_checkpoint_server(model,args):
    # self.phase_info = [current_phase, current_base, flip_remaining, fluctuate_epoch],
    # current_phase in {"recall", "precision", "fluctuate"}
    if args.dataset == 'COVID':
        filedir = os.path.join(args.params["checkpoint_dir"], f'server/{args.direction}')
        if os.path.isdir(filedir):
            pass
        else:
            os.system("mkdir -p {}".format(filedir))
        filename = os.path.join(filedir, f"rep{args.current_repeat_experiment_id}_globalepoch{args.current_global_epoch_id}_testid{str(args.test_id)}_saved_model.pth")
        torch.save({
            'state_dict': model.module.state_dict() if type(model) == nn.DataParallel else model.state_dict(),
        }, filename)
    elif args.dataset == 'RAD':
        filedir = os.path.join(args.params["checkpoint_dir"], 'server')
        if os.path.isdir(filedir):
            pass
        else:
            os.system("mkdir -p {}".format(filedir))
        filename = os.path.join(filedir, f"rep{args.current_repeat_experiment_id}_globalepoch{args.current_global_epoch_id}_saved_model.pth")
        torch.save({
            'state_dict': model.module.state_dict() if type(model) == nn.DataParallel else model.state_dict(),
        }, filename)

def evaluate_COVID(model, test_loader, params, device):
    model.eval()
    with torch.no_grad():
        vals = {
            'loss': 0,
            'tp': 0,
            'fp': 0,
            'fn': 0,
            "tot_pixels": 0,
        }
        for i, sample in enumerate(test_loader):
            current_batch_size, input_channel, width, height = sample["image"].shape
            vals["tot_pixels"] += current_batch_size * width * height
            image = sample["image"].to(device).float()
            label = sample["label"].to(device).float()
            pred = model(image)
            loss = unlf.cross_entropy_pixel_wise_2d_binary(pred, label,
                                                           balance_weights=params["balance_weights"]).item()
            # here we use not_voxel_weighted loss to illustrate the performance
            vals["loss"] += loss

            pred = (pred[:, 1, :, :] > pred[:, 0, :, :]).float().unsqueeze(1)
            pred_tp = pred * label
            tp = pred_tp.sum().float().item()
            vals["tp"] += tp
            vals["fp"] += pred.sum().float().item() - tp
            pred_fn = (1 - pred) * label
            vals["fn"] += pred_fn.sum().float().item()
        eps = 1e-6
        vals["loss"] = vals["loss"] / vals["tot_pixels"]

        beta = params['beta']  # number times recall is more important than precision

        vals["precision"] = (vals["tp"] + eps) / (vals["tp"] + vals["fp"] + eps)
        vals["recall"] = (vals["tp"] + eps) / (vals["tp"] + vals["fn"] + eps)
        vals["f1"] = (1 + beta * beta) * (vals["precision"] * vals["recall"] + eps) / \
                     (vals["precision"] * (beta * beta) + vals["recall"] + eps)

        if vals["tp"] + vals["fp"] < 10 * eps or vals["tp"] + vals["fn"] < 10 * eps or vals["precision"] + vals[
            "recall"] < 10 * eps:
            logger.info("Possibly incorrect precision, recall or f1 values")
        return vals

def evaluate_RAD(model, test_loader, params, device):
    model.eval()
    correct_output = 0
    total_output = 0
    with torch.no_grad():
        vals = {
            'loss': 0,
            'tp': 0,
            'fp': 0,
            'fn': 0,
            "tot_pixels": 0,
        }
        for x, y in tqdm(test_loader):
            x = x.to(device=device)
            y = y.to(device=device)
            score = model(x)
            _,predictions = score.max(1)
            correct_output += (y==predictions).sum()
            total_output += predictions.shape[0]
        test_acc = float(correct_output.item()/total_output)
        vals['acc'] = test_acc
        return vals

class UserAVG(User):
    def __init__(self, device, numeric_id, train_data, test_data, model, batch_size, learning_rate, beta, lamda,
                 local_epochs, optimizer, user_params=None):
        super().__init__(device, numeric_id, train_data, test_data, model, batch_size, learning_rate, beta, lamda,
                         local_epochs,user_params)

        self.loss = nn.NLLLoss()

        if args.optimizer=='SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        elif args.optimizer=='Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.user_params = user_params
        self.numeric_id = numeric_id

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def train(self,local_epochs=None):

        if args.dataset=='COVID':
            self.train_COVID()
            
        elif args.dataset=='RAD':
            self.train_RAD()

        elif args.dataset=='Mnist':
            LOSS = 0
            self.model.train()
            for epoch in range(1, self.local_epochs + 1):
                self.model.train()
                X, y = self.get_next_train_batch()
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()
                self.clone_model_paramenter(self.model.parameters(), self.local_model)
        #return LOSS

    def train_one_step_COVID(self):
        self.train_COVID_onestep()
        
    def train_one_step_RAD(self):
        self.train_RAD_onestep()

class UserPerAvg(User):
    def __init__(self, device, numeric_id, train_data, test_data, model, batch_size, learning_rate, beta, lamda,
                 local_epochs, optimizer, total_users, num_users, user_params=None):
        super().__init__(device, numeric_id, train_data, test_data, model, batch_size, learning_rate, beta, lamda,
                         local_epochs,user_params)
        self.total_users = total_users
        self.num_users = num_users
        self.user_params = user_params
        self.numeric_id = numeric_id

        self.loss = nn.NLLLoss()

        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        if args.optimizer=='SGD':
            self.optimizer = MySGD(self.model.parameters(), lr=self.learning_rate)
        elif args.optimizer=='Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def train(self, epochs):

        if args.dataset=='COVID':
            self.train_COVID()
        elif args.dataset=='RAD':
            self.train_RAD()
        elif args.dataset == 'Mnist':
            LOSS = 0
            self.model.train()
            for epoch in range(1, self.local_epochs + 1):  # local update
                self.model.train()

                temp_model = copy.deepcopy(list(self.model.parameters()))

                # step 1
                X, y = self.get_next_train_batch()
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()

                # step 2
                X, y = self.get_next_train_batch()
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss(output, y)
                loss.backward()

                # restore the model parameters to the one before first update
                for old_p, new_p in zip(self.model.parameters(), temp_model):
                    old_p.data = new_p.data.clone()

                #self.optimizer.step(beta=self.beta)
                self.optimizer.step()

                # clone model to user model
                self.clone_model_paramenter(self.model.parameters(), self.local_model)
        #return LOSS

    def train_one_step(self):
        self.model.train()
        # step 1
        X, y = self.get_next_test_batch()
        self.optimizer.zero_grad()
        output = self.model(X)
        loss = self.loss(output, y)
        loss.backward()
        self.optimizer.step()
        # step 2
        X, y = self.get_next_test_batch()
        self.optimizer.zero_grad()
        output = self.model(X)
        loss = self.loss(output, y)
        loss.backward()
        #self.optimizer.step(beta=self.beta)
        self.optimizer.step()

    def train_one_step_COVID(self):
        self.train_COVID_onestep()
        
    def train_one_step_RAD(self):
        self.train_RAD_onestep()

# Server

class Server:
    def __init__(self, device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda,
                 num_glob_iters, local_epochs, optimizer, num_users, times):

        # Set up the main attributes
        self.device = device
        self.dataset = dataset
        self.num_glob_iters = num_glob_iters
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.total_train_samples = 0
        self.model = copy.deepcopy(model)
        self.users = []
        self.selected_users = []
        self.num_users = num_users
        self.beta = beta
        self.lamda = lamda
        self.algorithm = algorithm
        self.rs_train_acc, self.rs_train_loss, self.rs_glob_acc, self.rs_train_acc_per, self.rs_train_loss_per, self.rs_glob_acc_per = [], [], [], [], [], []
        self.times = times
        # 每个参数都对应一个梯度
        self.cumulative_gradients=[]
        # Initialize the server's grads to zeros
        # for param in self.model.parameters():
        #    param.data = torch.zeros_like(param.data)
        #    param.grad = torch.zeros_like(param.data)
        # self.send_parameters()

    def aggregate_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param.grad = torch.zeros_like(param.data)
        for user in self.users:
            self.add_grad(user, user.train_samples / self.total_train_samples)

    def add_grad(self, user, ratio):
        user_grad = user.get_grads()
        for idx, param in enumerate(self.model.parameters()):
            param.grad = param.grad + user_grad[idx].clone() * ratio

    def send_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for user in self.users:
            user.set_parameters(self.model)

    # 传递模型参数

    def add_parameters(self, user, ratio):
        model = self.model.parameters()
        for server_param, user_param in zip(self.model.parameters(), user.get_parameters()):
            server_param.data = server_param.data + user_param.data.clone() * ratio

    def aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data) # 已经把server模型的参数清0
        total_train = 0
        # if(self.num_users = self.to)
        for user in self.selected_users:
            total_train += user.train_samples
        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train)

    # 传递梯度，然后用合并之后的梯度共享server model

    def compute_gradients(self, server_model, user_model):
        return [(new_param.data - old_param.data) / (-self.learning_rate) for old_param, new_param in
                zip(server_model.parameters(), user_model.get_parameters())]

    def compute_gradients_nolr(self, server_model, user_model):
        return [(new_param.data - old_param.data) for old_param, new_param in
                zip(server_model.parameters(), user_model.get_parameters())]

    def add_parameters_with_gradients(self, user, ratio):
        gradients = self.compute_gradients(self.model, user)
        try:
            self.cumulative_gradients=[self.cumulative_gradients[i] + gradients[i]*ratio for i in range(len(gradients))]
        except:
            self.cumulative_gradients=[grad*ratio for grad in gradients]

    def add_parameters_with_gradients_CSA(self, user, ratio, big_R_mask = None, HE = False):
        gradients = self.compute_gradients(self.model, user)
        if HE == False:
            if big_R_mask == None:
                try:
                    self.cumulative_gradients=[self.cumulative_gradients[i] + gradients[i]*ratio for i in range(len(gradients))]
                except:
                    self.cumulative_gradients=[grad*ratio for grad in gradients]
            else: # big_R != None
                try:
                    self.cumulative_gradients=[self.cumulative_gradients[i] + gradients[i]*ratio + big_R_mask[i] for i in range(len(gradients))]
                except:
                    self.cumulative_gradients=[gradients[i]*ratio + big_R_mask[i] for i in range(len(gradients))]
        elif HE == True:
            if big_R_mask == None:
                try:
                    self.cumulative_gradients=[self.cumulative_gradients[i] + ts.ckks_vector(context, (gradients[i]*ratio).flatten().cpu()) for i in range(len(gradients))]
                except:
                    self.cumulative_gradients=[ts.ckks_vector(context, (grad*ratio).flatten().cpu()) for grad in gradients]
            else: # big_R != None
                try:
                    self.cumulative_gradients=[self.cumulative_gradients[i] + ts.ckks_vector(context, (gradients[i]*ratio + big_R_mask[i]).flatten().cpu()) for i in range(len(gradients))]
                except:
                    self.cumulative_gradients=[ts.ckks_vector(context, (gradients[i]*ratio + big_R_mask[i]).flatten().cpu()) for i in range(len(gradients))]

    def add_parameters_with_gradients_nolr(self, user, ratio):
        gradients = self.compute_gradients_nolr(self.model, user)
        try:
            self.cumulative_gradients=[self.cumulative_gradients[i] + gradients[i]*ratio for i in range(len(gradients))]
        except:
            self.cumulative_gradients=[grad*ratio for grad in gradients]

    def private_add_parameters_with_gradients(self, user, ratio):
        gradients = self.compute_gradients(self.model, user)
        gradients = self.add_noise_any(self.clip_grad_norm_any(gradients, args.l2_norm_clip,norm_type=2),args)
        try:
            self.cumulative_gradients=[self.cumulative_gradients[i] + gradients[i]*ratio for i in range(len(gradients))]
        except:
            self.cumulative_gradients=[grad*ratio for grad in gradients]

    def aggregate_parameters_with_gradients(self):
        assert (self.users is not None and len(self.users) > 0)
        self.cumulative_gradients=[]
        # 如果是用梯度更新server，那么不应该把server模型的参数清0
        #for param in self.model.parameters():
        #    param.data = torch.zeros_like(param.data)
        total_train = 0
        # if(self.num_users = self.to)
        for user in self.selected_users:
            total_train += user.train_samples
        for user in self.selected_users:
            self.add_parameters_with_gradients(user, user.train_samples / total_train)
            #self.add_parameters_with_gradients(user, 1)
        # update param with gradients
        for server_param, param_update in zip(self.model.parameters(), self.cumulative_gradients):
            server_param.data += -1 * self.learning_rate * param_update

    def aggregate_parameters_with_gradients_CSA(self, glob_iter = 0, HE = True):
        assert (self.users is not None and len(self.users) > 0)
        self.cumulative_gradients=[]
        # 如果是用梯度更新server，那么不应该把server模型的参数清0
        #for param in self.model.parameters():
        #    param.data = torch.zeros_like(param.data)
        total_train = 0
        # if(self.num_users = self.to)

        initiator_index = glob_iter % len(self.selected_users)
        self.reordered_selected_users = self.selected_users[initiator_index:len(self.selected_users)] + self.selected_users[0:initiator_index]
        logger.info(f'[+] selected client {args.COVID_client_labels[initiator_index]} as the initiator')

        # calculate big_R
        gradients = self.compute_gradients(self.model, self.reordered_selected_users[0])
        self.sigma = 10000
        self.big_R_mask = [torch.tensor(np.random.normal(0, self.sigma, tensor.shape),device=self.device,dtype=torch.float) for tensor in gradients]
        self.shapes = [tensor.shape for tensor in self.big_R_mask]

        for user in self.reordered_selected_users:
            total_train += user.train_samples
        initiator_flag = True
        for user in self.reordered_selected_users:
            if initiator_flag == True:
                self.add_parameters_with_gradients_CSA(user, user.train_samples / total_train, big_R_mask = self.big_R_mask, HE = HE)
                initiator_flag = False
            else:
                self.add_parameters_with_gradients_CSA(user, user.train_samples / total_train, big_R_mask = None, HE = HE)
            #self.add_parameters_with_gradients(user, 1)

        if HE == False:
            # remove the big_R_mask from the cumulative_gradients
            self.cumulative_gradients = [self.cumulative_gradients[i] - self.big_R_mask[i] for i in range(len(self.cumulative_gradients))]
        elif HE == True:
            self.cumulative_gradients = [self.cumulative_gradients[i] - ts.ckks_vector(context, self.big_R_mask[i].flatten().cpu()) for i in range(len(self.cumulative_gradients))]
            self.cumulative_gradients = [torch.reshape(torch.tensor(self.cumulative_gradients[i].decrypt(), device=self.device),shape=self.shapes[i]) for i in range(len(self.cumulative_gradients))]

        # update param with gradients
        for server_param, param_update in zip(self.model.parameters(), self.cumulative_gradients):
            server_param.data += -1 * self.learning_rate * param_update

    # 模型save and load

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "server" + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset, "server" + ".pt")
        assert (os.path.exists(model_path))
        self.model = torch.load(model_path)

    def model_exists(self):
        return os.path.exists(os.path.join("models", self.dataset, "server" + ".pt"))

    def select_users(self, round, num_users):
        '''selects num_clients clients weighted by number of samples from possible_clients
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))

        Return:
            list of selected clients objects
        '''
        if (num_users == len(self.users)):
            logger.info("All users are selected")
            return self.users

        num_users = min(num_users, len(self.users))
        # np.random.seed(round)
        return np.random.choice(self.users, num_users, replace=False)  # , p=pk)

    # define function for persionalized agegatation.
    def persionalized_update_parameters(self, user, ratio):
        # only argegate the local_weight_update
        for server_param, user_param in zip(self.model.parameters(), user.local_weight_updated):
            server_param.data = server_param.data + user_param.data.clone() * ratio

    def persionalized_aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)

        # store previous parameters
        previous_param = copy.deepcopy(list(self.model.parameters()))
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        # if(self.num_users = self.to)
        for user in self.selected_users:
            total_train += user.train_samples

        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train)
            # self.add_parameters(user, 1 / len(self.selected_users))

        # aaggregate avergage model with previous model using parameter beta
        for pre_param, param in zip(previous_param, self.model.parameters()):
            param.data = (1 - self.beta) * pre_param.data + self.beta * param.data

    # Save loss, accurancy to h5 fiel
    def save_results(self):
        alg = self.dataset + "_" + self.algorithm
        alg = alg + "_" + str(self.learning_rate) + "_" + str(self.beta) + "_" + str(self.lamda) + "_" + str(
            self.num_users) + "u" + "_" + str(self.batch_size) + "b" + "_" + str(self.local_epochs)
        if (self.algorithm == "pFedMe" or self.algorithm == "pFedMe_p"):
            alg = alg + "_" + str(self.K) + "_" + str(self.personal_learning_rate)
        alg = alg + "_" + str(self.times)
        if (len(self.rs_glob_acc) != 0 & len(self.rs_train_acc) & len(self.rs_train_loss)):
            with h5py.File("./results/" + '{}.h5'.format(alg, self.local_epochs), 'w') as hf:
                hf.create_dataset('rs_glob_acc', data=self.rs_glob_acc)
                hf.create_dataset('rs_train_acc', data=self.rs_train_acc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                hf.close()

        # store persionalized value
        alg = self.dataset + "_" + self.algorithm + "_p"
        alg = alg + "_" + str(self.learning_rate) + "_" + str(self.beta) + "_" + str(self.lamda) + "_" + str(
            self.num_users) + "u" + "_" + str(self.batch_size) + "b" + "_" + str(self.local_epochs)
        if (self.algorithm == "pFedMe" or self.algorithm == "pFedMe_p"):
            alg = alg + "_" + str(self.K) + "_" + str(self.personal_learning_rate)
        alg = alg + "_" + str(self.times)
        if (len(self.rs_glob_acc_per) != 0 & len(self.rs_train_acc_per) & len(self.rs_train_loss_per)):
            with h5py.File("./results/" + '{}.h5'.format(alg, self.local_epochs), 'w') as hf:
                hf.create_dataset('rs_glob_acc', data=self.rs_glob_acc_per)
                hf.create_dataset('rs_train_acc', data=self.rs_train_acc_per)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss_per)
                hf.close()

    def test(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, ns = c.test()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct

    def test_COVID(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_f1 = []
        losses = []
        for c in self.users:
            vals, ns = c.test_COVID()
            tot_f1.append(vals['f1'] * 1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_f1
    
    def test_RAD(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_acc = []
        losses = []
        for c in self.users:
            vals, ns = c.test_RAD()
            tot_acc.append(vals['acc'] * 1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_acc

    def train_error_and_loss(self):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, cl, ns = c.train_error_and_loss()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.users]
        # groups = [c.group for c in self.clients]

        return ids, num_samples, tot_correct, losses

    def train_error_and_loss_COVID(self):
        num_samples = []
        tot_f1 = []
        losses = []
        for c in self.users:
            ct, cl, ns = c.train_error_and_loss_COVID()
            tot_f1.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.users]
        # groups = [c.group for c in self.clients]

        return ids, num_samples, tot_f1, losses
    
    def train_error_and_loss_RAD(self):
        num_samples = []
        tot_acc = []
        losses = []
        for c in self.users:
            ct, cl, ns = c.train_error_and_loss_RAD()
            tot_acc.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.users]
        # groups = [c.group for c in self.clients]

        return ids, num_samples, tot_acc, losses

    def test_persionalized_model(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        for c in self.users:
            ct, ns = c.test_persionalized_model()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct

    def test_persionalized_model_COVID(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_f1 = []
        for c in self.users:
            vals, ns = c.test_persionalized_model_COVID()
            tot_f1.append(vals['f1'] * 1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_f1

    def train_error_and_loss_persionalized_model(self):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, cl, ns = c.train_error_and_loss_persionalized_model()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.users]
        # groups = [c.group for c in self.clients]

        return ids, num_samples, tot_correct, losses

    def train_error_and_loss_persionalized_model_COVID(self):
        num_samples = []
        tot_f1 = []
        losses = []
        for c in self.users:
            ct, cl, ns = c.train_error_and_loss_persionalized_model_COVID()
            tot_f1.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.users]
        # groups = [c.group for c in self.clients]

        return ids, num_samples, tot_f1, losses

    def evaluate(self):
        stats = self.test()
        stats_train = self.train_error_and_loss()
        glob_acc = np.sum(stats[2]) * 1.0 / np.sum(stats[1])
        train_acc = np.sum(stats_train[2]) * 1.0 / np.sum(stats_train[1])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1])
        self.rs_glob_acc.append(glob_acc)
        self.rs_train_acc.append(train_acc)
        self.rs_train_loss.append(train_loss)
        # logger.info("stats_train[1]",stats_train[3][0])
        logger.info("Average Global Testing Accurancy: ", glob_acc)
        logger.info("Average Global Trainning Accurancy: ", train_acc)
        logger.info("Average Global Trainning Loss: ", train_loss)

    def evaluate_COVID(self):
        # TODO
        stats = self.test_COVID()
        stats_train = self.train_error_and_loss_COVID()
        glob_f1 = np.sum(stats[2])/len(stats[2])
        train_f1 = np.sum(stats_train[2])/len(stats_train[2])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        #train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]) / np.sum(stats_train[1])
        self.rs_glob_acc.append(glob_f1)
        self.rs_train_acc.append(train_f1)
        #self.rs_train_loss.append(train_loss)
        # logger.info("stats_train[1]",stats_train[3][0])
        logger.info("Average Global Testing F1: ", glob_f1)
        logger.info("Average Global Trainning F1: ", train_f1)
        logger.info(f"Details Global Testing F1: {stats[2]}")
        logger.info(f"Details Global Trainning F1: {stats_train[2]}")
        #logger.info("Average Global Trainning Loss: ", train_loss)
        
    def evaluate_RAD(self):
        # TODO
        stats = self.test_RAD()
        #stats_train = self.train_error_and_loss_RAD()
        glob_acc = np.sum(stats[2])/len(stats[2])
        #train_acc = np.sum(stats_train[2])/len(stats_train[2])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        #train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]) / np.sum(stats_train[1])
        self.rs_glob_acc.append(glob_acc)
        #self.rs_train_acc.append(train_acc)
        #self.rs_train_loss.append(train_loss)
        # logger.info("stats_train[1]",stats_train[3][0])
        logger.info(f"Average Global Testing acc: {glob_acc}" )
        #logger.info(f"Average Global Trainning acc: {train_acc}")
        logger.info(f"Details Global Testing acc: {stats[2]}")
        #logger.info(f"Details Global Trainning acc: {stats_train[2]}")
        #logger.info("Average Global Trainning Loss: ", train_loss)

    def evaluate_personalized_model(self):
        stats = self.test_persionalized_model()
        stats_train = self.train_error_and_loss_persionalized_model()
        glob_acc = np.sum(stats[2]) * 1.0 / np.sum(stats[1])
        train_acc = np.sum(stats_train[2]) * 1.0 / np.sum(stats_train[1])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1])
        self.rs_glob_acc_per.append(glob_acc)
        self.rs_train_acc_per.append(train_acc)
        self.rs_train_loss_per.append(train_loss)
        # logger.info("stats_train[1]",stats_train[3][0])
        logger.info("Average Personal Testing Accurancy: ", glob_acc)
        logger.info("Average Personal Trainning Accurancy: ", train_acc)
        logger.info("Average Personal Trainning Loss: ", train_loss)

    def evaluate_personalized_model_COVID(self):
        stats = self.test_persionalized_model_COVID()
        stats_train = self.train_error_and_loss_persionalized_model_COVID()
        glob_f1 = np.sum(stats[2]) * 1.0 / len(stats[2])
        train_f1 = np.sum(stats_train[2]) * 1.0 / len(np.sum(stats_train[2]))
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        #train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1])
        #self.rs_glob_acc_per.append(glob_acc)
        #self.rs_train_acc_per.append(train_acc)
        #self.rs_train_loss_per.append(train_loss)
        # logger.info("stats_train[1]",stats_train[3][0])
        logger.info("Average Personal Testing F1: ", glob_f1)
        logger.info("Average Personal Trainning F1: ", train_f1)
        logger.info(f"Details Personal Testing F1: {stats[2]}")
        logger.info(f"Details Personal Trainning F1: {stats_train[2]}")
        #logger.info("Average Personal Trainning Loss: ", train_loss)

    def evaluate_one_step(self):
        for c in self.users:
            c.train_one_step()

        stats = self.test()
        stats_train = self.train_error_and_loss()

        # set local model back to client for training process.
        for c in self.users:
            c.update_parameters(c.local_model)

        glob_acc = np.sum(stats[2]) * 1.0 / np.sum(stats[1])
        train_acc = np.sum(stats_train[2]) * 1.0 / np.sum(stats_train[1])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1])
        self.rs_glob_acc_per.append(glob_acc)
        self.rs_train_acc_per.append(train_acc)
        self.rs_train_loss_per.append(train_loss)
        # logger.info("stats_train[1]",stats_train[3][0])
        logger.info("Average Personal Testing Accurancy: ", glob_acc)
        logger.info("Average Personal Trainning Accurancy: ", train_acc)
        logger.info("Average Personal Trainning Loss: ", train_loss)

    def evaluate_one_step_COVID(self):
        # TODO
        for c in self.users:
            logger.info(f'>> onestep finetune @ client {c.id}')
            c.train_one_step_COVID()
            logger.info(f'>> load best onestepfinetune model @ client {c.id}')
            c.load_best_model_onestepfinetune()

        stats = self.test_COVID()
        stats_train = self.train_error_and_loss_COVID()

        # set local model back to client for training process.
        for c in self.users:
            c.update_parameters(c.local_model)

        glob_f1 = np.average(stats[2])
        train_f1 = np.average(stats_train[2])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        #train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]) / np.sum(stats_train[1])
        self.rs_glob_acc_per.append(glob_f1)
        self.rs_train_acc_per.append(train_f1)
        #self.rs_train_loss_per.append(train_loss)
        # logger.info("stats_train[1]",stats_train[3][0])
        logger.info("Average Personal Testing F1: ", glob_f1)
        logger.info("Average Personal Trainning F1: ", train_f1)
        logger.info(f"Details Personal Testing F1: {stats[2]}")
        logger.info(f"Details Personal Trainning F1: {stats_train[2]}")
        #logger.info("Average Personal Trainning Loss: ", train_loss)
        
    def evaluate_one_step_RAD(self):
        # TODO
        for c in self.users:
            logger.info(f'>> onestep finetune @ client {c.id}')
            c.train_one_step_RAD()
            logger.info(f'>> load best onestepfinetune model @ client {c.id}')
            c.load_best_model_onestepfinetune()

        stats = self.test_RAD()
        #stats_train = self.train_error_and_loss_RAD()

        # set local model back to client for training process.
        for c in self.users:
            c.update_parameters(c.local_model)

        glob_acc = np.average(stats[2])
        #train_acc = np.average(stats_train[2])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        #train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]) / np.sum(stats_train[1])
        self.rs_glob_acc_per.append(glob_acc)
        #self.rs_train_acc_per.append(train_acc)
        #self.rs_train_loss_per.append(train_loss)
        # logger.info("stats_train[1]",stats_train[3][0])
        logger.info(f"Average Personal Testing acc: {glob_acc}")
        #logger.info("Average Personal Trainning acc: ", train_acc)
        logger.info(f"Details Personal Testing acc: {stats[2]}")
        #logger.info(f"Details Personal Trainning acc: {stats_train[2]}")
        #logger.info("Average Personal Trainning Loss: ", train_loss)
        
# cyclic secure aggregation integrated Personalized-FedAvg

class CSAPerAvg(Server):
    def __init__(self, device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                 local_epochs, optimizer, num_users, times, args):
        super().__init__(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                         local_epochs, optimizer, num_users, times)

        # Initialize data for all  users
        if dataset=='COVID':
            total_users = len(args.COVID_client_labels)
            for i in range(total_users):
                id, train, test, user_params = read_user_data_COVID(i, args)
                user = UserPerAvg(device, id, train, test, model, batch_size, learning_rate, beta, lamda, local_epochs,
                               optimizer,total_users, num_users,user_params)
                self.users.append(user)
                self.total_train_samples += user.train_samples

            logger.info(f"Number of users / total users: {num_users}/{total_users}")
            logger.info("Finished creating FedAvg server.")
        elif dataset=='RAD':
            exp = args.exp
            df = pd.read_csv(f'{exp}.csv')
            clients = list(set(df.client))
            clients = [x for x in clients if x!='NAN']
            total_users = len(clients)
            for i in range(total_users):
                id, train, test, user_params = read_user_data_RAD(i, args)
                user = UserAVG(device, id, train, test, model, batch_size, learning_rate, beta, lamda, local_epochs, optimizer, user_params)
                self.users.append(user)
                self.total_train_samples += user.train_samples
            logger.info(f"Number of users / total users: {num_users}/{total_users}")
            logger.info("Finished creating FedAvg server.")

    def send_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad)
        for user in self.users:
            user.set_grads(grads)

    def train(self):
        loss = []
        for glob_iter in range(self.num_glob_iters):
            logger.info(f"--------------- Global Epoch: {glob_iter} ---------------")
            # send all parameter for users
            args.current_global_epoch_id = glob_iter
            self.send_parameters()
            logger.info('>> broadcast server model to selected clients')

            # Evaluate the server model on all without finetune
            logger.info(">> Evaluate the server model on all without finetune")
            if args.dataset == 'COVID':
                self.evaluate_COVID()
                #pass
            elif args.dataset == 'RAD':
                self.evaluate_RAD()
            else:
                self.evaluate()

            # Evaluate gloal model on user for each interation
            logger.info("")
            logger.info(">> Evaluate client model with one step update")
            if args.dataset == 'COVID':
                self.evaluate_one_step_COVID()
                #pass
            elif args.dataset == 'RAD':
                self.evaluate_one_step_RAD()
            else:
                self.evaluate_one_step()
            #self.evaluate_personalized_model()

            # 由于上一步的测试已经finetune了client模型，所以重新把server模型发送到client
            logger.info(">> broadcast server model to selected clients again")
            self.send_parameters()

            # choose several users to send back upated model to server
            logger.info(">> train on selected clients")
            self.selected_users = self.users
            for user in self.selected_users:
                user.train(self.local_epochs)  # * user.train_samples

            # 在合并之前，需要让所有的user load best model
            logger.info(f'>> load best model during training for all clients')
            if args.dataset == 'COVID':
                for user in self.selected_users:
                    user.load_best_model()
            elif args.dataset == 'RAD':
                for user in self.selected_users:
                    user.load_best_model()

            logger.info(f'>> aggregate client model to server model')
            # 直接合并模型参数, 文章中的算法是直接average模型参数
            #self.aggregate_parameters()
            # 合并梯度，再用梯度一次更新server model
            self.aggregate_parameters_with_gradients_CSA(glob_iter = glob_iter, HE=args.HE)

            # 每一个global epoch保存server模型
            logger.info(f'>> save server model')
            if args.dataset=='COVID':
                save_checkpoint_server(self.model, args)
            elif args.dataset=='RAD':
                save_checkpoint_server(self.model, args)

        #self.save_results()
        #self.save_model()

# Personalized-FedAvg

class PerAvg(Server):
    def __init__(self, device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                 local_epochs, optimizer, num_users, times, args):
        super().__init__(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                         local_epochs, optimizer, num_users, times)

        # Initialize data for all  users
        if dataset=='COVID':
            total_users = len(args.COVID_client_labels)
            for i in range(total_users):
                id, train, test, user_params = read_user_data_COVID(i, args)
                user = UserPerAvg(device, id, train, test, model, batch_size, learning_rate, beta, lamda, local_epochs,
                               optimizer,total_users, num_users,user_params)
                self.users.append(user)
                self.total_train_samples += user.train_samples

            logger.info(f"Number of users / total users: {num_users}/{total_users}")
            logger.info("Finished creating FedAvg server.")
        elif dataset=='RAD':
            exp = args.exp
            df = pd.read_csv(f'{exp}.csv')
            clients = list(set(df.client))
            clients = [x for x in clients if x!='NAN']
            total_users = len(clients)
            for i in range(total_users):
                id, train, test, user_params = read_user_data_RAD(i, args)
                user = UserAVG(device, id, train, test, model, batch_size, learning_rate, beta, lamda, local_epochs, optimizer, user_params)
                self.users.append(user)
                self.total_train_samples += user.train_samples
            logger.info(f"Number of users / total users: {num_users}/{total_users}")
            logger.info("Finished creating FedAvg server.")

    def send_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad)
        for user in self.users:
            user.set_grads(grads)

    def train(self):
        loss = []
        for glob_iter in range(self.num_glob_iters):
            logger.info(f"--------------- Global Epoch: {glob_iter} ---------------")
            # send all parameter for users
            args.current_global_epoch_id = glob_iter
            self.send_parameters()
            logger.info('>> broadcast server model to selected clients')

            # Evaluate the server model on all without finetune
            logger.info(">> Evaluate the server model on all without finetune")
            if args.dataset == 'COVID':
                self.evaluate_COVID()
            elif args.dataset == 'RAD':
                self.evaluate_RAD()
            else:
                self.evaluate()

            # Evaluate gloal model on user for each interation
            logger.info("")
            logger.info(">> Evaluate client model with one step update")
            if args.dataset == 'COVID':
                self.evaluate_one_step_COVID()
            elif args.dataset == 'RAD':
                self.evaluate_one_step_RAD()
            else:
                self.evaluate_one_step()
            #self.evaluate_personalized_model()

            # 由于上一步的测试已经finetune了client模型，所以重新把server模型发送到client
            logger.info(">> broadcast server model to selected clients again")
            self.send_parameters()

            # choose several users to send back upated model to server
            logger.info(">> train on selected clients")
            self.selected_users = self.select_users(glob_iter, self.num_users)
            for user in self.selected_users:
                user.train(self.local_epochs)  # * user.train_samples

            # 在合并之前，需要让所有的user load best model
            logger.info(f'>> load best model during training for all clients')
            if args.dataset == 'COVID':
                for user in self.selected_users:
                    user.load_best_model()
            elif args.dataset == 'RAD':
                for user in self.selected_users:
                    user.load_best_model()

            logger.info(f'>> aggregate client model to server model')
            # 直接合并模型参数, 文章中的算法是直接average模型参数
            #self.aggregate_parameters()
            # 合并梯度，再用梯度一次更新server model
            self.aggregate_parameters_with_gradients()

            # 每一个global epoch保存server模型
            logger.info(f'>> save server model')
            if args.dataset=='COVID':
                save_checkpoint_server(self.model, args)
            elif args.dataset=='RAD':
                save_checkpoint_server(self.model, args)

            ## 对于每一轮
            ## 先测试finetune
            ## A/X/rep0_globalepoch0_testid0_onestepfinetune_saved_model.pth 是基于上一轮server model, local finetune的最佳模型  --> 从这里拿finetune的结果
            ## A/X/current_rep0_globalepoch0_testid0_onestepfinetune_saved_model.pth 是基于上一轮server model, local finetune的当前模型
            ## 再训练local模型
            ## A/X/rep0_globalepoch0_testid0_saved_model.pth 是local训练的最佳模型
            ## A/X/current_rep0_globalepoch0_testid0_saved_model.pth 是local训练的当前模型
            ## 合并当前最佳local模型，变成最佳server模型
            ## server/X/rep0_globalepoch0_testid0_saved_model.pth  --> 从这里拿finetune前的结果，要注意globalepoch数要-1
            ## 这个server模型被用于下一轮的finetune

        #self.save_results()
        #self.save_model()

# Traditional-FedAvg

class FedAvg(Server):
    def __init__(self, device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                 local_epochs, optimizer, num_users, times, args):
        super().__init__(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                         local_epochs, optimizer, num_users, times)

        # Initialize data for all  users
        if dataset=='COVID':
            total_users = len(args.COVID_client_labels)
            for i in range(total_users):
                id, train, test, user_params = read_user_data_COVID(i, args)
                user = UserAVG(device, id, train, test, model, batch_size, learning_rate, beta, lamda, local_epochs,
                               optimizer,user_params)
                self.users.append(user)
                self.total_train_samples += user.train_samples

            logger.info(f"Number of users / total users: {num_users}/{total_users}")
            logger.info("Finished creating FedAvg server.")
        elif dataset=='RAD':
            exp = args.exp
            df = pd.read_csv(f'{exp}.csv')
            clients = list(set(df.client))
            clients = [x for x in clients if x!='NAN']
            total_users = len(clients)
            for i in range(total_users):
                id, train, test, user_params = read_user_data_RAD(i, args)
                user = UserAVG(device, id, train, test, model, batch_size, learning_rate, beta, lamda, local_epochs, optimizer, user_params)
                self.users.append(user)
                self.total_train_samples += user.train_samples
            logger.info(f"Number of users / total users: {num_users}/{total_users}")
            logger.info("Finished creating FedAvg server.")

    def send_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad)
        for user in self.users:
            user.set_grads(grads)

    def train(self):
        loss = []
        for glob_iter in range(self.num_glob_iters):
            logger.info(f"--------------- Global Epoch: {glob_iter} ---------------")
            # loss_ = 0
            args.current_global_epoch_id = glob_iter
            self.send_parameters()
            logger.info('>> broadcast server model to selected clients')

            # Evaluate model each interation
            logger.info('>> evaluate server model on all client in each global epoch')
            if args.dataset=='COVID':
                self.evaluate_COVID()
            elif args.dataset=='RAD':
                self.evaluate_RAD()
            else:
                self.evaluate()

            # Evaluate on clients
            # Evaluate gloal model on user for each interation
            logger.info("")
            logger.info(">> Evaluate client model with one step update")
            if args.dataset == 'COVID':
                self.evaluate_one_step_COVID()
            elif args.dataset=='RAD':
                self.evaluate_one_step_RAD()
            else:
                self.evaluate_one_step()
            # self.evaluate_personalized_model()

            # 由于上一步的测试已经finetune了client模型，所以重新把server模型发送到client
            logger.info(">> broadcast server model to selected clients again")
            self.send_parameters()

            logger.info('>> train on each client')
            self.selected_users = self.select_users(glob_iter, self.num_users)
            for user in self.selected_users:
                user.train(self.local_epochs)  # * user.train_samples

            # 在合并之前，需要让所有的user load best model
            logger.info(f'>> load best model for all clients')
            if args.dataset == 'COVID':
                for user in self.selected_users:
                    user.load_best_model()
            elif args.dataset == 'RAD':
                for user in self.selected_users:
                    user.load_best_model()

            # 直接合并模型参数
            #self.aggregate_parameters()
            # 合并梯度，再用梯度一次更新server model
            logger.info(f'>> aggregate client model to server model')
            self.aggregate_parameters_with_gradients()
            # loss_ /= self.total_train_samples
            # loss.append(loss_)
            # logger.info(loss_)

            # 每一个global epoch保存server模型
            if args.dataset=='COVID':
                save_checkpoint_server(self.model,args)
            elif args.dataset=='RAD':
                save_checkpoint_server(self.model,args)

        # logger.info(loss)
        #self.save_results()
        #self.save_model()

def main(dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
         local_epochs, optimizer, numusers, K, personal_learning_rate, times, gpu, args):
    # Get device status: Check GPU or CPU
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")
    if dataset=='COVID':
        for test_id in args.test_ids:
            for direction in args.directions: #['X','Y','Z']
                logger.info(f'--------------- train with test_id {test_id} at direction {direction} ---------------')
                for i in range(times):
                    logger.info(f"--------------- Repeat Experiment: {i} ---------------")

                    args.current_repeat_experiment_id = i
                    args.test_id = test_id
                    args.direction = direction

                    logger.info(f"Using {torch.cuda.device_count()} GPUs")
                    model = unm.UNet(in_channels=3, out_channels=2, init_features=16).to(device)

                    # select algorithm
                    if (algorithm == "FedAvg"):
                        server = FedAvg(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda,
                                        num_glob_iters,
                                        local_epochs, optimizer, numusers, i, args)

                    if (algorithm == "PerAvg"):
                        #logger.info(model)
                        server = PerAvg(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda,
                                        num_glob_iters,
                                        local_epochs, optimizer, numusers, i, args)

                    if algorithm == 'CSAPerAvg':
                        server = CSAPerAvg(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda,
                                        num_glob_iters,
                                        local_epochs, optimizer, numusers, i, args)

                    server.train()
                    server.test_COVID()
            
    elif dataset=='RAD':
        for i in range(times):
            logger.info(f"--------------- Repeat Experiment: {i} ---------------")
            args.current_repeat_experiment_id = i
            
            logger.info(f"Using {torch.cuda.device_count()} GPUs")
            
            model = DenseNet(
                growth_rate=32,
                block_config=(1, 3, 6, 4),
                n_classes = 2,
                in_channels=1)
            model = model.to(device=device)
            
            # select algorithm
            if (algorithm == "FedAvg"):
                server = FedAvg(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                                local_epochs, optimizer, numusers, i, args)

            if (algorithm == "PerAvg"):
                server = PerAvg(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                                local_epochs, optimizer, numusers, i, args)
            
            if algorithm == 'CSAPerAvg':
                        server = CSAPerAvg(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda,
                                        num_glob_iters,
                                        local_epochs, optimizer, numusers, i, args)

            server.train()
            server.test_RAD()
        

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # Main
    parser.add_argument("--dataset", type=str, default="RAD", choices=["COVID", 'RAD'])
    parser.add_argument("--algorithm", type=str, default="FedAvg", choices=["CSAPerAvg" ,"PerAvg", "FedAvg"])
    # Basic
    parser.add_argument("--model", type=str, default="", choices=["cnn","covseg"])
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=0.005, help="Local learning rate")
    parser.add_argument("--beta", type=float, default=0.001,
                        help="Second learning rate of Per-FedAvg, very important")
    parser.add_argument("--lamda", type=int, default=15, help="Regularization term")
    parser.add_argument("--num_global_iters", type=int, default=800, help='number of epochs')
    parser.add_argument("--local_epochs", type=int, default=20)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--numusers", type=int, default=3, help="Number of selected Users per round, should calculate with r and the total number of users")
    parser.add_argument("--r", type=float, default=1, help="fraction of users to choose in each iteration")
    parser.add_argument("--K", type=int, default=5, help="Computation steps")
    parser.add_argument("--personal_learning_rate", type=float, default=0.1,
                        help="Persionalized learning rate to caculate theta aproximately using K steps for pFedMe")
    parser.add_argument("--onestepupdate",type=int,default=1)
    parser.add_argument("--times", type=int, default=5, help="number of repeat experiments")
    parser.add_argument("--gpu", type=int, default=0,
                        help="Which GPU to run the experiments, -1 mean CPU, 0,1,2 for GPU")
    
    #COVID dataset related
    parser.add_argument("--COVID_data_root",
                        default='/home/zhouj0d/Science/PID15.PPPMLCOV/repo/data/raw_data/device_patients_smalltest',
                        help='path/to/COVID/dataset/root')
    parser.add_argument("--COVID_checkpoint_root",
                        default='/home/zhouj0d/Science/PID15.PPPMLCOV/repo/result/pppml_exp1',
                        help="path/to/COVID/checkpoint/root")
    parser.add_argument("--directions",default='X,Y,Z',help='directions to train')
    parser.add_argument("--test_ids",
                        default='0,1,2,3,4',
                        help='ids as testdataset for COVID')
    
    parser.add_argument("--COVID_client_labels", default='P_I256_1.0 G_CT520_1.25 S_SDE_1.0 P_I_1.0 G_CT540_1.25 P_B16_7.5 S_SS_2.0 P_B16_2.0 P_I256_5.0 G_B_5.0 TinyMix', help="label for COVID clients")
    
    # RAD related
    parser.add_argument("--exp",default='c3_split1')
    parser.add_argument("--RAD_checkpoint_root",default='/home/zhouj0d/Science/PID15.PPPMLCOV/repo/result')
    
    # Other parameters
    parser.add_argument("--HE", type=int, default=1)
    parser.add_argument("--debug",default='None')
    #parser.add_argument("--log",default='/home/zhouj0d/Science/PID15.PPPMLCOV/repo/result/pppml_exp1/log.txt',help='path/to/log')
    #parser.add_argument("--logerr", default='/home/zhouj0d/Science/PID15.PPPMLCOV/repo/result/pppml_exp1/logerr.txt',help='path/to/logerr')
    args = parser.parse_args()

    if os.path.isdir(args.COVID_checkpoint_root):
        pass
    else:
        os.system(f"mkdir -p {args.COVID_checkpoint_root}")
    sys.stdout = Logger(f"{args.COVID_checkpoint_root}/log_{args.directions}_{args.test_ids}.txt", sys.stdout)
    sys.stderr = Logger(f"{args.COVID_checkpoint_root}/logerr_{args.directions}_{args.test_ids}.txt", sys.stderr)

    ## Setting for HE ##
    if args.HE == 1:
        args.HE = True
    else:
        args.HE = False

    bits_scale = 26
    # Create TenSEAL context
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31]
    )
    # set the scale
    context.global_scale = pow(2, bits_scale)
    # galois keys are required to do ciphertext rotations
    context.generate_galois_keys()

    ###################

    if args.dataset=='RAD':
        args.params = {"checkpoint_dir": f'{args.RAD_checkpoint_root}/{args.exp}'}
        exp = args.exp
        df = pd.read_csv(f'{exp}.csv')
        clients = list(set(df.client))
        clients = [x for x in clients if x!='NAN']
        args.numusers = len(clients)
        if os.path.isdir(args.params["checkpoint_dir"]) == False:
            os.system(f'mkdir -p {args.params["checkpoint_dir"]}')
        logger = log_creater(f'{args.params["checkpoint_dir"]}/log.txt')
    if args.dataset=='COVID':
        args.COVID_client_labels=args.COVID_client_labels.split(' ')
        args.numusers=len(args.COVID_client_labels)
        args.test_ids=[int(x) for x in args.test_ids.split(',')]
        args.directions = args.directions.split(',')

        args.params = {
            "n_epochs": args.num_global_iters,  # this is the maximum epoch, but usually the training finished less than 100 epochs.
            "batch_size": args.batch_size,
            "lr": args.learning_rate,  # learning rate

            #########################################
            # do not forget to change!
            "channels": 3,  # the input channels
            "output_channels": 2,  # the output semantics, channel 0 (first channel) is the negative
            "no_rotate": False,  # if the sample shape [x, y], x != y, set it to True
            # the input channel of the sample: window_width * data_channels + enhanced_channels
            # channels: 3 for stage one and 5 for stage two
            #########################################

            'workers': 32,  # num CPU for the parallel data loading
            "balance_weights": [1000000000, 1],  # give freq bound during training
            "train_data_dir": args.COVID_data_root,
            "weight_dir": None,
            "default_weight": True,
            # if you don't have or do not want to use the voxel-wise balance weight, set it True
            "test_data_dir": None,  # use train for test
            "checkpoint_dir": args.COVID_checkpoint_root,
            "saved_model_filename": None,
            "device": "cuda:0" if torch.cuda.is_available() else "cpu",
            "test_id": 0,  # the fold of training. one in [0, 1, 2, 3, 4] as the program ONLY support five fold.
            "wrong_patient_id": [],  # ['xwqg-A00085', 'xwqg-A00121', 'xwqg-B00027', 'xwqg-B00034'], 用的是这里的[-1]%5来决定训练和测试
            "best_f1": None,  # initially the "best_f1" is None
            "init_features": 16,  # see source_codes.models.Unet_2D.U_net_Model.py
            "beta": 1.25,  # number times recall is more important than precision.
            "target_performance": 0.7,
            # what does "high performance" mean for [precision, recall], this is freq very important parameter. make sure the label
            # quality is enough for freq high "target_performance". If the quality is very good, this can be higher like 0.93-0.95.
            # Usually we require the model is able to satisfy:
            # recall(precision) > "target_performance" and precision(recall) > 0.5 * "target_performance", simultaneously.
            "baseline_performance_recall": 0.5,
            "flip_remaining:": 1
            # flip_remaining is one of the most important parameters. Initially, the model will have freq early 100% recall, but
            # the precision is low.
            # "precision phase": during "precision phase" we try to make precision > "target_performance"
            # "recall phase":  during "recall phase" we try to make recall > "target_performance"
            # "fluctuate phase": during "fluctuate phase", we require recall fluctuate around "target_performance"
            # the training process is:
            # "precision phase" -> "recall phase" -> "precision phase" -> "recall phase" ->... flip_remaining -= 1 when convert
            # from "recall phase" to "precision phase". if flip_remaining == 0, change to "fluctuate phase". 20 epochs later,
            # the training finishes.
        }

    logger.info("=" * 80)
    logger.info("Summary of training process:")
    logger.info("Algorithm\t\t\t{}".format(args.algorithm))
    logger.info("Batch size\t\t\t{}".format(args.batch_size))
    logger.info("Learing rate\t\t\t{}".format(args.learning_rate))
    logger.info("Subset of users\t\t\t{}".format(args.numusers))
    logger.info("Global rounds\t\t\t{}".format(args.num_global_iters))
    logger.info("Local rounds\t\t\t{}".format(args.local_epochs))
    logger.info("Dataset\t\t\t{}".format(args.dataset))
    logger.info("HE\t\t\t{}".format(args.HE))
    if args.dataset=='COVID':
        logger.info("Dataset root\t\t\t{}".format(args.COVID_data_root))
    logger.info("Local Model\t\t\t{}".format(args.model))
    logger.info("=" * 80)

    start = time.time()
    logger.info(f"start time: {start}")
    main(
        dataset=args.dataset,
        algorithm=args.algorithm,
        model=args.model,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        beta=args.beta,
        lamda=args.lamda,
        num_glob_iters=args.num_global_iters,
        local_epochs=args.local_epochs,
        optimizer=args.optimizer,
        numusers=args.numusers,
        K=args.K,
        personal_learning_rate=args.personal_learning_rate,
        times=args.times,
        gpu=args.gpu,
        args=args
    )
    stop = time.time()
    logger.info(f"end time: {start}")
    logger.info("Total running time: ", stop - start)
