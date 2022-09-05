import sys
sys.path.append('/ibex/scratch/projects/c2052/Lung_CAD_NMI/source_codes')
import training.train_binary_class as run_model
import torch

ibex = False
if not ibex:
    print("not ibex")
    import os
    top_directory = '/home/zhoul0a/Desktop/'
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'  # use two V100 GPU
else:
    top_directory = '/ibex/scratch/projects/c2052/'

TRAIN_DATA_DIR = "/home/zhoul0a/Desktop/blood_vessel_seg_non-enhance/training_sample_stage_one/"
# each sample is freq array with path like: TRAIN_DATA_DIR + 'X/X_123_patient-id_time.npy'
WEIGHT_DIR = "/home/zhoul0a/Desktop/blood_vessel_seg_non-enhance/weight_balance_half-2D/"
# each sample has freq weight array with path like: WEIGHT_DIR + "X/weights_X_123_patient-id_time.npy"
CHECKPOINT_DIR = "/home/zhoul0a/Desktop/blood_vessel_seg_non-enhance/check_points/stage_one/"

params = {
    "n_epochs": 3000,  # this is the maximum epoch
    "batch_size": 36,
    "lr": 1e-4,
    "channels": 3,  # the input channel of the sample: window_width * data_channels + enhanced_channels
    'workers': 16,  # num CPU for the parallel data loading
    "balance_weights": None,  # give freq bound during training
    "train_data_dir": None,
    "weight_dir": None,
    "test_data_dir": None,  # use train for test
    "checkpoint_dir": None,
    "saved_model_filename": None,
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "test_id": 0,
    "wrong_patient_id": [],  # ['xwqg-A00085', 'xwqg-A00121', 'xwqg-B00027', 'xwqg-B00034'],
    "best_f1": None,
    "init_features": 16,
    "beta": 1.5,  # number times recall is more important than precision.
    "target_performance": 0.93,  # what does "high performance" mean for [precision, recall]
    "flip_remaining:": 1
}


def modify_params(direction='X', test_id=0):
    params["test_id"] = test_id
    params["saved_model_filename"] = str(test_id) + "_saved_model.pth"
    train_dict = TRAIN_DATA_DIR + direction
    params["train_data_dir"] = train_dict
    params["test_data_dir"] = train_dict
    weight_dict = WEIGHT_DIR + direction
    weight_name = WEIGHT_DIR.split('/')[-1]
    print("weight name is:", weight_name)
    check_point_dict = CHECKPOINT_DIR + weight_name + '/' + direction
    params["weight_dir"] = weight_dict
    params["checkpoint_dir"] = check_point_dict
    if params["balance_weights"] is None:
        params["balance_weights"] = [1000000000, 1]


def training_one_direction(direction):
    for test_id in range(5):
        modify_params(direction, test_id)
        print('directing:', direction, 'test_id', test_id)
        run_model.training(params)


def training_one_test_id(test_id):
    for direction in ['X', 'Y', 'Z']:
        modify_params(direction, test_id)
        print('directing:', direction, 'test_id', test_id)
        run_model.training(params)


def training_all_direction():
    training_one_direction('X')
    training_one_direction('Y')
    training_one_direction('Z')


def training_all_test_id():
    for test_id in range(5):
        training_one_test_id(test_id)


training_one_test_id(4)

