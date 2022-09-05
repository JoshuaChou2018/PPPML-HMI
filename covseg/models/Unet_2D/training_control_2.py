import sys
sys.path.append('/ibex/scratch/projects/c2052/Lung_CAD_NMI/source_codes')
import models.Unet_2D.train as run_model
import torch

TRAIN_DATA_DIR = "/ibex/scratch/projects/c2052/blood_vessel_seg/training_samples_enhanced/"
WEIGHT_DIR = "/ibex/scratch/projects/c2052/blood_vessel_seg/"
CHECKPOINT_DIR = "/ibex/scratch/projects/c2052/blood_vessel_seg/check_points/stage_two/"

params = {
    "n_epochs": 3000,  # this is the maximum epoch
    "batch_size": 64,
    "lr": 1e-4,
    "channels": 5,  # the input channel of the sample: window_width * data_channels + enhanced_channels
    'workers': 32,  # num CPU for the parallel data loading
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


def modify_params(direction='X', rim_enhanced=False, test_id=0):
    params["test_id"] = test_id
    params["saved_model_filename"] = str(test_id) + "_saved_model.pth"
    train_dict = TRAIN_DATA_DIR + direction
    params["train_data_dir"] = train_dict
    params["test_data_dir"] = train_dict
    if rim_enhanced:
        weight_dict = WEIGHT_DIR + "balance_weight_array_rim_enhance/" + direction
        check_point_dict = CHECKPOINT_DIR + "balance_weight_array_rim_enhance/" + direction
    else:
        weight_dict = WEIGHT_DIR + "balance_weight_array/" + direction
        check_point_dict = CHECKPOINT_DIR + "balance_weight_array/" + direction
    params["weight_dir"] = weight_dict
    params["checkpoint_dir"] = check_point_dict
    if params["balance_weights"] is None:
        params["balance_weights"] = [1000000000, 1]


def training_one_direction(direction, rim_enhanced=False):
    for test_id in range(5):
        modify_params(direction, rim_enhanced, test_id)
        print('directing:', direction, "rim_enhanced:", rim_enhanced, 'test_id', test_id)
        run_model.training(params)


def training_one_test_id(test_id, rim_enhanced=False):
    for direction in ['X', 'Y', 'Z']:
        modify_params(direction, rim_enhanced, test_id)
        print('directing:', direction, "rim_enhanced:", rim_enhanced, 'test_id', test_id)
        run_model.training(params)


def training_all_direction(rim_enhanced=False):
    training_one_direction('X', rim_enhanced)
    training_one_direction('Y', rim_enhanced)
    training_one_direction('Z', rim_enhanced)


def training_all_test_id(rim_enhanced=False):
    for test_id in range(5):
        training_one_test_id(test_id, rim_enhanced)


training_one_test_id(3)
training_one_test_id(4)

exit()
CHECKPOINT_DIR = "/ibex/scratch/projects/c2052/blood_vessel_seg/check_points/stage_one_infinite_flip/"
params["flip_remaining:"] = 10
training_one_test_id(0)
exit()

CHECKPOINT_DIR = "/ibex/scratch/projects/c2052/blood_vessel_seg/check_points/stage_one_32features/"
print("32 features")
params["n_epochs"] = 500
params["flip_remaining:"] = 2
params["batch_size"] = 48
params["init_features"] = 32
training_all_test_id()
exit()

exit()
