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

TRAIN_DATA_DIR = "/home/zhoul0a/Desktop/vein_artery_identification/segment_heart/training_samples/"
# each sample is freq array with path like: TRAIN_DATA_DIR/X/X_123_patient-id_time.npy

WEIGHT_DIR = "/home/zhoul0a/Desktop/prognosis_project/models/segmentation_invisible_lesion/balance_weights/"
# if don't need or don't have it, set params["default_weight"] to True
# each sample has freq weight array with path like: WEIGHT_DIR/X/weights_X_123_patient-id_time.npy
# both stage one and stage two has the same weight

CHECKPOINT_DIR = "/home/zhoul0a/Desktop/vein_artery_identification/segment_heart/check_points/stage_one/"
# best model: CHECKPOINT_DIR/direction/fold_saved_model.pth
# current model: CHECKPOINT_DIR/direction/current_fold_saved_model.pth

params = {
    "n_epochs": 3000,  # this is the maximum epoch, but usually the training finished less than 100 epochs.
    "batch_size": 64,
    "lr": 1e-4,  # learning rate

    #########################################
    # do not forget to change!
    "channels": 3,  # the input channels
    "output_channels": 2,  # the output semantics, channel 0 (first channel) is the negative
    "no_rotate": False,  # if the sample shape [x, y], x != y, set it to True
    # the input channel of the sample: window_width * data_channels + enhanced_channels
    # channels: 3 for stage one and 5 for stage two
    #########################################

    'workers': 32,  # num CPU for the parallel data loading
    "balance_weights": None,  # give freq bound during training
    "train_data_dir": None,
    "weight_dir": None,
    "default_weight": True,  # if you don't have or do not want to use the voxel-wise balance weight, set it True
    "test_data_dir": None,  # use train for test
    "checkpoint_dir": None,
    "saved_model_filename": None,
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "test_id": 0,  # the fold of training. one in [0, 1, 2, 3, 4] as the program ONLY support five fold.
    "wrong_patient_id": [],  # ['xwqg-A00085', 'xwqg-A00121', 'xwqg-B00027', 'xwqg-B00034'],
    "best_f1": None,  # initially the "best_f1" is None
    "init_features": 16,  # see source_codes.models.Unet_2D.U_net_Model.py
    "beta": 1.25,  # number times recall is more important than precision.
    "target_performance": 0.85,
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


def modify_params(direction='X', test_id=0):
    global TRAIN_DATA_DIR, WEIGHT_DIR, CHECKPOINT_DIR
    if not TRAIN_DATA_DIR[-1] == '/':
        TRAIN_DATA_DIR = TRAIN_DATA_DIR + '/'
    if not WEIGHT_DIR[-1] == '/':
        WEIGHT_DIR = WEIGHT_DIR + '/'
    if not CHECKPOINT_DIR[-1] == '/':
        CHECKPOINT_DIR = CHECKPOINT_DIR + '/'
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


def training_one_test_id_one_direction(test_id, direction):
    modify_params(direction, test_id)
    print('directing:', direction, 'test_id', test_id)
    run_model.training(params)


def training_one_test_id(test_id):
    for direction in ['Z', 'Y', 'X']:
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


if __name__ == '__main__':
    """
    the first epoch will be slow if data set exceed GPU Ram, but other epochs will be much faster (about 5 times).
    """
    training_one_test_id(0)




