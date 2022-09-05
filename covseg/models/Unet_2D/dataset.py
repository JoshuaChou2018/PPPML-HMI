import numpy as np
import os
import torch
import torchvision.transforms
import glob


class RandomFlip(object):
    def __init__(self, dict_keys=["image", "label"]):
        self.dict_keys = dict_keys

    def __call__(self, sample):
        flag = np.random.rand() > 0.5
        transformed = [RandomFlip.flip_or_not(sample[k], flag) for k in self.dict_keys]
        return dict(zip(self.dict_keys, transformed))

    @staticmethod
    def flip_or_not(ts, flag):
        if flag:
            return torch.flip(ts, (1,))
        else:
            return ts


class RandomRotate(object):
    def __init__(self, dict_keys=["image", "label"]):
        self.dict_keys = dict_keys

    def __call__(self, sample):
        rot = np.random.randint(4)
        transformed = [torch.rot90(sample[k], rot, (1, 2)) for k in self.dict_keys]
        return dict(zip(self.dict_keys, transformed))


class ToTensor(object):
    def __init__(self, dict_keys=["image", "label"]):
        self.dict_keys = dict_keys

    def __call__(self, sample):
        transformed = [torch.from_numpy(sample[k].transpose([2, 0, 1]))
                       for k in self.dict_keys]
        return dict(zip(self.dict_keys, transformed))


class COVID19Dataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, pattern_or_list="*.npy", transform=None, filter_dict=None, channels=5):
        self.image_dir = image_dir
        if type(pattern_or_list) == str:
            image_files = [os.path.basename(f) for f in glob.glob(os.path.join(image_dir, pattern_or_list))]
        elif type(pattern_or_list) == list:
            image_files = pattern_or_list
        else:
            image_files = None
            assert image_files is not None
        files2 = []
        if filter_dict:
            assert type(filter_dict) == dict
            for fn in image_files:
                fields = os.path.splitext(fn)[0].split('_')
                include = True
                for k in filter_dict:
                    if fields[k] not in filter_dict[k]:
                        include = False
                        break
                if include:
                    files2.append(fn)
        else:
            files2 = image_files
        self.image_files = np.array(sorted(files2)).astype(np.string_)
        self.transform = transform
        self.channels = channels

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        assert 0 <= idx < len(self)
        arr = np.load(os.path.join(self.image_dir, self.image_files[idx].decode('utf-8')))
        image = arr[:, :, :self.channels]
        label = arr[:, :, self.channels:]
        sample = {"image": image, "label": label}

        if self.transform:
            return self.transform(sample)
        else:
            return sample


class RandomFlipWithWeight(object):
    def __init__(self, dict_keys=("image", "label", "weight")):
        self.dict_keys = dict_keys

    def __call__(self, sample):
        flag = np.random.rand() > 0.5
        transformed = [RandomFlip.flip_or_not(sample[k], flag) for k in self.dict_keys]
        return dict(zip(self.dict_keys, transformed))

    @staticmethod
    def flip_or_not(ts, flag):
        if flag:
            return torch.flip(ts, (1,))
        else:
            return ts


class RandomRotateWithWeight(object):
    def __init__(self, dict_keys=("image", "label", "weight")):
        self.dict_keys = dict_keys

    def __call__(self, sample):
        rot = np.random.randint(4)
        transformed = [torch.rot90(sample[k], rot, (1, 2)) for k in self.dict_keys]
        return dict(zip(self.dict_keys, transformed))


class ToTensorWithWeight(object):
    def __init__(self, dict_keys=("image", "label", "weight")):
        self.dict_keys = dict_keys

    def __call__(self, sample):
        transformed = [torch.from_numpy(sample[k].transpose([2, 0, 1]))
                       for k in self.dict_keys]
        return dict(zip(self.dict_keys, transformed))


class WeightedTissueDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir,
                 weight_dir,
                 image_pattern_or_list="*.npy",
                 transform=None,
                 channels=5,
                 mode='train',  # mode can be 'test' or 'train'
                 test_id=0,  # use patient_id % 5 == test_id as the test patients
                 wrong_patient_id=None,  # some patients have wrong label.
                 default_weight=True  # do not use the balance weight (like you don't have it)
                 ):
        self.image_dir = image_dir
        if not default_weight:
            self.weight_dir = weight_dir
        image_files = None
        if type(image_pattern_or_list) == str:
            image_files = [os.path.basename(f) for f in glob.glob(os.path.join(image_dir, image_pattern_or_list))]
        elif type(image_pattern_or_list) == list:
            image_files = image_pattern_or_list
        assert image_files is not None
        print(default_weight)
        #print(image_files) #['X_133_xgfy-A000068_2020-01-29.npy', 'X_31_xgfy-A000068_2020-01-29.npy']

        if not default_weight:
            print("feature-enhanced weight is activated!!!")
            weight_fn_list_all = os.listdir(weight_dir)
            if wrong_patient_id is not None:
                for fn in weight_fn_list_all:
                    patient_id = fn.split('_')[3]
                    if patient_id in wrong_patient_id:
                        weight_fn_list_all.remove(fn)
                        print("removed wrong scan:", fn)
            if mode == 'train':
                # sometimes we have sample but cannot calculate its weight,
                # so here use weight file names to determine the training samples:
                # weight files with name: "weight_" + training_sample_file_name
                weight_fn_list = [fn for fn in weight_fn_list_all if not int(fn.split('_')[3][-1]) % 5 == test_id]
            else:
                weight_fn_list = [fn for fn in weight_fn_list_all if int(fn.split('_')[3][-1]) % 5 == test_id]

            files2 = [fn for fn in image_files if 'weights_' + fn in weight_fn_list]

            image_files_filtered = sorted(files2)
            self.image_files = np.array(image_files_filtered).astype(np.string_)
            self.weight_files = np.array(['weights_' + image_fn for image_fn in image_files_filtered]).astype(np.string_)
            print("# all image files:", len(image_files), "# all weight files in weight_dir:",
                  len(weight_fn_list), "# image files with weight", len(self.image_files))
        else:
            print("feature enhanced weight is defaulted!!!")
            print("# all image files:", len(image_files))
            if mode == 'train':
                image_files_filtered = [fn for fn in image_files if not int(fn.split('_')[2][-1]) % 5 == test_id]
                print("# training image files:", len(image_files_filtered))
            else:
                image_files_filtered = [fn for fn in image_files if int(fn.split('_')[2][-1]) % 5 == test_id]
                print("# testing image files:", len(image_files_filtered))
            self.image_files = np.array(image_files_filtered).astype(np.string_)
            self.weight_ones = None

        print(self.image_files)
        self.transform = transform
        self.channels = channels
        self.default_weight = default_weight

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        assert 0 <= idx < len(self)
        arr = np.load(os.path.join(self.image_dir, self.image_files[idx].decode('utf-8')))
        image = arr[:, :, :self.channels]
        label = arr[:, :, self.channels:]
        if not self.default_weight:
            weight = np.load(os.path.join(self.weight_dir, self.weight_files[idx].decode('utf-8')))
        else:
            if self.weight_ones is None:
                self.weight_ones = np.ones(np.shape(label), 'float32')
            weight = self.weight_ones
        sample = {"image": image, "label": label, 'weight': weight}
        if self.transform:
            return self.transform(sample)
        else:
            return sample
