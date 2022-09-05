import numpy as np
import pandas as pd
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)


def load_in_csv(target_column=0, csv_path=None):
    if csv_path is None:
        from sklearn.datasets import load_boston
        return load_boston()
    else:
        data_set_dict = {'filename': csv_path}
        csv_file = pd.read_csv(csv_path)
        data_array = np.array(csv_file)
        target = data_array[:, target_column]
        data_set_dict['target'] = target
        column_list = list(csv_file.columns)
        target_name = column_list[target_column]
        column_list.remove(target_name)
        print(column_list)
        data_set_dict['feature_names'] = column_list
        data_array = np.concatenate([data_array[:, 0:target_column], data_array[:, target_column + 1::]], axis=1)
        data_set_dict['data'] = data_array
        print("we have (sample, features):", np.shape(data_array))
        assert len(target) == len(data_array)
        return data_set_dict


def convert_csv_to_numpy(csv_path):
    # note, the first line will be deleted!
    csv_file = pd.read_csv(csv_path)
    csv_file = np.array(csv_file)
    return csv_file


def read_in_text(txt_path):
    f = open(txt_path, mode='r')
    return_list = f.readlines()
    return return_list


def get_important_feature():  # for the '/home/zhoul0a/Desktop/example_files/data_prognosis_all.csv'
    subjective = [2, 3]  # 2 is the difficulty of breath; 3 is the SGRQ
    objective = [4, ]


if __name__ == '__main__':
    line_list = read_in_text('/home/zhoul0a/Desktop/prognosis_project/threshold.txt')
    value_list = []
    for line in line_list:
        value_list.append(float(line.split(' ')[3][1:-1]))
    import numpy as np
    print(np.std(value_list))

    median_lesion_severity_list = []
    data_list = read_in_text('/home/zhoul0a/Desktop/prognosis_project/transfer/median_lesion_severity')

    for data in data_list:
        median_lesion_severity_list.append(float(data[:-1]))
    print(np.std(median_lesion_severity_list))
    print(np.mean(median_lesion_severity_list))


