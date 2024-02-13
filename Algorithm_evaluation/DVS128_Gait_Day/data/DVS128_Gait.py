import os
import sys

rootPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(rootPath)[0]
rootPath = os.path.split(rootPath)[0]
sys.path.append(rootPath)

import os
import struct
import tarfile

import h5py
import numpy as np


def DVS128_Gait_txt_to_npy(path, save_path):
    save_path = os.path.join(save_path, "npy")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # train
    train_path = os.path.join(path, "train")
    save_train_path = os.path.join(save_path, "train")

    if not os.path.exists(save_train_path):
        os.makedirs(save_train_path)

    for root, dirs, files in os.walk(train_path):
        break

    print("root", root)  # 当前目录路径
    print("dirs", dirs)  # 当前路径下所有子目录

    train_data = []
    train_target = []
    for dir in dirs:
        train_path_sub = os.path.join(root, dir)
        files = os.listdir(train_path_sub)
        print(dir)
        for file in files:
            print(file)
            train_path_file = os.path.join(train_path_sub, file)
            data = []
            with open(train_path_file) as f:
                for line in f.readlines():
                    temp = line.split()
                    data.append(np.array(temp).astype(np.int64))

            train_data.append(np.array(data))
            train_target.append(int(dir))

    save_train_path_file_data = os.path.join(save_train_path, "train_data.npy")
    save_train_path_file_target = os.path.join(save_train_path, "train_target.npy")

    np.save(save_train_path_file_data, np.array(train_data))
    np.save(save_train_path_file_target, np.array(train_target))

    # test
    test_path = os.path.join(path, "test")
    save_test_path = os.path.join(save_path, "test")

    if not os.path.exists(save_test_path):
        os.makedirs(save_test_path)

    for root, dirs, files in os.walk(test_path):
        break

    print("root", root)  # 当前目录路径
    print("dirs", dirs)  # 当前路径下所有子目录

    test_data = []
    test_target = []
    for dir in dirs:
        test_path_sub = os.path.join(root, dir)
        files = os.listdir(test_path_sub)
        print(dir)
        for file in files:
            print(file)
            test_path_file = os.path.join(test_path_sub, file)
            data = []
            with open(test_path_file) as f:
                for line in f.readlines():
                    temp = line.split()
                    data.append(np.array(temp).astype(np.int64))

            test_data.append(np.array(data))
            test_target.append(int(dir))

    save_test_path_file_data = os.path.join(save_test_path, "test_data.npy")
    save_test_path_file_target = os.path.join(save_test_path, "test_target.npy")

    np.save(save_test_path_file_data, np.array(test_data))
    np.save(save_test_path_file_target, np.array(test_target))


if __name__ == "__main__":
    DVS128_Gait_txt_to_npy(path="./", save_path="./DVS128-Gait-Day/")
