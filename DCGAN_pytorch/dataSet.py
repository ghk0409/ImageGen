import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from util import *


class DataSet(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, task=None, opts=None):
        self.data_dir = data_dir
        self.transform = transform
        self.task = task
        self.opts = opts
        # input을 네트워크에 올리기 위해 텐서로 변환하는 함수
        self.to_tensor = ToTensor()

        data_list = os.listdir(self.data_dir)
        data_list = [f for f in data_list if f.endswith('jpg') | f.endswith('jpeg') | f.endswith('png')]

        data_list.sort()
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    # iterator 만들기
    def __getitem__(self, index):
        img = plt.imread(os.path.join(self.data_dir, self.data_list[index]))
        img_shape = img.shape

        if img.ndim == 2:
            img = img[:, :, np.newaxis]

        if img.dtype == np.uint8:
            img = img / 255.0

        data = {'label': img}

        # transform이 정의됐다면 변환 수행
        if self.transform:
            data = self.transform(data)

        data = self.to_tensor(data)

        return data

# transform 구현 클래스1
class ToTensor(object):
    def __call__(self, data):
        for key, value in data.items():
            value = value.transpose((2, 0, 1)).astype(np.float32)
            data[key] = torch.from_numpy(value)

        return data

# transform 구현 클래스2
class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        for key, value in data.items():
            data[key] = (value - self.mean) / self.std

        return data

# transform 구현 클래스3
# DCGAN에 사용할 image data가 DCGAN 모델의 generator output 크기와 맞지 않을 때 사용
class Resize(object):
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, data):
        for key, value in data.items():
            data[key] = resize(value, output_shape=(self.shape[0], self.shape[1], self.shape[2]))

        return data