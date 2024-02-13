import functools
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


def make_dataset(root_path, train, T, txt_path):
    if train == True:
        text_path = os.path.join(txt_path, "train_label.txt")
    else:
        text_path = os.path.join(txt_path, "test_label.txt")
    with open(text_path, encoding="utf-8") as file:
        connect = file.readlines()
    # random.shuffle(connect)
    data = []
    label = []
    for line in connect:
        data_path, _, label_ = line.split(" ")
        assert int(label_) >= 0
        line_ = os.path.join(root_path, data_path)
        # if os.path.exists(line_):
        for p in os.listdir(line_):
            data.append(os.path.join(line_, p))
            label.append(int(label_))
    assert len(data) == len(label)

    whole_data = list(zip(data, label))
    random.shuffle(whole_data)
    data, label = zip(*whole_data)

    return data, label


def video_loader(
    video_dir_path, frame_indices, datanames, start_time, img_size, img_tranform
):
    video = torch.zeros((frame_indices, 2, img_size, img_size))
    tensor_03 = torch.ones(1, img_size, img_size) * 0.3
    if datanames.__len__() >= frame_indices:
        for i in range(frame_indices):
            image_path = os.path.join(video_dir_path, datanames[start_time + i])
            if os.path.exists(image_path):
                transform_tensor = img_tranform(Image.open(image_path).convert("L"))
                # 0.12 0.78 ; 0.2, 0.9
                video[i][0] = torch.where(transform_tensor < 0.12, 1.0, 0.0)
                transform_tensor_1 = torch.where(
                    transform_tensor > 0.78, tensor_03, transform_tensor
                )
                video[i][1] = torch.where(transform_tensor_1 > 0.3, 1.0, 0.0)
            else:
                return video
    else:
        for i in range(datanames.__len__()):
            image_path = os.path.join(video_dir_path, datanames[start_time + i])
            if os.path.exists(image_path):
                transform_tensor = img_tranform(Image.open(image_path).convert("L"))
                video[i][0] = torch.where(transform_tensor < 0.12, 1.0, 0.0)
                transform_tensor_1 = torch.where(
                    transform_tensor > 0.78, tensor_03, transform_tensor
                )
                video[i][1] = torch.where(transform_tensor_1 > 0.3, 1.0, 0.0)
            else:
                return video
    return video


def get_default_video_loader():
    return functools.partial(video_loader)


class HARDVS_dataset(data.Dataset):
    def __init__(
        self,
        root_path,
        train,
        img_size,
        T,
        txt_path,
        pic_tranform,
        get_loader=get_default_video_loader,
    ):
        self.T = T
        self.data, self.label = make_dataset(root_path, train, T, txt_path)
        self.loader = get_loader()
        self.img_transform = pic_tranform
        self.img_size = img_size

    def __getitem__(self, index):
        video = []
        data_path = self.data[index]
        label = self.label[index]
        datanames = os.listdir(data_path)
        pic_len = datanames.__len__()

        if pic_len < self.T:
            start_time = 0
            clips = self.loader(
                data_path,
                self.T,
                datanames,
                start_time,
                self.img_size,
                self.img_transform,
            )
            label_tensor = torch.tensor(label)
        else:
            start_time = random.randint(0, pic_len - self.T)
            clips = self.loader(
                data_path,
                self.T,
                datanames,
                start_time,
                self.img_size,
                self.img_transform,
            )
            label_tensor = torch.tensor(label)

        return clips, label_tensor

    def __len__(self):
        return len(self.data)


class EventMix:
    def __init__(self, sensor_size, T=8, num_classes=100, mode="distance"):
        h, w, _ = sensor_size
        self.sensor_size = (h, w)
        self.T = T
        self.num_classes = num_classes
        mask = self._gen_mask()
        self.mask = torch.from_numpy(mask.reshape((T, 1, h, w)))
        self.mode = mode

    def mix(self, frames, labels):
        # frames, should be N, T, C, H, W && N > 1
        # labels, should be N, 1          && N > 1
        frames_rolled = frames.roll(1, 0)

        # -----------------------------
        # Based on the number of events
        # -----------------------------
        self.mask = self.mask.to(frames.device)
        if self.mode == "events":
            sum_A = torch.mul(self.mask, frames).sum() / frames.sum()
            sum_B = torch.mul(1 - self.mask, frames_rolled).sum() / frames_rolled.sum()
            alpha = sum_A / (sum_A + sum_B)
        elif self.mode == "distance":
            # ------------------------------
            # Based on the relative distance
            # ------------------------------
            # print(frames.shape, self.mask.shape)
            x_mean = F.adaptive_avg_pool2d(
                frames.mul_(self.mask) + frames_rolled.mul_(1 - self.mask), 1
            )
            frames_pooled = F.adaptive_avg_pool2d(frames, 1)
            frames_rolled_pooled = F.adaptive_avg_pool2d(frames_rolled, 1)

            distance = nn.MSELoss(reduction="sum")
            sum_A = distance(frames_pooled, x_mean).item() ** 2
            sum_B = distance(frames_rolled_pooled, x_mean).item() ** 2
            alpha = sum_B / (sum_A + sum_B)
        # print(alpha)

        lambda_param = float(torch._sample_dirichlet(torch.tensor([alpha, alpha]))[0])

        # labels mix
        labels_onehot = F.one_hot(labels, num_classes=self.num_classes).to(
            dtype=frames.dtype
        )
        labels_rolled = labels_onehot.roll(1, 0)
        labels_rolled.mul_(1.0 - lambda_param)
        labels_onehot.mul_(lambda_param).add_(labels_rolled)

        # frames mix
        frames_rolled.mul_(1.0 - lambda_param)
        frames.mul_(lambda_param).add_(frames_rolled)

        return frames, labels_onehot

    def _gen_mask(self):
        mean = np.random.randn(self.T)
        cov = np.random.randn(self.T, 1)
        cov = 0.5 * cov * cov.T
        mask = np.random.multivariate_normal(mean, cov, self.sensor_size).reshape(
            -1,
        )
        lam = int(np.random.beta(1, 1) * mask.size)
        threshold = mask[np.argpartition(mask, kth=-lam)[-lam]]
        for i, value in enumerate(mask):
            if value < threshold:
                mask[i] = 1
            else:
                mask[i] = 0
        return mask


if __name__ == "__main__":
    dataset = HARDVS_dataset(
        root_path="./rawframes",
        train=True,
        T=8,
        txt_path="./HARDVS_train-val-test-split-txt-file",
        pic_tranform=transforms.Compose([transforms.ToTensor()]),
    )
    print(len(dataset))
    data, label = dataset.__getitem__(1)
    print(data.shape, label)
