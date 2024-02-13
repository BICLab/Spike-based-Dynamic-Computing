import argparse
import datetime
import math
import os
import random
import sys
import time

import nni
import numpy as np
import sew_resnet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from datasets.HAR_DVS import EventMix, HARDVS_dataset
from spikingjelly.clock_driven import functional, neuron, surrogate
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes=300, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        # targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)  # one_hot
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


class RASampler(torch.utils.data.Sampler):
    """Sampler that restricts data loading to a subset of the dataset for distributed,
    with repeated augmentation.
    It ensures that different each augmented version of a sample will be visible to a
    different process (GPU)
    Heavily based on torch.utils.data.DistributedSampler
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 3.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        # self.num_selected_samples = int(math.ceil(len(self.dataset) / self.num_replicas))
        self.num_selected_samples = int(
            math.floor(len(self.dataset) // 256 * 256 / self.num_replicas)
        )
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices = [ele for ele in indices for i in range(3)]
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices[: self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def main():
    parser = argparse.ArgumentParser(description="Classify DVS Gesture")
    parser.add_argument("-T", default=8, type=int, help="simulating time-steps")
    parser.add_argument("-b", default=128, type=int, help="batch size")
    parser.add_argument("-b_test", default=128, type=int, help="batch size")
    parser.add_argument("-attn", default="no", help="batch size")
    parser.add_argument("-attn_where", default="0000", type=str, help="attn_where")
    parser.add_argument(
        "-epochs",
        default=200,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "-j",
        default=8,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "-data_dir",
        default="../data/rawframes",
        type=str,
        help="root dir of DVS Gesture dataset",
    )
    parser.add_argument(
        "-text_dir",
        default="../data/HARDVS_train-val-test-split-txt-file",
        type=str,
        help="root dir of DVS Gesture dataset",
    )
    parser.add_argument(
        "-out-dir",
        type=str,
        default="./logs",
        help="root dir for saving logs and checkpoint",
    )
    parser.add_argument("-resume", type=str, help="resume from the checkpoint path")
    parser.add_argument(
        "-amp", action="store_true", help="automatic mixed precision training"
    )
    parser.add_argument("-cupy", action="store_true", help="use cupy backend")
    parser.add_argument(
        "-opt", default="adam", type=str, help="use which optimizer. SDG or Adam"
    )
    parser.add_argument("-momentum", default=0.9, type=float, help="momentum for SGD")
    parser.add_argument("-lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("-betas", default=[0.5, 0.999], help="learning rate")
    parser.add_argument("-eps", default=1e-8, type=float, help="learning rate")
    parser.add_argument("-weight_decay", default=1e-4, type=float, help="learning rate")
    parser.add_argument(
        "-train_correct", default=0, type=float, help="channels of CSNN"
    )
    parser.add_argument("-test_correct", default=0, type=float, help="channels of CSNN")
    parser.add_argument("-only_test", default=True, help="channels of CSNN")
    parser.add_argument("-seed", default=42, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    args = parser.parse_args()

    params = {
        "attn": "TCA_NN",
        "attn_where": "2222",
    }
    optimized_params = nni.get_next_parameter()
    params.update(optimized_params)
    print(params)

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    num_classes = 300

    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(args.local_rank)
    net = sew_resnet.sew_resnet18(
        num_classes=num_classes,
        T=args.T,
        attn=params["attn"],
        attn_where=params["attn_where"],
    )

    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = torch.nn.parallel.DistributedDataParallel(
        net.cuda(), device_ids=[args.local_rank], find_unused_parameters=False
    )

    functional.set_step_mode(net, "m")
    if args.cupy:
        functional.set_backend(net, "cupy", instance=neuron.LIFNode)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    transforms_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3)),
            transforms.RandomHorizontalFlip(0.5),
            # transforms.RandomVerticalFlip(0.5),
            # transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
        ]
    )
    transforms_test = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
    )
    train_set = HARDVS_dataset(
        root_path=args.data_dir,
        train=True,
        img_size=224,
        T=args.T,
        txt_path=args.text_dir,
        pic_tranform=transforms_train,
    )
    test_set = HARDVS_dataset(
        root_path=args.data_dir,
        train=False,
        img_size=224,
        T=args.T,
        txt_path=args.text_dir,
        pic_tranform=transforms_test,
    )
    train_sampler = RASampler(train_set)
    # test_sampler = RASampler(test_set)
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.b,
        shuffle=False,
        drop_last=True,
        num_workers=args.j,
        pin_memory=True,
        # sampler=DistributedSampler(train_set),
        sampler=train_sampler,
    )
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.b_test,
        shuffle=False,
        drop_last=True,
        num_workers=args.j,
        pin_memory=True,
        sampler=DistributedSampler(test_set),
        # sampler=test_sampler
    )
    mix = EventMix(sensor_size=(224, 224, 2), num_classes=num_classes, T=args.T)
    criterion = CrossEntropyLabelSmooth().cuda()

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    start_epoch = 0
    max_test_acc = -1

    optimizer = None
    if args.opt == "sgd":
        optimizer = torch.optim.SGD(
            net.parameters(), lr=args.lr, momentum=args.momentum
        )
    elif args.opt == "adam":
        optimizer = torch.optim.Adam(
            net.parameters(),
            lr=args.lr,
            betas=args.betas,
            eps=args.eps,
            weight_decay=args.weight_decay,
        )
    else:
        raise NotImplementedError(args.opt)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.epochs, T_mult=1, eta_min=1e-6, verbose=True
    )

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        net.load_state_dict(checkpoint["net"])
        max_test_acc = checkpoint["max_test_acc"]

    for epoch in range(start_epoch, args.epochs):
        if epoch == 50:
            break
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        args.train_correct = 0
        real_batch = 0
        train_sampler.set_epoch(epoch)
        # test_sampler.set_epoch(epoch)
        for iters, (frame, label) in enumerate(train_data_loader):
            functional.reset_net(net)

            optimizer.zero_grad()
            frame = frame.float().cuda(non_blocking=True)
            real_batch += frame.size(0)
            label = label.cuda(non_blocking=True)

            frame, label = mix.mix(frame, label)
            # label = F.one_hot(label, num_classes).float()

            frame = frame.transpose(
                0, 1
            ).contiguous()  # [N, T, C, H, W] -> [T, N, C, H, W]
            if scaler is not None:
                with amp.autocast():
                    out_fr, _ = net(frame)
                    loss = criterion(out_fr, label)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out_fr, _ = net(frame)
                loss = criterion(out_fr, label)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(out_fr.data, 1)
            _, truth = torch.max(label.data, 1)
            args.train_correct += (predicted == truth).sum().item()

            functional.reset_net(net)

        lr_scheduler.step(epoch)

        train_time = time.time()
        train_speed = real_batch / (train_time - start_time)
        train_loss /= len(train_data_loader)
        train_acc = 100 * float(args.train_correct) / real_batch
        # writer.add_scalar('train_loss', train_loss, epoch)
        # writer.add_scalar('train_acc', train_acc, epoch)

        net.eval()
        test_loss = 0
        test_acc = 0
        args.test_correct = 0
        real_batch = 0
        with torch.no_grad():
            for val_iters, (frame, label) in enumerate(test_data_loader):
                functional.reset_net(net)
                real_batch += frame.size(0)

                frame = frame.float().cuda(non_blocking=True)
                frame = frame.transpose(
                    0, 1
                ).contiguous()  # [N, T, C, H, W] -> [T, N, C, H, W]
                label = label.cuda(non_blocking=True)

                out_fr, _ = net(frame)
                loss = F.cross_entropy(out_fr, label)
                _, predicted = torch.max(out_fr.data, 1)
                args.test_correct += (predicted == label).sum().item()

                test_loss += loss.item()
                functional.reset_net(net)

        test_time = time.time()
        test_speed = real_batch / (test_time - train_time)
        test_loss /= len(test_data_loader)
        test_acc = 100 * float(args.test_correct) / real_batch
        # writer.add_scalar('test_loss', test_loss, epoch)
        # writer.add_scalar('test_acc', test_acc, epoch)

        save_max = False
        if test_acc >= max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {"net": net.state_dict(), "max_test_acc": max_test_acc}

        nni.report_intermediate_result(test_acc)

        # if save_max:
        #     torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))

        # torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))

        print(
            f"epoch = {epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}"
        )
        print(
            f"train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s"
        )
        print(
            f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n'
        )
    nni.report_final_result(max_test_acc)


if __name__ == "__main__":
    main()
