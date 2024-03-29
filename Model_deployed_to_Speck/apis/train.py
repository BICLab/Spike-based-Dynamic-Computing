import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from spikingjelly.clock_driven import functional
import sinabs.layers as sl

def reset_state(model):
    for lyrs in model.modules():
        if isinstance(lyrs, sl.IAF):
            lyrs.reset_states(randomize=True)

def train(args):
    args.train_loss = 0
    args.train_correct = 0
    bar_train = tqdm(total=len(args.train_loader))

    for i, (input, labels) in enumerate(args.train_loader):
        
        if args.spike_mode == 'IFCell':
            functional.reset_net(args.model)
        elif args.spike_mode == 'sinabs':
            reset_state(args.model)

        args.optimizer.zero_grad()

        input = input.float().to(args.device)
        if len(labels.shape) == 2:
            labels = labels.float().to(args.device)
        elif len(labels.shape) == 1:
            labels = F.one_hot(labels, args.num_classes).float().to(args.device)
        else:
            labels = labels[:, 1, :].float().to(args.device)

        outputs, _ = args.model(input)

        _, label = torch.max(labels.data, 1)

        loss = args.criterion(outputs, label)

        _, predicted = torch.max(outputs.data, 1)
        _, labelTest = torch.max(labels.data, 1)
        args.train_correct += (predicted == labelTest).sum().item()

        args.train_loss += loss.item()

        loss.backward()

        args.optimizer.step()

        bar_train.update()
        bar_train.set_description("Train:Epoch[%d/%d]" % (args.epoch + 1, args.num_epochs))
        bar_train.set_postfix(Loss=loss.item())

    bar_train.close()
