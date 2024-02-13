import torch
from spikingjelly.clock_driven import functional
from tqdm import tqdm


def train(config):

    config.train_loss = 0
    config.train_correct = 0
    bar_train = tqdm(total=len(config.train_loader))
    for _, (input, labels) in enumerate(config.train_loader):
        functional.reset_net(config.model)

        config.model.zero_grad()
        config.optimizer.zero_grad()

        input = input.float().to(config.device)
        if len(labels.shape) == 2:
            labels = labels.float().to(config.device)
        else:
            labels = labels[:, 1, :].float().to(config.device)

        outputs, lasso = config.model(input)

        if hasattr(config, "fbs") or config.gate:
            loss = config.criterion(outputs, labels) + config.lasso_lambda * lasso.sum()
        else:
            loss = config.criterion(outputs, labels)

        _, predicted = torch.max(outputs.data, 1)
        _, labelTest = torch.max(labels.data, 1)
        config.train_correct += (predicted == labelTest).sum().item()

        config.train_loss += loss.item()

        loss.backward()

        config.optimizer.step()

        bar_train.update()
        bar_train.set_description(
            "Train:Epoch[%d/%d]" % (config.epoch + 1, config.num_epochs)
        )
        bar_train.set_postfix(Loss=loss.item())

    bar_train.close()
