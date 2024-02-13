import torch

from DVS128_Gait_Day.data_process.DVS128_Gait_Dataset import DVS128GaitDataset


def create_data(config):
    # Data set
    if config.onlyTest == False:
        config.train_dataset = DVS128GaitDataset(
            config.savePath,
            train=True,
            is_train_Enhanced=config.is_train_Enhanced,
            ds=config.ds,
            dt=config.dt * 1000,
            T=config.T,
            is_spike=config.is_spike,
            #             interval_scaling = config.interval_scaling,
        )

    config.test_dataset = DVS128GaitDataset(
        config.savePath,
        train=False,
        ds=config.ds,
        dt=config.dt * 1000,
        T=config.T,
        clips=config.clip,
        is_spike=config.is_spike,
        #         interval_scaling = config.interval_scaling,
    )
    # Data loader
    if config.onlyTest == False:
        #         sampler = torch.utils.data.distributed.DistributedSampler(config.train_dataset)
        config.train_loader = torch.utils.data.DataLoader(
            config.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=config.drop_last,
            num_workers=config.num_work,
            pin_memory=config.pip_memory,
            #             sampler = sampler
        )
    config.test_loader = torch.utils.data.DataLoader(
        config.test_dataset,
        batch_size=config.batch_size_test,
        shuffle=False,
        drop_last=config.drop_last,
        num_workers=config.num_work,
        pin_memory=config.pip_memory,
    )
