import torch

from DVS128_Gesture.Networks.RM_SNN_Network import create_net
from DVS128_Gesture.RM.Config import configs
from DVS128_Gesture.utils.dataset import create_data
from DVS128_Gesture.utils.process import process
from DVS128_Gesture.utils.save import save_csv
from util.util import get_parameter_number


def main(i, dt, T, rate_t, rate_c, reserve=True):
    config = configs()

    config.c_sparsity_ratio = rate_c
    config.t_sparsity_ratio = rate_t

    config.attention = "TCR"

    config.reserve_coefficient = reserve
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(config.device)

    config.device_ids = range(torch.cuda.device_count())
    print(config.device_ids)

    config.dt = dt
    config.T = T
    config.modelPath = config.modelPath + config.attention
    config.name = (
        config.attention
        + "_SNN_DVS128_Gesture_dt="
        + str(config.dt)
        + "ms"
        + "_T="
        + str(config.T)
        + "_rate(c)="
        + str(1.0 - config.c_sparsity_ratio)
        + "_rate(t)="
        + str(1.0 - config.t_sparsity_ratio)
        + "_num="
        + str(i)
    )
    config.modelNames = config.name + ".t7"
    config.recordNames = config.name + ".csv"

    print(config)

    create_net(config=config)

    print(config.model)

    print(get_parameter_number(config.model))

    create_data(config=config)

    process(config=config)

    print("best acc:", config.best_acc, "best_epoch:", config.best_epoch)

    save_csv(config=config)
