# Spike-based dynamic computing with asynchronous neuromorphic chip

#### Installation

1. Python 3.7.4
2. PyTorch 1.7.1
3. numpy 1.19.2
4. spikingjelly 0.0.0.0.12
5. nni 2.5

Please see `requirements.txt` for more requirements.

### **Instructions**

#### 1. DVS128 Gesture

1. Download [DVS128 Gesture](https://www.research.ibm.com/dvsgesture/) and put the downloaded dataset to `/DVS128_Gesture/data`, then run `/DVS128_Gesture/data/DVS_Gesture`.py.

```
/DVS128_Gesture/
│  ├── /data/
│  │  ├── DVS_Gesture.py
│  │  └── DvsGesture.tar.gz
```

2. Change the values of T and dt in `/DVS128_Gesture/RM/Config.py` , `/DVS128_Gesture/CRM/Config.py` or `/DVS128_Gesture/TRM/Config.py` then run the tasks in `/DVS128` Gesture.

eg:

```
python RM_main.py
```

3. View the results in `/DVS128_Gesture/RM/Result/` 、 `/DVS128_Gesture/TRM/Result/` or `/DVS128_Gesture/CRM/Result/`.

#### 2. DVS128 Gait Day

1. Download [DVS128 Gait Day](https://github.com/zhangxiann/TPAMI_Gait_Identification) and put the downloaded dataset to `/DVS128_Gait_Day/data`, then run `/DVS128_Gait_Day/data/DVS128_Gait.py`.

```
../data/
├── test
│   ├── test_data.npy
│   └── test_target.npy
├── train
│   ├── train_data.npy
│   └── train_target.npy
```

2. Change the values of T and dt in `/DVS128_Gait_Day/RM/Config.py` ,`/DVS128_Gait_Day/TRM/Config.py` or `/DVS128_Gait_Day/CRM/Config.py` then run the tasks in `/DVS128_Gait_Day`.

eg:

```
python RM_main.py
```

3. View the results in `/DVS128_Gait_Day/RM/Result/` , `/DVS128_Gait_Day/CRM/ Result/` or `/DVS128_Gait_Day/TRM/Result/`.

#### 3. DVS128 Gait Night

> The training and validation steps for this dataset are almost the same as the DVS128 Gait Day.

1. Download [DVS128 Gait Night](https://github.com/zhangxiann/TPAMI_Gait_Identification) and put the downloaded dataset to `/DVS128_Gait_Night/data`, then run `/DVS128_Gait_Night/data/DVS128_Gait.py`.

```
../data/
├── test
│   ├── test_data.npy
│   └── test_target.npy
├── train
│   ├── train_data.npy
│   └── train_target.npy
```

2. Change the values of T and dt in `/DVS128_Gait_Night/RM/Config.py` ,`/DVS128_Gait_Night/TRM/Config.py` or `/DVS128_Gait_Night/CRM/Config.py` then run the tasks in `/DVS128_Gait_Night`.

eg:

```
python RM_main.py
```

3. View the results in `/DVS128_Gait_Night/RM/Result/` , `/DVS128_Gait_Night/CRM/Result/` or `/DVS128_Gait_Night/TRM/Result/`.

#### 4. HAR-DVS

1. Download [HAR-DVS](https://github.com/Event-AHU/HARDVS) and put the downloaded dataset to `/HAR-DVS/data`.

```
HAR-DVS/
│  ├── /data/
│  │  └── /rawframes/
|  |  └── /HARDVS_train-val-test-split-txt-file/
```

2. We offer a number of attention modules with different advantages and provide the `search.py`, which is used to find the most suitable attention module for the HAR-DVS and how it should be placed in SEW-ResNet.

eg:

```
python search.py
```

3. View the results at localhost:8080

#### 5. Extra

1. `/module/RM.py` defines RM,TRM,CRM layer and `/module/LIF.py`, `LIF_Module.py` defines LIF module.
2. Explain again if the parameters in the `Config.py` are different from the paper name

```
conifg.c_sparsity_ratio = 1 - beat_c
config.t_sparsity_ratio = 1 - beat_t
```

3. If it produces an error like

```
RuntimeError: cuDNN error: CUDNN_STATUS_NOT_SUPPORTED. This error may appear if you passed in a non-contiguous input.
```

, please reduce the batch-size selected for this experimence.

> see https://github.com/pytorch/pytorch/issues/32564#issuecomment-635062872 for more info.
