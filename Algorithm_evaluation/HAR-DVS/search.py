from pathlib import Path

import nni
from nni.experiment import Experiment

experiment = Experiment("local")
experiment.config.training_service.use_active_gpu = True

experiment.config.experiment_name = "HAR-DVS"

experiment.config.trial_command = "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch --master_port 12398 --nproc_per_node 8 nni_train.py -b 32 -b_test 32 -epochs 400 -lr 3e-4 -T 8"
experiment.config.trial_code_directory = Path(__file__).parent

experiment.config.search_space = {
    "attn": {
        "_type": "choice",
        "_value": [
            "HAM",
            "RM_SEW",
            "TCA_SEW",
            "RM_TCA_SEW",
            "RM_SEW_only_CA",
            "HTSA_sa",
        ],
    },
    "attn_where": {
        "_type": "choice",
        "_value": [
            "0011",
            "1100",
            "0022",
            "2200",
            "0001",
            "1000",
            "1111",
            "2222",
            "1122",
            "2211",
        ],
    },
}

experiment.config.tuner.name = "TPE"
experiment.config.assessor.name = "Medianstop"
experiment.config.tuner.class_args = {
    "optimize_mode": "maximize",
}
experiment.config.assessor.class_args = {
    "optimize_mode": "maximize",
    "start_step": 5,
}

experiment.config.max_trial_number = 40
experiment.config.trial_concurrency = 1

experiment.run(8080)
