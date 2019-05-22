from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ray
from ray.tune import run_experiments
from ray import tune
from env import CarlaEnv, ENV_CONFIG
# from models import register_carla_model
from models_td3_vae import register_carla_model


from scenarios import TOWN2_STRAIGHT, TOWN2_ONE_CURVE, TOWN2_NAVIGATION
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

env_config = ENV_CONFIG.copy()
# update config
env_config.update({
    "verbose": False,
    "x_res": 128,
    "y_res": 128,
    "use_depth_camera": False,
    "discrete_actions": False,
    "server_map": "/Game/Maps/Town02",
    "scenarios": TOWN2_ONE_CURVE,
})
register_carla_model()

ray.init(redirect_output=True)
run_experiments({
    "carla": {
        "run": "DDPG",
        "env": CarlaEnv,
        "checkpoint_freq": 2,
        # "restore":"/home/gu/ray_results/carla/PPO_CarlaEnv_0_2019-05-20_00-05-05zvce33w7/checkpoint_1124/checkpoint-1124",
        "config": {
            "env_config": env_config,
            "model": {
                "custom_model": "carla",   # defined in model
                # "use_lstm": True,
                # "lstm_use_prev_action_reward": True,
                "custom_options": {
                    "image_shape": [
                        env_config["x_res"], env_config["y_res"], 7
                    ],
                }
            },
            "num_workers": 1,
            "twin_q": True,
            # "train_batch_size": 2000,
            # "sample_batch_size": 400, # Size of batches collected from each worker
            # "lambda": 0.95,
            # "clip_param": 0.2,
            # "num_sgd_iter": 30,
            # "vf_share_layers": True,
            # "lr": 0.0003,
            # "sgd_minibatch_size": 800,
        },
    },
})
# /tmp/ray/session_2019-05-15_20-15-57_14951/tmpcpypfea8.json
