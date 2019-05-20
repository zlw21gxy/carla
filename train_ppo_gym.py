from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ray
from ray.tune import run_experiments

from gym_env import GYMEnv
# from models import register_carla_model
from models_gym import register_gym_model

register_gym_model()

ray.init(redirect_output=True)
run_experiments({
    "gym": {
        "run": "PPO",
        "env": GYMEnv,
        "checkpoint_freq": 5,
        # "use_lstm": True,
        # "restore":"/home/gu/ray_results/carla/PPO_CarlaEnv_0_2019-05-15_12-26-41z6llpae8/checkpoint_1030/checkpoint-1030",
        "config": {
            # "env_config": env_config,
            "model": {
                "custom_model": "gym",   # defined in model
                "use_lstm": True,
                # "lstm_use_prev_action_reward": True,
                # "custom_options": {
                #     "image_shape": [
                #         env_config["x_res"], env_config["y_res"], 7
                #     ],
                # }
            },
            "num_workers": 12,
            "train_batch_size": 2000,
            "sample_batch_size": 200,
            "lambda": 0.95,
            "clip_param": 0.2,
            "num_sgd_iter": 64,
            "vf_share_layers": True,
            "lr": 0.0001,
            "sgd_minibatch_size": 64,
            "num_gpus": 2,
        },
    },
})