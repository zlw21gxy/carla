from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers import xavier_initializer
import sys
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.misc import normc_initializer
from ray.rllib.models.model import Model
# tf.enable_eager_execution()

class CarlaModel(Model):
    """Carla model that can process the observation tuple.

    The architecture processes the image using convolutional layers, the
    metrics using fully connected layers, and then combines them with
    further fully connected layers.
    """

    def _build_layers_v2(self, input_dict, num_outputs, options):
        """{'obs': [ < tf.Tensor 'default/Reshape:0' shape = (?, 96, 96, 8) dtype = float32 >,   image space
                     < tf.Tensor 'default/Reshape_1:0' shape = (?, 5) dtype = float32 >,         discrete 5
                     < tf.Tensor 'default/Reshape_2:0' shape = (?, 2) dtype = float32 >],        box 2
            'prev_actions': <tf.Tensor 'default / action_1: 0' shape=(?, 2) dtype=float32>,
            'prev_rewards': <tf.Tensor 'default / prev_reward: 0' shape=(?,) dtype=float32>,
            'is_training': <tf.Tensor 'default / PlaceholderWithDefault: 0' shape=() dtype = bool >}"""

        convs = options.get("structure", [
            [32, [4, 4], 3],
            [48, [4, 4], 2],
            [64, [3, 3], 2],
            [72, [3, 3], 1],
            [1024, [8, 8], 1],
        ])

        hiddens = options.get("fcnet_hiddens", [700, 100])
        fcnet_activation = options.get("fcnet_activation", "elu")
        # print(options)
        if fcnet_activation == "tanh":
            activation = tf.nn.tanh
        elif fcnet_activation == "relu":
            activation = tf.nn.relu
        elif fcnet_activation == "elu":
            activation = tf.nn.elu

        vision_in = input_dict['obs'][0]
        metrics_in = tf.concat([input_dict['obs'][1], input_dict['obs'][2]], axis=-1)

        with tf.name_scope("carla_vision"):
            for i, (out_size, kernel, stride) in enumerate(convs[:-1], 1):
                vision_in = slim.conv2d(
                    vision_in,
                    out_size,
                    kernel,
                    stride,
                    scope="conv{}".format(i))
            out_size, kernel, stride = convs[-1]
            vision_in = slim.conv2d(
                vision_in,
                out_size,
                kernel,
                stride,
                padding="VALID",
                scope="conv_out")
            vision_in = tf.squeeze(vision_in, [1, 2])

        # Setup metrics layer
        with tf.name_scope("carla_metrics"):
            metrics_in = slim.fully_connected(
                metrics_in,
                90,
                weights_initializer=xavier_initializer(),
                activation_fn=activation,
                scope="metrics_out")

        with tf.name_scope("carla_out"):
            i = 1
            last_layer = tf.concat([vision_in, metrics_in], axis=1)
            print("Shape of concatenated out is", last_layer.shape)
            for size in hiddens:
                last_layer = slim.fully_connected(
                    last_layer,
                    size,
                    weights_initializer=xavier_initializer(),
                    activation_fn=activation,
                    scope="fc{}".format(i))
                i += 1
            output = slim.fully_connected(
                last_layer,
                num_outputs,
                weights_initializer=normc_initializer(0.01),
                activation_fn=None,
                scope="fc_out")

        return output, last_layer


def register_carla_model():
    ModelCatalog.register_custom_model("carla", CarlaModel)
