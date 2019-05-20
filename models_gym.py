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
from collections import OrderedDict

import gym
import tensorflow as tf

from ray.rllib.models.misc import linear, normc_initializer
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.utils.annotations import PublicAPI

import tensorflow as tf

import resnet_utils


class GymModel(Model):
    """Carla model that can process the observation tuple.

    The architecture processes the image using convolutional layers, the
    metrics using fully connected layers, and then combines them with
    further fully connected layers.
    """

    def _build_layers_v2(self, input_dict, num_outputs, options):
        # {'prev_rewards': < tf.Tensor'default/prev_reward:0'shape = (?,)dtype = float32 >,
        # 'prev_actions': < tf.Tensor'default/action_1:0'shape = (?,)dtype = int64 >,
        # 'obs': [ < tf.Tensor'default/Reshape:0'shape = (?,)dtype = float32 >,
        # < tf.Tensor'default/Reshape_1:0'shape = (?, 4)dtype = float32 >],
        # 'is_training': < tf.Tensor'default/PlaceholderWithDefault:0'shape = ()dtype = bool >}

        convs = options.get("structure", [
            [48, [4, 4], 3],
            [64, [4, 4], 2],
            [72, [3, 3], 2],
            [128, [3, 3], 1],
            # [256, [3, 3], 1],
            [512, [8, 8], 1],
        ])

        hiddens = options.get("fcnet_hiddens", [512, 256])
        fcnet_activation = options.get("fcnet_activation", "elu")
        print(">>>>>>>>>", num_outputs)
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
                # vision_in = tf.layers.batch_normalization(
                #     vision_in, training=input_dict["is_training"])

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
            # last_layer = tf.concat([vision_in, metrics_in], axis=1)
            last_layer = vision_in
            # last_layer = metrics_in
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
        # return tf.random.uniform([-1, num_outputs]), last_layer

    def value_function(self):
        """Builds the value function output.

        This method can be overridden to customize the implementation of the
        value function (e.g., not sharing hidden layers).

        Returns:
            Tensor of size [BATCH_SIZE] for the value function.
        """
        hiddens = [400, 300]
        last_layer = self.last_layer
        with tf.name_scope("carla_out"):
            i = 1
            for size in hiddens:
                last_layer = slim.fully_connected(
                    last_layer,
                    size,
                    weights_initializer=xavier_initializer(),
                    activation_fn=tf.nn.elu,
                    scope="value_function{}".format(i))
                i += 1
            output = slim.fully_connected(
                last_layer,
                1,
                weights_initializer=normc_initializer(0.01),
                activation_fn=None,
                scope="value_out")
        # return output
        return tf.reshape(output, [-1])

def register_gym_model():
    ModelCatalog.register_custom_model("gym", GymModel)