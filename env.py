"""OpenAI gym environment for Carla. Run this file for a demo."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import atexit
import cv2
import os
import json
import random
import signal
import subprocess
import sys
import time
import traceback
from vae_unit import encode, create_vae, decode
import numpy as np
try:
    import scipy.misc
except Exception:
    pass

import gym
from gym.spaces import Box, Discrete, Tuple

from scenarios import DEFAULT_SCENARIO, LANE_KEEP, TOWN2_STRAIGHT, TOWN2_ONE_CURVE, TOWN2_CUSTOM, TOWN1_CUSTOM

import os
#
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# Set this where you want to save image outputs (or empty string to disable)
CARLA_OUT_PATH = os.environ.get("CARLA_OUT", os.path.expanduser("~/carla_out"))
if CARLA_OUT_PATH and not os.path.exists(CARLA_OUT_PATH):
    os.makedirs(CARLA_OUT_PATH)

# Set this to the path of your Carla binary
SERVER_BINARY = os.environ.get("CARLA_SERVER",
                               os.path.expanduser("/data/carla8/CarlaUE4.sh"))

# assert os.path.exists(SERVER_BINARY)
if "CARLA_PY_PATH" in os.environ:
    sys.path.append(os.path.expanduser(os.environ["CARLA_PY_PATH"]))
else:
    # TODO(ekl) switch this to the binary path once the planner is in master
    sys.path.append(os.path.expanduser("/data/carla8/PythonClient/"))

try:
    from carla.client import CarlaClient
    from carla.sensor import Camera
    from carla.settings import CarlaSettings
    from carla.planner.planner import Planner, REACH_GOAL, GO_STRAIGHT, \
        TURN_RIGHT, TURN_LEFT, LANE_FOLLOW
except Exception as e:
    print("Failed to import Carla python libs, try setting $CARLA_PY_PATH")
    raise e

# Carla planner commands
COMMANDS_ENUM = {
    REACH_GOAL: "REACH_GOAL",
    GO_STRAIGHT: "GO_STRAIGHT",
    TURN_RIGHT: "TURN_RIGHT",
    TURN_LEFT: "TURN_LEFT",
    LANE_FOLLOW: "LANE_FOLLOW",
}

# Mapping from string repr to one-hot encoding index to feed to the model
COMMAND_ORDINAL = {
    "REACH_GOAL": 0,
    "GO_STRAIGHT": 1,
    "TURN_RIGHT": 2,
    "TURN_LEFT": 3,
    "LANE_FOLLOW": 4,
}

# Number of retries if the server doesn't respond
RETRIES_ON_ERROR = 7

# Dummy Z coordinate to use when we only care about (x, y)
GROUND_Z = 0.22

# Default environment configuration
ENV_CONFIG = {
    "log_images": False,  # log images in _read_observation().
    "convert_images_to_video": False,  # convert log_images to videos. when "verbose" is True.
    "verbose": False,    # print measurement information; write out measurement json file
    "enable_planner": True,
    "framestack": 1,  # note: only [1, 2] currently supported
    "early_terminate_on_collision": True,
    "reward_function": "custom2",
    "render_x_res": 400,
    "render_y_res": 300,
    "x_res": 128,  # cv2.resize()
    "y_res": 128,  # cv2.resize()
    "server_map": "/Game/Maps/Town01",
    "scenarios": TOWN1_CUSTOM,  # [LANE_KEEP]
    "use_depth_camera": False,  # use depth instead of rgb.
    "discrete_actions": False,
    "squash_action_logits": False,
    "encode_measurement": True,  # encode measurement information into channel
    "use_seg": False,  # use segmentation camera
    "VAE": True,
    "SAC": True,
    "out_vae": True,
    "action_repeat": 2,
}

k = 0
DISCRETE_ACTIONS = {}
for i in [-0.5, 0, 0.7]:
    for j in [-0.9, -0.6, -0.3, 0, 0.3, 0.6, 0.9]:
        DISCRETE_ACTIONS[k] = [i, j]
        k += 1

live_carla_processes = set()  # Carla Server


def cleanup():
    print("Killing live carla processes", live_carla_processes)
    for pgid in live_carla_processes:
        os.killpg(pgid, signal.SIGKILL)


atexit.register(cleanup)


class CarlaEnv(gym.Env):
    def __init__(self, config=ENV_CONFIG, enable_autopilot = False):
        self.config = config
        self.enable_autopilot = enable_autopilot
        self.city = self.config["server_map"].split("/")[-1]
        if self.config["enable_planner"]:
            self.planner = Planner(self.city)

        # The Action Space
        if config["discrete_actions"]:
            self.action_space = Discrete(len(DISCRETE_ACTIONS))  # It will be transformed to continuous 2D action.
        else:
            self.action_space = Box(-1.0, 1.0, shape=(2, ), dtype=np.float32)   # 2D action.

        if config["use_depth_camera"]:
            image_space = Box(
                -1.0,
                1.0,
                shape=(config["y_res"], config["x_res"],
                       1 * config["framestack"]),
                dtype=np.float32)
        else:
            image_space = Box(
                0,
                255,
                shape=(config["y_res"], config["x_res"],
                       3 * config["framestack"]),
                dtype=np.uint8)
        # encode_measure ---> rgb + measurement_encode + pre_rgb #channel_type
        # 3 + 3 + 1 = 7 #channel_number
        if config["encode_measurement"]:
             image_space = Box(
                 0,
                 255,
                 shape=(config["y_res"], config["x_res"], 7),
                 dtype=np.float32)

        if config["use_seg"]:
            image_space = Box(
                0,
                255,
                shape=(config["y_res"], config["x_res"], 5),
                dtype=np.float32)
        # The Observation Space
        if config["VAE"]:
            image_space = Box(
                0,
                255,
                shape=(515,),
                dtype=np.float32)
        self.observation_space = Tuple(
            [
                image_space,
                Discrete(len(COMMANDS_ENUM)),  # next_command
                Box(-128.0, 128.0, shape=(2, ), dtype=np.float32)  # forward_speed, dist to goal
            ])

        # TODO(ekl) this isn't really a proper gym spec
        self._spec = lambda: None
        self._spec.id = "Carla-v0"

        self.server_port = None
        self.server_process = None
        self.client = None
        self.num_steps = 0
        self.total_reward = 0
        self.prev_measurement = None
        self.prev_image = None
        self.episode_id = None
        self.measurements_file = None
        self.weather = None
        self.scenario = None
        self.start_pos = None
        self.end_pos = None
        self.start_coord = None
        self.end_coord = None
        self.last_obs = None
        self.latent_dim = 256
        self.vae = create_vae(self.latent_dim, return_kl_loss_op=False)
        # filepath = "/home/gu/project/ppo/ppo_carla/models/carla_model/high_ld_256_beta_1_r_1_lr_0.0001.hdf5"
        filepath = "/home/gu/project/ppo/ppo_carla/models/carla_model/large_high_ld_256_beta_1.2_r_1_lr_5e-05_bc_128.hdf5"
        self.vae.load_weights(filepath)
        self.vae.trainable = False

    def init_server(self):
        print("Initializing new Carla server...")
        # Create a new server process and start the client.
        self.server_port = random.randint(1000, 60000)
        self.server_process = subprocess.Popen(
            [
                SERVER_BINARY, self.config["server_map"], "-windowed",
                "-ResX=400", "-ResY=300", "-carla-server", "-benchmark -fps=10", #: to run the simulation at a fixed time-step of 0.1 seconds
                "-carla-world-port={}".format(self.server_port)
            ],
            preexec_fn=os.setsid,
            stdout=open(os.devnull, "w"))
        live_carla_processes.add(os.getpgid(self.server_process.pid))

        for i in range(RETRIES_ON_ERROR):
            try:
                self.client = CarlaClient("localhost", self.server_port)
                return self.client.connect()
            except Exception as e:
                print("Error connecting: {}, attempt {}".format(e, i))
                time.sleep(2)

    def clear_server_state(self):
        print("Clearing Carla server state")
        try:
            if self.client:
                self.client.disconnect()
                self.client = None
        except Exception as e:
            print("Error disconnecting client: {}".format(e))
            pass
        if self.server_process:
            pgid = os.getpgid(self.server_process.pid)
            os.killpg(pgid, signal.SIGKILL)
            live_carla_processes.remove(pgid)
            self.server_port = None
            self.server_process = None

    def __del__(self):  # the __del__ method will be called when the instance of the class is deleted.(memory is freed.)
        self.clear_server_state()

    def reset(self):
        error = None
        for _ in range(RETRIES_ON_ERROR):
            try:
                if not self.server_process:
                    self.init_server()
                return self._reset()
            except Exception as e:
                print("Error during reset: {}".format(traceback.format_exc()))
                self.clear_server_state()
                error = e
        raise error

    def _reset(self):
        self.num_steps = 0
        self.total_reward = 0
        self.prev_measurement = None
        self.prev_image = None
        self.episode_id = datetime.today().strftime("%Y-%m-%d_%H-%M-%S_%f")
        self.measurements_file = None

        # Create a CarlaSettings object. This object is a wrapper around
        # the CarlaSettings.ini file. Here we set the configuration we
        # want for the new episode.
        settings = CarlaSettings()
        self.scenario = random.choice(self.config["scenarios"])
        assert self.scenario["city"] == self.city, (self.scenario, self.city)
        self.weather = random.choice(self.scenario["weather_distribution"])
        settings.set(
            SynchronousMode=True,
            # ServerTimeOut=10000, # CarlaSettings: no key named 'ServerTimeOut'
            SendNonPlayerAgentsInfo=True,
            NumberOfVehicles=self.scenario["num_vehicles"],
            NumberOfPedestrians=self.scenario["num_pedestrians"],
            WeatherId=self.weather)
        settings.randomize_seeds()

        # if self.config["encode_measurement"]:
        #     camera1 = Camera("CameraDepth", PostProcessing="Depth")
        #     camera1.set_image_size(self.config["render_x_res"],
        #                            self.config["render_y_res"])
        #     # camera1.set_position(30, 0, 170)
        #     camera1.set_position(0.5, 0.0, 1.6)
        #     camera1.set(FOV=120)
        #     settings.add_sensor(camera1)
        #
        #     camera2 = Camera("CameraRGB")
        #     camera2.set_image_size(self.config["render_x_res"],
        #                        self.config["render_y_res"])
        #     camera2.set(FOV=120)
        #     camera2.set_position(0.5, 0.0, 1.6)
        #     settings.add_sensor(camera2)
        # elif self.config["use_depth_camera"]:
        #     camera1 = Camera("CameraDepth", PostProcessing="Depth")
        #     camera1.set_image_size(self.config["render_x_res"],
        #                            self.config["render_y_res"])
        #     # camera1.set_position(30, 0, 170)
        #     camera1.set_position(0.5, 0.0, 1.6)
        #     # camera1.set_rotation(0.0, 0.0, 0.0)
        #
        #     settings.add_sensor(camera1)
        # else:
        camera2 = Camera("CameraRGB")
        camera2.set_image_size(self.config["render_x_res"],
                                self.config["render_y_res"])
        camera2.set(FOV=110)
        camera2.set_position(1.2, 0.0, 1.7)
        settings.add_sensor(camera2)

        if self.config["use_seg"]:
            camera1 = Camera("CameraDepth", PostProcessing="Depth")
            camera1.set_image_size(self.config["render_x_res"],
                                   self.config["render_y_res"])
            # camera1.set_position(30, 0, 170)
            camera1.set_position(0.5, 0.0, 1.6)
            camera1.set(FOV=120)
            settings.add_sensor(camera1)
            camera3 = Camera('Segmentation', PostProcessing='SemanticSegmentation')
            camera3.set(FOV=120.0)
            camera3.set_image_size(self.config["render_x_res"],
                                   self.config["render_y_res"])
            camera3.set_position(x=0.50, y=0.0, z=1.6)
            settings.add_sensor(camera3)

        # Setup start and end positions
        scene = self.client.load_settings(settings)
        self.positions = scene.player_start_spots
        self.start_pos = self.positions[self.scenario["start_pos_id"]]
        self.end_pos = self.positions[self.scenario["end_pos_id"]]
        self.start_coord = [
            self.start_pos.location.x, self.start_pos.location.y
        ]
        self.end_coord = [
            self.end_pos.location.x, self.end_pos.location.y
        ]
        print("Start pos {} ({}), end {} ({})".format(
            self.scenario["start_pos_id"], [int(x) for x in self.start_coord],
            self.scenario["end_pos_id"], [int(x) for x in self.end_coord]))

        # Notify the server that we want to start the episode at the
        # player_start index. This function blocks until the server is ready
        # to start the episode.
        print("Starting new episode...")
        self.client.start_episode(self.scenario["start_pos_id"])

        # Process observations: self._read_observation() returns image and py_measurements.
        image, py_measurements = self._read_observation()
        self.prev_measurement = py_measurements
        return self.encode_obs(self.preprocess_image(image), py_measurements)

    # rgb depth forward_speed next_comment
    def encode_obs(self, image, py_measurements):
        assert self.config["framestack"] in [1, 2]
        prev_image = self.prev_image
        self.prev_image = image
        if prev_image is None:
            prev_image = image

        feature_map = np.zeros([4, 4])
        feature_map[1, :] = (py_measurements["forward_speed"] - 30) / 30
        feature_map[1, 3] = py_measurements["intersection_otherlane"]
        feature_map[2, :] = (COMMAND_ORDINAL[py_measurements["next_command"]] - 2) / 2
        feature_map[0, 0] = py_measurements["x_orient"]
        feature_map[0, 1] = py_measurements["y_orient"]
        feature_map[0, 2] = (py_measurements["distance_to_goal"] - 170) / 170
        feature_map[0, 1] = (py_measurements["distance_to_goal_euclidean"] - 170) / 170
        feature_map[3, 0] = (py_measurements["x"] - 50) / 150
        feature_map[3, 1] = (py_measurements["y"] - 50) / 150
        feature_map[3, 2] = (py_measurements["end_coord"][0] - 150) / 150
        feature_map[3, 3] = (py_measurements["end_coord"][1] - 150) / 150
        feature_map = np.tile(feature_map, (32, 32))
        image_ = np.concatenate(
               [prev_image, image, feature_map[:, :, np.newaxis]], axis=2)
        # obs = (image, COMMAND_ORDINAL[py_measurements["next_command"]], [
        #     py_measurements["forward_speed"],
        #     py_measurements["distance_to_goal"]
        # ])
        # print('distance to goal', py_measurements["distance_to_goal"])
        # print("speed", py_measurements["forward_speed"])
        if ENV_CONFIG["VAE"]:
            image_in = np.stack([image, prev_image], axis=0)
            latent_encode = encode(self.vae, image_in)   # encode image to latent space
            if ENV_CONFIG["out_vae"]:
                out_dir = os.path.join(CARLA_OUT_PATH, "vae")
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                out_file = os.path.join(
                    out_dir, "{}_{:>04}.jpg".format(self.episode_id,
                                                    self.num_steps))
                image = decode(self.vae, np.expand_dims(latent_encode[-1], axis=0))
                scipy.misc.imsave(out_file, np.squeeze(image))

            if ENV_CONFIG["SAC"]:
                metric = np.array([COMMAND_ORDINAL[py_measurements["next_command"]]/4,
                                           py_measurements["forward_speed"]/30,
                                           py_measurements["distance_to_goal"]/100])
                latent_encode = np.append(latent_encode.flatten(), metric)
            obs = (latent_encode, COMMAND_ORDINAL[py_measurements["next_command"]], [
                py_measurements["forward_speed"],
                py_measurements["distance_to_goal"]
            ])
        else:
            obs = (image_, COMMAND_ORDINAL[py_measurements["next_command"]], [
                py_measurements["forward_speed"],
                py_measurements["distance_to_goal"]
            ])
        self.last_obs = obs

        return obs
    # TODO:example of py_measurement
    # {'episode_id': '2019-02-22_11-26-36_990083',
    #  'step': 0,
    #  'x': 71.53003692626953,
    #  'y': 302.57000732421875,
    #  'x_orient': -1.0,
    #  'y_orient': -6.4373016357421875e-06,
    #  'forward_speed': -3.7578740032934155e-13,
    #  'dstance_to_goal': 0.6572,
    #  'distance_to_goal_euclidean': 0.6200001239776611,
    #  'collision_vehicles': 0.0,
    #  'collision_pedestrians': 0.0,
    #  'collision_other': 0.0,
    #  'intersection_offroad': 0.0,
    #  'intersection_otherlane': 0.0,
    #  'weather': 0,
    #  'map': '/Game/Maps/Town02',
    #  'start_coord': [0.0, 3.0],
    #  'end_coord': [0.0, 3.0],
    #  'current_scenario': {'city': 'Town02',
    #                       'num_vehicles': 0,
    #                       'num_pedestrians': 20,
    #                       'weather_distribution': [0],
    #                       'start_pos_id': 36,
    #                       'end_pos_id': 40,
    #                       'max_steps': 600},
    #  'x_res': 10,
    #  'y_res': 10,
    #  'num_vehicles': 0,
    #  'num_pedestrians': 20,
    #  'max_steps': 600,
    #  'next_command': 'GO_STRAIGHT',
    #  'action': (0, 1),
    #  'control': {'steer': 1.0,
    #              'throttle': 0.0,
    #              'brake': 0.0,
    #              'reverse': False,
    #              'hand_brake': False},
    #  'reward': 0.0,
    #  'total_reward': 0.0,
    #  'done': False}

    def step(self, action):
        try:
            for _ in range(ENV_CONFIG["action_repeat"]):
                obs = self._step(action)
            return obs
        except Exception:
            print("Error during step, terminating episode early",
                  traceback.format_exc())
            self.clear_server_state()
            return (self.last_obs, 0.0, True, {})

    def _step(self, action):
        if self.config["discrete_actions"]:
            action = DISCRETE_ACTIONS[int(action)]  # Carla action is 2D.
        assert len(action) == 2, "Invalid action {}".format(action)
        if self.enable_autopilot:
            action[0] = self.autopilot.brake if self.autopilot.brake < 0 else self.autopilot.throttle
            action[1] = self.autopilot.steer
        if self.config["squash_action_logits"]:
            forward = 2 * float(sigmoid(action[0]) - 0.5)
            throttle = float(np.clip(forward, 0, 1))
            brake = float(np.abs(np.clip(forward, -1, 0)))
            steer = 2 * float(sigmoid(action[1]) - 0.5)
        else:
            throttle = float(np.clip(action[0], 0, 1))
            brake = float(np.abs(np.clip(action[0], -1, 0)))
            steer = float(np.clip(action[1], -1, 1))

        # reverse and hand_brake are disabled.
        reverse = False
        hand_brake = False

        if self.config["verbose"]:
            print("steer", steer, "throttle", throttle, "brake", brake,
                  "reverse", reverse)

        self.client.send_control(
            steer=steer,
            throttle=throttle,
            brake=brake,
            hand_brake=hand_brake,
            reverse=reverse)

        # Process observations: self._read_observation() returns image and py_measurements.
        image, py_measurements = self._read_observation()
        # print(image.shape)
        if self.config["verbose"]:
            print("Next command", py_measurements["next_command"])
        if type(action) is np.ndarray:
            py_measurements["action"] = [float(a) for a in action]
        else:
            py_measurements["action"] = action
        py_measurements["control"] = {
            "steer": steer,
            "throttle": throttle,
            "brake": brake,
            "reverse": reverse,
            "hand_brake": hand_brake,
        }

        # compute reward
        reward = compute_reward(self, self.prev_measurement, py_measurements)

        self.total_reward += reward
        py_measurements["reward"] = reward
        py_measurements["total_reward"] = self.total_reward

        # done or not
        done = (self.num_steps > self.scenario["max_steps"]
                # or py_measurements["next_command"] == "REACH_GOAL"
                or (self.config["early_terminate_on_collision"]
                    and collided_done(py_measurements)))

        py_measurements["done"] = done
        self.prev_measurement = py_measurements

        # Write out measurements to file
        if self.config["verbose"] and CARLA_OUT_PATH:
            if not self.measurements_file:
                self.measurements_file = open(
                    os.path.join(
                        CARLA_OUT_PATH,
                        "measurements_{}.json".format(self.episode_id)), "w")
            self.measurements_file.write(json.dumps(py_measurements))
            self.measurements_file.write("\n")
            if done:
                self.measurements_file.close()
                self.measurements_file = None
                if self.config["convert_images_to_video"]:
                    self.images_to_video()

        self.num_steps += 1
        image = self.preprocess_image(image)
        # print(image.shape)
        # print(py_measurements["next_command"])
         #print(self.end_coord)
        # print(py_measurements["distance_to_goal"])
        return (self.encode_obs(image, py_measurements), reward, done,
                py_measurements)

    # def images_to_video(self):
    #     videos_dir = os.path.join(CARLA_OUT_PATH, "Videos")
    #     if not os.path.exists(videos_dir):
    #         os.makedirs(videos_dir)
    #     ffmpeg_cmd = (
    #         "ffmpeg -loglevel -8 -r 60 -f image2 -s {x_res}x{y_res} "
    #         "-start_number 0 -i "
    #         "{img}_%04d.jpg -vcodec libx264 {vid}.mp4 && rm -f {img}_*.jpg "
    #     ).format(
    #         x_res=self.config["render_x_res"],
    #         y_res=self.config["render_y_res"],
    #         vid=os.path.join(videos_dir, self.episode_id),
    #         # img=os.path.join(CARLA_OUT_PATH, "CameraRGB", self.episode_id))
    #         img=os.path.join(CARLA_OUT_PATH, "vae", self.episode_id))
    #     print("Executing ffmpeg command", ffmpeg_cmd)
    #     subprocess.call(ffmpeg_cmd, shell=True)


    def images_to_video(self):
        videos_dir = os.path.join(CARLA_OUT_PATH, "Videos")
        if not os.path.exists(videos_dir):
            os.makedirs(videos_dir)
        ffmpeg_cmd = (
            "ffmpeg -loglevel -8 -r 60 -f image2 -s {x_res}x{y_res} "
            "-start_number 0 -i "
            "{img}_%04d.jpg -vcodec libx264 {vid}.mp4 && rm -f {img}_*.jpg "
        ).format(
            x_res=self.config["x_res"],
            y_res=self.config["y_res"],
            vid=os.path.join(videos_dir, self.episode_id),
            # img=os.path.join(CARLA_OUT_PATH, "CameraRGB", self.episode_id))
            img=os.path.join(CARLA_OUT_PATH, "vae", self.episode_id))
        print("Executing ffmpeg command", ffmpeg_cmd)
        subprocess.call(ffmpeg_cmd, shell=True)

    def preprocess_image(self, image):
        if self.config["encode_measurement"]:
            # data = (image - 0.5) * 2
            data = image.reshape(self.config["render_y_res"],
                                self.config["render_x_res"], -1)
            data = cv2.resize(
                data, (self.config["x_res"], self.config["y_res"]),
                interpolation=cv2.INTER_AREA)
        return data

    def _read_observation(self):
        # Read the data produced by the server this frame.
        measurements, sensor_data = self.client.read_data()

        # Print some of the measurements.
        if self.enable_autopilot:
            self.autopilot = measurements.player_measurements.autopilot_control
        if self.config["verbose"]:
            print_measurements(measurements)

        observation = None
        if self.config["use_seg"]:
            observation = np.concatenate(((sensor_data["Segmentation"].data[:, :, np.newaxis] - 7)/7,
                                        (sensor_data["CameraDepth"].data[:, :, np.newaxis] - 0.5)*0.5), axis=2)
        elif self.config["encode_measurement"]:
            # print(sensor_data["CameraRGB"].data.shape, sensor_data["CameraDepth"].data.shape)
            #observation = np.concatenate(((sensor_data["CameraRGB"].data.astype(np.float32)-128)/128,
            #                            (sensor_data["CameraDepth"].data[:, :, np.newaxis] - 0.5)*0.5), axis=2)
            observation = (sensor_data["CameraRGB"].data.astype(np.float32) - 128)/128
            # observation = (sensor_data["CameraRGB"].data.astype(np.float32)/255) - 0.5
            # print("observation_shape", observation.shape)

        else:
            if self.config["use_depth_camera"]:
                camera_name = "CameraDepth"

            else:
                camera_name = "CameraRGB"

            for name, image in sensor_data.items():
                if name == camera_name:
                    observation = image

        cur = measurements.player_measurements

        if self.config["enable_planner"]:
            next_command = COMMANDS_ENUM[self.planner.get_next_command(
                [cur.transform.location.x, cur.transform.location.y, GROUND_Z],
                [
                    cur.transform.orientation.x, cur.transform.orientation.y,
                    GROUND_Z
                ],
                [self.end_pos.location.x, self.end_pos.location.y, GROUND_Z], [
                    self.end_pos.orientation.x, self.end_pos.orientation.y,
                    GROUND_Z
                ])]
        else:
            next_command = "LANE_FOLLOW"

        if next_command == "REACH_GOAL":
            distance_to_goal = 0.0  # avoids crash in planner
            self.end_pos = self.positions[self.scenario["end_pos_id"]]
        elif self.config["enable_planner"]:
            distance_to_goal = self.planner.get_shortest_path_distance([
                cur.transform.location.x, cur.transform.location.y, GROUND_Z
            ], [
                cur.transform.orientation.x, cur.transform.orientation.y,
                GROUND_Z
            ], [self.end_pos.location.x, self.end_pos.location.y, GROUND_Z], [
                self.end_pos.orientation.x, self.end_pos.orientation.y,
                GROUND_Z
            ])  
        # now the metrix should be meter carla8.0
        # TODO run experience to verify the scale of distance in order to determine reward function          
        else:
            distance_to_goal = -1

        distance_to_goal_euclidean = float(
            np.linalg.norm([
                cur.transform.location.x - self.end_pos.location.x,
                cur.transform.location.y - self.end_pos.location.y
            ]))

        py_measurements = {
            "episode_id": self.episode_id,
            "step": self.num_steps,
            "x": cur.transform.location.x,
            "y": cur.transform.location.y,
            "x_orient": cur.transform.orientation.x,
            "y_orient": cur.transform.orientation.y,
            "forward_speed": cur.forward_speed*3.6,
            "distance_to_goal": distance_to_goal,
            "distance_to_goal_euclidean": distance_to_goal_euclidean,
            "collision_vehicles": cur.collision_vehicles,
            "collision_pedestrians": cur.collision_pedestrians,
            "collision_other": cur.collision_other,
            "intersection_offroad": cur.intersection_offroad,
            "intersection_otherlane": cur.intersection_otherlane,
            "weather": self.weather,
            "map": self.config["server_map"],
            "start_coord": self.start_coord,
            "end_coord": self.end_coord,
            "current_scenario": self.scenario,
            "x_res": self.config["x_res"],
            "y_res": self.config["y_res"],
            "num_vehicles": self.scenario["num_vehicles"],
            "num_pedestrians": self.scenario["num_pedestrians"],
            "max_steps": self.scenario["max_steps"],
            "next_command": next_command,
        }

        if CARLA_OUT_PATH and self.config["log_images"]:
            for name, image in sensor_data.items():
                out_dir = os.path.join(CARLA_OUT_PATH, name)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                out_file = os.path.join(
                    out_dir, "{}_{:>04}.jpg".format(self.episode_id,
                                                    self.num_steps))
                scipy.misc.imsave(out_file, image.data)

        assert observation is not None, sensor_data
        return observation, py_measurements


def compute_reward_corl2017(env, prev, current):
    reward = 0.0

    cur_dist = current["distance_to_goal"]

    prev_dist = prev["distance_to_goal"]

    if env.config["verbose"]:
        print("Cur dist {}, prev dist {}".format(cur_dist, prev_dist))

    # Distance travelled toward the goal in m
    reward += np.clip(prev_dist - cur_dist, -50.0, 50.0)

    # Change in speed (km/h)
    reward += 0.05 * (current["forward_speed"] - prev["forward_speed"])

    # New collision damage
    reward -= .00002 * (
        current["collision_vehicles"] + current["collision_pedestrians"] +
        current["collision_other"] - prev["collision_vehicles"] -
        prev["collision_pedestrians"] - prev["collision_other"])

    # New sidewalk intersection
    reward -= 2 * (
        current["intersection_offroad"] - prev["intersection_offroad"])

    # New opposite lane intersection
    reward -= 2 * (
        current["intersection_otherlane"] - prev["intersection_otherlane"])

    return reward


def compute_reward_custom(env, prev, current):
    reward = 0.0

    cur_dist = current["distance_to_goal"]
    prev_dist = prev["distance_to_goal"]

    if env.config["verbose"]:
        print("Cur dist {}, prev dist {}".format(cur_dist, prev_dist))

    # Distance travelled toward the goal in m
    reward += 0.5 * np.clip(prev_dist - cur_dist, -12.0, 12.0)

    # Speed reward, up 30.0 (km/h)
    reward += np.clip(current["forward_speed"], 0.0, 30.0) / 10
    if current["forward_speed"] > 40:
        reward -= (current["forward_speed"] - 40)/12 
    # New collision damage
    new_damage = (
        current["collision_vehicles"] + current["collision_pedestrians"] +
        current["collision_other"] - prev["collision_vehicles"] -
        prev["collision_pedestrians"] - prev["collision_other"])
    # print(current["collision_other"], current["collision_vehicles"], current["collision_pedestrians"])
    # 0.0 41168.109375 0.0
    if new_damage:
        reward -= 15 + current["forward_speed"] * 3

    # Sidewalk intersection
    reward -= np.clip(10 * current["forward_speed"] * int(current["intersection_offroad"] > 0.001), 0, 50)   # [0, 1]
    # print(current["intersection_offroad"])
    # Opposite lane intersection
    reward -= 4 * current["intersection_otherlane"]  # [0, 1]
    # print(current["intersection_offroad"], current["intersection_otherlane"])
    # Reached goal
    if current["next_command"] == "REACH_GOAL":
        reward += 200.0
        print('bro, you reach the goal, well done!!!')

    return reward


def compute_reward_custom_2(env, prev, current):
    reward = 0.0

    # cur_dist = current["distance_to_goal"]
    # prev_dist = prev["distance_to_goal"]
    # print(">>>>>>>>>>", prev_dist - cur_dist)
    # if env.config["verbose"]:
    #     print("Cur dist {}, prev dist {}".format(cur_dist, prev_dist))

    # Distance travelled toward the goal in m
    # reward += 0.3 * np.clip(prev_dist - cur_dist, -12.0, 12.0)

    # Speed reward, up 30.0 (km/h)
    reward += np.clip(current["forward_speed"], 0.0, 30.0) / 6
    if current["forward_speed"] > 40:
        reward -= (current["forward_speed"] - 40)/12
    # New collision damage
    new_damage = (
        current["collision_vehicles"] + current["collision_pedestrians"] +
        current["collision_other"] - prev["collision_vehicles"] -
        prev["collision_pedestrians"] - prev["collision_other"])
    if new_damage:
        reward -= 15 + (current["forward_speed"]/3)**2
        print("<<<<<<<<<<<<<<<<<<damage>>>>>>>>>>>>>>>>>>>")
    reward -= 0.03 * (current["control"]["steer"]**2)
    reward -= np.clip(10 * current["forward_speed"] * int(current["intersection_offroad"] > 0.001), 0, 20)   # [0, 1]
    reward -= 4 * current["intersection_otherlane"]  # [0, 1]
    reward -= 0.03 if current["forward_speed"] < 1 else 0

    # if current["next_command"] == "REACH_GOAL":
    #     reward += 10
    #     print('bro, you reach the goal, well done!!!')
    return reward

def compute_reward_lane_keep(env, prev, current):
    reward = 0.0

    # Speed reward, up 30.0 (km/h)
    reward += np.clip(current["forward_speed"], 0.0, 30.0) / 5

    # New collision damage
    new_damage = (
        current["collision_vehicles"] + current["collision_pedestrians"] +
        current["collision_other"] - prev["collision_vehicles"] -
        prev["collision_pedestrians"] - prev["collision_other"])
    if new_damage:
        reward -= 80.0

    # Sidewalk intersection
    reward -= 0.2*current["intersection_offroad"]

    # Opposite lane intersection
    reward -= 0.5*current["intersection_otherlane"]

    return reward


REWARD_FUNCTIONS = {
    "corl2017": compute_reward_corl2017,
    "custom": compute_reward_custom,
    "custom2": compute_reward_custom_2,
    "lane_keep": compute_reward_lane_keep,
}
def compute_reward(env, prev, current):
    return REWARD_FUNCTIONS[env.config["reward_function"]](env, prev, current)


def print_measurements(measurements):
    number_of_agents = len(measurements.non_player_agents)
    player_measurements = measurements.player_measurements
    message = "Vehicle at ({pos_x:.1f}, {pos_y:.1f}), "
    message += "{speed:.2f} km/h, "
    message += "Collision: {{vehicles={col_cars:.0f}, "
    message += "pedestrians={col_ped:.0f}, other={col_other:.0f}}}, "
    message += "{other_lane:.0f}% other lane, {offroad:.0f}% off-road, "
    message += "({agents_num:d} non-player agents in the scene)"
    message = message.format(
        pos_x=player_measurements.transform.location.x ,  # m in calra8
        pos_y=player_measurements.transform.location.y ,
        speed=player_measurements.forward_speed,
        col_cars=player_measurements.collision_vehicles,
        col_ped=player_measurements.collision_pedestrians,
        col_other=player_measurements.collision_other,
        other_lane=100 * player_measurements.intersection_otherlane,
        offroad=100 * player_measurements.intersection_offroad,
        agents_num=number_of_agents)
    print(message)


def sigmoid(x):
    x = float(x)
    return np.exp(x) / (1 + np.exp(x))


def collided_done(py_measurements):
    m = py_measurements
    collided = (m["collision_vehicles"] > 0 or m["collision_pedestrians"] > 0
                or m["collision_other"] > 0 or m["intersection_offroad"] > 0.05)
    return bool(collided or m["total_reward"] < -80)


if __name__ == "__main__":
    for _ in range(2):
        env = CarlaEnv(enable_autopilot=True)
        obs = env.reset()      
        print(obs[0].shape) 
        start = time.time
        # import matplotlib.pyplot as plt
        # plt.show()
        # plt.imshow(obs[0])
        done = False
        i = 0
        total_reward = 0.0
        while 1:
            # print(i)
            i += 1
            if i > 1000:
                i = 0
                env.images_to_video()
                env.reset()
            if ENV_CONFIG["discrete_actions"]:
                obs, reward, done, info = env.step(1)
                # print(obs[0].shape)
            else:
                obs, reward, done, info = env.step([0, 0])
                print(obs[0].shape)
                # print(reward)
            total_reward += reward
        # print("{:.2f} fps".format(float(i / (time.time() - start))))
