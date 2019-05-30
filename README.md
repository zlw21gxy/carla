## DRL for carla
running carla with RL method
for now using ray to train a ppo agent
1. he can learn turn left or right 20190226
2. using 8 channel rbg + rgb + measurement_encode
3. now wu use rgb camera only without depth or segmentation 
4. choose which camera to use in env.py
5. pip install ray first
## use vae for feature extract
1. lr = 1e-4 otherwise doesn't converge
2. 10k trainig example, record on carla
## training
1. run vae_carla.py directly
2. change file dir in vae_unit.py if u want use your own dataset
3. change config for trainig in config.py
4. image size (128 128 3) (28 28 3) (28 28 1) are available for now 
## DRL + VAE
1. running carla with SAC by extract feature with VAE first
2. action repeat = 2 (maybe 1 is better)
3. agent can driving alongside the road but can't avoid collision with other cars
## DEBUG
1. use mnist for debug purpose
