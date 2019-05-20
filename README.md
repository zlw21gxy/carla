# carla
running carla with RL method
for now using ray to train a ppo agent
1. he can learn turn left or right 20190226
2. using 8 channel rbg + rgb + measurement_encode
3. now wu use rgb camera only without depth or segmentation 
4. choose which camera to use in env.py
5. pip install ray first
# add vae for featur extract
1. lr 1e-5 otherwise doesn't converge
2. 10k trainig example record on carla
