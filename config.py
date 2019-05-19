# mode = "mnist"
# mode = "carla"
mode = "check"
if mode == "mnist":
    IMG_SIZE=(28, 28, 1)
    epochs = 15
    latent_dim = 5
    beta = 1
    scale = 1
    scale_r = 600
    lr = 1e-5
    use_pretrained = False
    filepath = "/home/gu/project/ppo/ppo_carla/models/mnist/ld_{}_beta_{}_r_{}.hdf5".format(latent_dim, beta, scale_r)
elif mode == "carla":
    IMG_SIZE=(128, 128, 3)
    epochs = 100
    latent_dim = 256
    beta = 10
    scale = 1/beta
    scale_r = 1100
    lr = 1e-5
    use_pretrained = False
    filepath = "/home/gu/project/ppo/ppo_carla/models/carla/ld_{}_beta_{}_r_{}.hdf5".format(latent_dim, beta, scale_r)
else:
    IMG_SIZE=(128, 128, 3)
    epochs = 100
    latent_dim = 256
    beta = 10
    scale = 1/beta
    scale_r = 1100
    lr = 1e-5
    use_pretrained = True
    mode = "carla"
    # filepath = "/home/gu/project/ppo/ppo_carla/models/carla/ld_{}_beta_{}_r_{}.hdf5".format(latent_dim, beta, scale_r)
    filepath = "ld_256_beta_0.1_r_1100.hdf5"
