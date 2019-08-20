import numpy as np
from dim import encoder,load_carla_data
import imageio

# model_train.load_weights('dim_test.ckpt')

data = load_carla_data((128, 128), normalize=True, flag="test", num=5000, train_dir="/home/gu/carla_out_data/town1_data/*")
# data = load_carla_data((256, 256), normalize=True, train_dir="/home/gu/carla_out_data/town1_data/*", num=5000)
img_dim = data.shape[1]
# data = load_carla_data((128, 128), normalize=True, flag="test", num=5000)

# encoder.load_weights("ckpt/dim_large.ckpt")
# encoder.load_weights("ckpt/dim_test10:50:08_128.ckpt")
encoder.load_weights("ckpt/dim_alpha_0.5_beta1.7_gamma_0.1_img_128.ckpt")

zs = encoder.predict(data, verbose=True)
print(zs.mean(), zs.std())

def sample_knn(path):
    n = 10
    topn = 10
    figure1 = np.zeros((img_dim*n, img_dim*topn, 3))
    figure2 = np.zeros((img_dim*n, img_dim*topn, 3))
    zs_ = zs / (zs**2).sum(1, keepdims=True)**0.5
    for i in range(n):
        one = np.random.choice(len(data))
        idxs = ((zs**2).sum(1) + (zs[one]**2).sum() - 2 * np.dot(zs, zs[one])).argsort()[:topn]
        for j,k in enumerate(idxs):
            digit = data[k]
            figure1[i*img_dim: (i+1)*img_dim,
                   j*img_dim: (j+1)*img_dim] = digit
        idxs = np.dot(zs_, zs_[one]).argsort()[-n:][::-1]
        for j,k in enumerate(idxs):
            digit = data[k]
            figure2[i*img_dim: (i+1)*img_dim,
                   j*img_dim: (j+1)*img_dim] = digit
    figure1 = (figure1 + 1) / 2 * 255
    figure1 = np.clip(figure1, 0, 255)
    figure2 = (figure2 + 1) / 2 * 255
    figure2 = np.clip(figure2, 0, 255)
    imageio.imwrite(path+'_l2_.png', figure1)
    imageio.imwrite(path+'_cos_.png', figure2)


sample_knn('test')