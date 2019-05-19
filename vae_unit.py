import numpy as np

import keras

from keras import backend as K
from keras import layers
from keras.models import Model, Sequential

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import glob
import cv2
from keras.datasets import mnist

import matplotlib.pyplot as plt

# Shape of MNIST images
from config import IMG_SIZE
image_shape = IMG_SIZE


def create_encoder(latent_dim):
    '''
    Creates a convolutional encoder model for MNIST images.

    - Input for the created model are MNIST images.
    - Output of the created model are the sufficient statistics
      of the variational distriution q(t|x;phi), mean and log
      variance.
    '''
    encoder_iput = layers.Input(shape=image_shape, name='image')

    if image_shape == (28, 28, 3):
        x = layers.Conv2D(32, 3, padding='same', activation='relu')(encoder_iput)
        x = layers.Conv2D(64, 3, padding='same', activation='relu', strides=(2, 2))(x)
        x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
        x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation='relu')(x)

    elif image_shape == (28, 28, 1):
        x = layers.Conv2D(32, 3, padding='same', activation='relu')(encoder_iput)
        x = layers.Conv2D(64, 3, padding='same', activation='relu', strides=(2, 2))(x)
        x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
        x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(48, activation='relu')(x)

    elif image_shape == (128, 128, 3):
        kwargs = dict(strides=(2, 2), activation="elu", padding="same")
        kwargs2 = dict(strides=(1, 1), activation="elu", padding="same")
        kwargs3 = dict(strides=(1, 1), padding="same")
        x = layers.Conv2D(48, 3, **kwargs)(encoder_iput)
        x = layers.Conv2D(64, 3, **kwargs)(x)
        x = layers.Conv2D(72, 3, **kwargs)(x)
        x = layers.Conv2D(256, 3, **kwargs)(x)
        x = layers.Conv2D(512, 3, **kwargs)(x)
        # x = keras.layers.GlobalAveragePooling2D()(x)
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation="elu")(x)
        # x = layers.Dense(256, activation='elu')(x)

    t_mean = layers.Dense(latent_dim, name='t_mean')(x)
    t_log_var = layers.Dense(latent_dim, name='t_log_var')(x)

    return Model(encoder_iput, [t_mean, t_log_var], name='encoder')


def create_decoder(latent_dim):
    '''
    Creates a (de-)convolutional decoder model for MNIST images.

    - Input for the created model are latent vectors t.
    - Output of the model are images of shape (28, 28, 1) where
      the value of each pixel is the probability of being white.
    '''
    decoder_input = layers.Input(shape=(latent_dim,), name='t')
    if image_shape == (28, 28, 3):
        x = layers.Dense(12544, activation='relu')(decoder_input)
        x = layers.Reshape((14, 14, 64))(x)
        x = layers.Conv2DTranspose(32, 3, padding='same', activation='relu', strides=(2, 2))(x)
        x = layers.Conv2D(3, 3, padding='same', name='image')(x)

    if image_shape == (28, 28, 1):
        x = layers.Dense(12544, activation='relu')(decoder_input)
        x = layers.Reshape((14, 14, 64))(x)
        x = layers.Conv2DTranspose(32, 3, padding='same', activation='relu', strides=(2, 2))(x)
        x = layers.Conv2D(1, 3, padding='same', name='image')(x)
        # x = layers.Conv2D(1, 3, padding='same', activation="sigmoid", name='image')(x)

    elif image_shape == (128, 128, 3):
        kwargs = dict(strides=(2, 2), activation="elu", padding="same")
        kwargs2 = dict(strides=(4, 4), activation="elu", padding="same")
        kwargs3 = dict(strides=(1, 1), padding="same")
        kwargs4 = dict(strides=(2, 2), padding="same")
        x = layers.Dense(1024, activation='elu')(decoder_input)
        x = layers.Reshape((4, 4, 64))(x)
        x = layers.Conv2DTranspose(256, 3, **kwargs)(x)
        x = layers.Conv2DTranspose(128, 3, **kwargs)(x)
        x = layers.Conv2DTranspose(64, 3, **kwargs)(x)
        x = layers.Conv2DTranspose(48, 3, **kwargs)(x)
        x = layers.Conv2DTranspose(32, 3, **kwargs)(x)
        x = layers.Conv2D(3, 3, **kwargs3)(x)
    return Model(decoder_input, x, name='decoder')


def sample(args):
    '''
    Draws samples from a standard normal and scales the samples with
    standard deviation of the variational distribution and shifts them
    by the mean.

    Args:
        args: sufficient statistics of the variational distribution.

    Returns:
        Samples from the variational distribution.
    '''
    t_mean, t_log_var = args
    t_sigma = K.sqrt(K.exp(t_log_var))
    epsilon = K.random_normal(shape=K.shape(t_mean), mean=0., stddev=1.)
    return t_mean + t_sigma * epsilon


def create_sampler():
    '''
    Creates a sampling layer.
    '''
    return layers.Lambda(sample, name='sampler')


def create_predictor_linear(latent_dim):
    '''
    Creates a regressor that estimates digit values
    from latent variables.
    '''
    predictor_input = layers.Input(shape=(latent_dim,))

    x = layers.Dense(128, activation='relu')(predictor_input)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(1, activation='linear')(x)

    return Model(predictor_input, x, name='predictor')


def create_classifier():
    '''
    Creates a classifier that predicts digit labels
    from digit images.
    '''
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=image_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model


def plot_nll(gx, gy, nll):
    fig = plt.figure(figsize=(15, 6))
    plt.subplots_adjust(hspace=0.4)

    for i in range(10):
        plt.subplot(2, 5, i + 1)
        gz = nll(i).reshape(gx.shape)
        im = plt.contourf(gx, gy, gz,
                          cmap='coolwarm',
                          norm=LogNorm(),
                          levels=np.logspace(0.2, 1.8, 100))
        plt.title('Target = {i}')

    fig.subplots_adjust(right=0.8)
    fig.colorbar(im, fig.add_axes([0.82, 0.13, 0.02, 0.74]),
                 ticks=np.logspace(0.2, 1.8, 11), format='%.2f',
                 label='Negative log likelihood')


def load_mnist_data(normalize=False):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))

    if normalize:
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.

    return (x_train, y_train), (x_test, y_test)


def load_carla_data(normalize=False, num=None):
    train_dir = "/home/gu/carla_out/train/*.jpg"
    val_dir = "/home/gu/carla_out/val/*.jpg"
    file_dir = glob.glob(train_dir)
    if not num:
        num = len(file_dir)
    image_shape = IMG_SIZE[:2]
    i = 0
    images_train = []
    for file in file_dir:
        images_train.append(cv2.resize(cv2.imread(file), image_shape, interpolation=cv2.INTER_AREA))
        if i > num:
            break
        i += 1

    x_train = np.stack(images_train)

    images_val = [cv2.resize(cv2.imread(file), image_shape, interpolation=cv2.INTER_AREA)
                    for file in glob.glob(val_dir)]
    x_test = np.stack(images_val)

    if normalize:
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.

    return x_train[:,:,:,::-1], x_test[:,:,:,::-1]

def plot_image_rows(images_list, title_list):
    rows = len(images_list)
    cols = len(images_list[0])
    # plt.figure(figsize=(cols, 3))
    def plot_image_row(images, title):
        plt.figure(figsize=(cols, 3))
        plt.gcf().suptitle(title)
        for i, img in enumerate(images):
            plt.subplot(rows, cols, i + 1)
            img = img*255
            num_channel = img.shape[-1]
            if num_channel == 1:
                plt.imshow(np.clip(img[:,:,0].astype("int32"), 0, 255))
            else:
                plt.imshow(np.clip(img.astype("int32"), 0, 255))
            plt.axis('off')

    for images, title in zip(images_list, title_list):
        plot_image_row(images, title)



def plot_laplacian_variances(lvs_1, lvs_2, lvs_3, title):
    plt.hist(lvs_1, alpha=0.2, bins=50, label='Original images')
    plt.hist(lvs_2, alpha=0.2, bins=50, label='Images generated by plain VAE')
    plt.hist(lvs_3, alpha=0.2, bins=50, label='Images generated by DFC VAE')
    plt.xlabel('Laplacian variance')
    plt.title(title)
    plt.legend()