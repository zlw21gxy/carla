from vae_unit import load_mnist_data, plot_image_rows, load_carla_data
import matplotlib.pyplot as plt
from keras.losses import mse, binary_crossentropy
from keras import backend as K
from keras import layers
import keras
from keras.models import Model, load_model
import numpy as np
from config import IMG_SIZE, mode, epochs, latent_dim, beta, scale, scale_r, lr, use_pretrained, filepath

# filepath = "/home/gu/project/ppo/ppo_carla/models/carla/carla_vae_model_beta_3_r_1100.hdf5"
batch_size = 100
from keras.callbacks import ModelCheckpoint
import vae_unit as vae_util

# filepath = "/home/gu/project/ppo/ppo_carla/models/mnist/ld_{}_beta_{}_r_{}.hdf5".format(latent_dim, beta, scale_r)


# filepath = "/home/gu/project/ppo/ppo_carla/models/mnist/ld_{}_beta_{}_r_{}.hdf5".format(latent_dim, beta, scale_r)

def create_vae(latent_dim, return_kl_loss_op=False):
    '''
    Creates a VAE able to auto-encode MNIST images.

    Args:
        latent_dim: dimensionality of latent space
        return_kl_loss_op: whether to return the operation for
                           computing the KL divergence loss.

    Returns:
        The VAE model. If return_kl_loss_op is True, then the
        operation for computing the KL divergence loss is
        additionally returned.
    '''
    encoder = vae_util.create_encoder(latent_dim)
    decoder = vae_util.create_decoder(latent_dim)
    sampler = vae_util.create_sampler()

    x = layers.Input(shape=IMG_SIZE, name='image')
    t_mean, t_log_var = encoder(x)
    t = sampler([t_mean, t_log_var])
    t_decoded = decoder(t)

    model = Model(x, t_decoded, name='vae')

    if return_kl_loss_op:
        kl_loss = -0.5 * K.sum(1 + t_log_var - K.square(t_mean) - K.exp(t_log_var), axis=-1)
        return model, kl_loss
    else:
        return model


def vae_loss(x, t_decoded):
    '''Total loss for the plain VAE'''
    return scale * K.mean(scale_r * reconstruction_loss(x, t_decoded) + beta * vae_kl_loss)  # adjust losss scale


def reconstruction_loss(x, t_decoded):
    '''Reconstruction loss for the plain VAE'''
    return K.mean(K.square(x - t_decoded), axis=(1, 2, 3))  # average over images


# def reconstruction_loss(x, t_decoded):
#     '''Reconstruction loss for the plain VAE'''
#     return K.sum(K.binary_crossentropy(
#         K.batch_flatten(x),
#         K.batch_flatten(t_decoded)), axis=-1)


if mode == "carla":
    x_train, x_test = load_carla_data(normalize=True)
else:
    (x_train, _), (x_test, _) = load_mnist_data(normalize=True)

# Keep only a single checkpoint, the best over test loss
checkpoint = ModelCheckpoint(filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             period=1,
                             mode='min')

# Create plain VAE model and associated KL divergence loss operation
vae, vae_kl_loss = create_vae(latent_dim, return_kl_loss_op=True)
if use_pretrained:
    vae.load_weights(filepath)
else:
    learning_rate = lr  # if we set lr to 0.005 network will blow up...that is...
    decay_rate = 0.5e-4
    adam = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay_rate,
                                 amsgrad=False)
    vae.compile(optimizer=adam, loss=vae_loss)
    history = vae.fit(x=x_train, y=x_train, epochs=epochs, batch_size=batch_size,
                      shuffle=True, validation_data=(x_test, x_test), verbose=2,
                      callbacks=[checkpoint])  # , SGDLearningRateTracker()])


def encode(model, images):
    '''Encodes images with the encoder of the given auto-encoder model'''
    return model.get_layer('encoder').predict(images)[0]


def decode(model, codes):
    '''Decodes latent vectors with the decoder of the given auto-encoder model'''
    return model.get_layer('decoder').predict(codes)


def encode_decode(model, images):
    '''Encodes and decodes an image with the given auto-encoder model'''
    return decode(model, encode(model, images))


if not use_pretrained:
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
# x_test = train_generator.next()
selected_idx = np.random.choice(range(x_test.shape[0]), 10, replace=False)
selected = x_test[selected_idx]
selected_dec_vae = encode_decode(vae, selected)

plot_image_rows([selected, selected_dec_vae],
                ['Original images',
                 'Images generated by plain VAE'])
plt.show()
