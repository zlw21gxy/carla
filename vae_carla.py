from vae_unit import load_mnist_data, plot_image_rows, load_carla_data, plot_laplacian_variances
import matplotlib.pyplot as plt
from keras.losses import mse, binary_crossentropy
from keras import backend as K
from keras import layers
import keras
from keras.models import Model, load_model
import numpy as np
from config import IMG_SIZE, mode, epochs, latent_dim, beta, scale, scale_r, lr, use_pretrained, filepath, batch_size
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.callbacks import ModelCheckpoint
import vae_unit as vae_util
from sklearn.model_selection import train_test_split
import cv2

# selected_pm_layer_weights = [1.0, 1.0, 1.0]  # weight for pre-train model
if mode[:5] == "carla":
    f_p_train = lambda x: (x - 128 + 0 * np.random.randn(IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])) / 128
    f_p_test = lambda x: (x - 128) / 128
    # train_datagen = ImageDataGenerator(shear_range=7,
    #                                    horizontal_flip=True,
    #                                    preprocessing_function=f_p_train)
    train_datagen = ImageDataGenerator(preprocessing_function=f_p_train)
    train_generator = train_datagen.flow_from_directory(
        '/home/gu/carla_out/data_debug/train',
        target_size=IMG_SIZE[:2],
        batch_size=batch_size,
        class_mode='input')

    test_datagen = ImageDataGenerator(preprocessing_function=f_p_test)

    val_generator = test_datagen.flow_from_directory(
        '/home/gu/carla_out/data_debug/train',
        target_size=IMG_SIZE[:2],
        batch_size=batch_size,
        class_mode='input')
    # x_train, x_test = load_carla_data(normalize=False, num=2100)
    # x_train = x_train.astype('float32') / 255.
    # x_test = x_test.astype('float32') / 255.
else:
    (x_train, _), (x_test, y_test) = load_mnist_data(normalize=False)
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    # f_p_test = lambda x: x / 255.
    f_p_test = lambda x: (x - 128) / 128
    train_datagen = ImageDataGenerator(preprocessing_function=f_p_test)
    val_datagen = ImageDataGenerator(preprocessing_function=f_p_test)


    def data_generator(generator):
        for data in generator:
            yield data, data


    train_generator = data_generator(train_datagen.flow(x_train, batch_size=100))
    val_generator = data_generator(val_datagen.flow(x_test, batch_size=100))


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


# x Out[2]: <tf.Tensor 'decoder_target:0' shape=(?, ?, ?, ?) dtype=float32>
# t_decoded Out[3]: <tf.Tensor 'decoder/image/BiasAdd:0' shape=(?, ?, ?, 1) dtype=float32>

def reconstruction_loss(x, t_decoded):
    '''Reconstruction loss for the plain VAE'''
    # return K.mean(K.square(x - t_decoded), axis=(1, 2, 3))  # average over images
    return K.sum(K.square(x - t_decoded), axis=(1, 2, 3))  # average over images


def perceptual_loss(x, t_decoded):
    '''Perceptual loss for the DFC VAE'''
    base_model = VGG16(weights='imagenet', include_top=False)
    selected_pm_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1']
    selected_pm_layer_weights = [1.0, 1.0, 1.0]  # weight for pre-train model
    model = Model(inputs=base_model.input, outputs=[base_model.get_layer(l).output for l in selected_pm_layers])
    model.trainable = False
    # f_p = lambda x_: x_ * 255.  # rescale to [0 255]
    f_p = lambda x_: x_ * 128. + 128.  # rescale to [0 255]
    x = f_p(x)
    t_decoded = f_p(t_decoded)
    h1_list = model(x)
    h2_list = model(t_decoded)

    rc_loss = 0.0

    for h1, h2, weight in zip(h1_list, h2_list, selected_pm_layer_weights):
        h1 = K.batch_flatten(h1)
        h2 = K.batch_flatten(h2)
        rc_loss = rc_loss + weight * K.mean(K.square(h1 - h2), axis=-1)
    # rc_loss += K.mean(K.square(x - t_decoded), axis=(1, 2, 3))
    rc_loss += K.sum(K.maximum(t_decoded - 255, K.zeros_like(t_decoded)))
    rc_loss += K.sum(K.maximum(-t_decoded, K.zeros_like(t_decoded)))
    return rc_loss


def vae_dfc_loss(x, t_decoded):
    '''Total loss for the DFC VAE'''
    return K.mean(0.001 * perceptual_loss(x, t_decoded) + vae_dfc_kl_loss)


# def reconstruction_loss(x, t_decoded):
#     '''Reconstruction loss for the plain VAE'''
#     return K.sum(K.binary_crossentropy(
#         K.batch_flatten(x),
#         K.batch_flatten(t_decoded)), axis=-1)

# Keep only a single checkpoint, the best over test loss


checkpoint = ModelCheckpoint(filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             period=1,
                             mode='min')
# Create plain VAE model and associated KL divergence loss operation
vae, vae_kl_loss = create_vae(latent_dim, return_kl_loss_op=True)
vae_dfc, vae_dfc_kl_loss = create_vae(latent_dim, return_kl_loss_op=True)

use_pretrained = 1
if use_pretrained:
    print(filepath)
    vae_dfc.load_weights(filepath)
else:
    # vae_dfc.load_weights(filepath)
    adam = keras.optimizers.Adam(lr=1e-4)
    vae_dfc.compile(optimizer=adam, loss=vae_dfc_loss)
    # vae_dfc.compile(optimizer='rmsprop', loss=vae_dfc_loss)
    history = vae_dfc.fit_generator(train_generator,
                                    epochs=20,
                                    steps_per_epoch=100,
                                    shuffle=True,
                                    validation_data=val_generator,
                                    validation_steps=100,
                                    verbose=2,
                                    callbacks=[checkpoint])

    # adam = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0,
    #                              amsgrad=False)
    # vae.compile(optimizer="adam", loss=vae_loss)
    # history = vae.fit(x=x_train, y=x_train, epochs=epochs, batch_size=batch_size,
    #                   shuffle=True, validation_data=(x_test, x_test), verbose=2,
    #                   callbacks=[checkpoint])  # , SGDLearningRateTracker()])


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

# _test = (x_test - 128.) / 128.
# x_test = next(val_generator)[0]
selected_idx = np.random.choice(range(x_test.shape[0]), 10, replace=False)
# selected_idx = np.random.choice(range(x_train.shape[0]), 10, replace=False)
selected = x_test[selected_idx]
selected_dec_vae = encode_decode(vae_dfc, selected)
print("select:", "min:", selected.min(), "max:", selected.max(), "mean:", selected.mean())
print("select_vae:", "min:", selected_dec_vae.min(), "max:", selected_dec_vae.max(), "mean:", selected_dec_vae.mean())

plot_image_rows([selected, selected_dec_vae],
                ['Original images',
                 'Images generated by plain VAE'])

plt.figure()


def laplacian_variance(images):
    return [cv2.Laplacian(image, cv2.CV_32F).var() for image in images]


x_dec_vae = encode_decode(vae, x_test)
x_dec_vae_dfc = encode_decode(vae_dfc, x_test)

not_ones = y_test != 1
lvs_1 = laplacian_variance(x_test[not_ones])
lvs_2 = laplacian_variance(x_dec_vae[not_ones])
lvs_3 = laplacian_variance(x_dec_vae_dfc[not_ones])

plot_laplacian_variances(lvs_1, lvs_2, lvs_3, title='Laplacian variance of digit images (class != 1)')
plt.show()
