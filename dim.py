import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import glob
import imageio
from keras.models import Model
from keras.layers import *
from keras import backend as K
from keras.optimizers import Adam
from keras.datasets import cifar10
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.datasets.cifar import load_batch
from tensorflow.python.keras.utils.data_utils import get_file
# from tensorflow.python.util.tf_export import keras_export
import os
import keras
from keras.preprocessing.image import ImageDataGenerator

img_dim = 256
lr = 1e-4
epochs = 601
z_dim = 256  # 隐变量维度
alpha = 0.5  # 全局互信息的loss比重
beta = 1.5  # 局部互信息的loss比重
gamma = 0.01  # 先验分布的loss比重
mode = "gen"
batch_size = 128
img_shape = (256, 256, 3)


def load_carla_data(normalize=False, num=None, flag="train"):
    if flag == "train":
        train_dir = "/home/gu/carla_out/train/*.jpg"
    elif flag == "test":
        train_dir = "/home/gu/carla_out2/train/*.jpg"
    file_dir = glob.glob(train_dir)
    if not num:
        num = len(file_dir)
    image_shape = (256, 256)
    i = 0
    images_train = []
    for file in file_dir:
        images_train.append(cv2.resize(cv2.imread(file), image_shape, interpolation=cv2.INTER_AREA))
        if i > num:
            break
        i += 1
    # x_train, x_test = train_test_split(np.stack(images_train), train_size=num-test_size-3, test_size=test_size)
    x_train = np.stack(images_train)
    if normalize:
        x_train = (x_train.astype('float32') - 128) / 128.
        x_train += np.random.uniform(0, 1 / 255, x_train.shape)
        # x_test = (x_test.astype('float32') - 128) / 128.
        # x_test += np.random.uniform(0, 1/255, x_test.shape)
    return x_train[:, :, :, ::-1]  # , x_test[:, :, :, ::-1]  # convert BGR to RGB


if mode == "load":
    x_train = load_carla_data(normalize=True, flag="train")

elif mode == "gen":

    f_p_train = lambda x: (x - 128 + 0 * np.random.randn(img_shape[0], img_shape[1], img_shape[2])) / 128
    f_p_test = lambda x: (x - 128) / 128
    # train_datagen = ImageDataGenerator(shear_range=7,
    #                                    horizontal_flip=True,
    #                                    preprocessing_function=f_p_train)
    train_datagen = ImageDataGenerator(preprocessing_function=f_p_train)

    train_generator = train_datagen.flow_from_directory(
        '/home/gu/carla_out2/train',
        target_size=img_shape[:2],
        batch_size=batch_size,
        class_mode='input')

    test_datagen = ImageDataGenerator(preprocessing_function=f_p_test)

    val_generator = test_datagen.flow_from_directory(
        '/home/gu/carla_out2/train',
        target_size=img_shape[:2],
        batch_size=batch_size,
        class_mode='input')

# x_train = x_train.astype('float32') / 255 - 0.5
# x_test = x_test.astype('float32') / 255 - 0.5
# y_train = y_train.reshape(-1)
# y_test = y_test.reshape(-1)
# 编码器（卷积与最大池化）
x_in = Input(shape=img_shape)
x = x_in

for i in range(3):
    x = Conv2D(int(z_dim / 2 ** (2 - i)),
               kernel_size=(3, 3),
               strides=(2, 2),
               padding='SAME')(x)

    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = MaxPooling2D((2, 2))(x)

feature_map = x  # 截断到这里，认为到这里是feature_map（局部特征）
feature_map_encoder = Model(x_in, x)

for i in range(2):
    x = Conv2D(z_dim,
               kernel_size=(3, 3),
               padding='SAME')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

x = GlobalMaxPooling2D()(x)  # 全局特征

z_mean = Dense(z_dim)(x)  # 均值，也就是最终输出的编码
z_log_var = Dense(z_dim)(x)  # 方差，这里都是模仿VAE的

encoder = Model(x_in, z_mean)  # 总的编码器就是输出z_mean


# 重参数技巧
def sampling(args):
    z_mean, z_log_var = args
    u = K.random_normal(shape=K.shape(z_mean))
    return z_mean + K.exp(z_log_var / 2) * u


# 重参数层，相当于给输入加入噪声
z_samples = Lambda(sampling)([z_mean, z_log_var])
prior_kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var))


# shuffle层，打乱第一个轴
def shuffling(x):
    idxs = K.arange(0, K.shape(x)[0])
    idxs = tf.random_shuffle(idxs)
    return K.gather(x, idxs)


# 与随机采样的特征拼接（全局）
z_shuffle = Lambda(shuffling)(z_samples)
z_z_1 = Concatenate()([z_samples, z_samples])
z_z_2 = Concatenate()([z_samples, z_shuffle])

# 与随机采样的特征拼接（局部）
feature_map_shuffle = Lambda(shuffling)(feature_map)
z_samples_repeat = RepeatVector(4 * 4)(z_samples)
z_samples_map = Reshape((4, 4, z_dim))(z_samples_repeat)
z_f_1 = Concatenate()([z_samples_map, feature_map])
z_f_2 = Concatenate()([z_samples_map, feature_map_shuffle])

# 全局判别器
z_in = Input(shape=(z_dim * 2,))
z = z_in
z = Dense(z_dim, activation='relu')(z)
z = Dense(z_dim, activation='relu')(z)
z = Dense(z_dim, activation='relu')(z)
z = Dense(1, activation='sigmoid')(z)

GlobalDiscriminator = Model(z_in, z)

z_z_1_scores = GlobalDiscriminator(z_z_1)
z_z_2_scores = GlobalDiscriminator(z_z_2)
global_info_loss = - K.mean(K.log(z_z_1_scores + 1e-6) + K.log(1 - z_z_2_scores + 1e-6))

# 局部判别器
z_in = Input(shape=(None, None, z_dim * 2))
z = z_in
z = Dense(z_dim, activation='relu')(z)
z = Dense(z_dim, activation='relu')(z)
z = Dense(z_dim, activation='relu')(z)
z = Dense(1, activation='sigmoid')(z)

LocalDiscriminator = Model(z_in, z)

z_f_1_scores = LocalDiscriminator(z_f_1)
z_f_2_scores = LocalDiscriminator(z_f_2)
local_info_loss = - K.mean(K.log(z_f_1_scores + 1e-6) + K.log(1 - z_f_2_scores + 1e-6))


def dim_loss(y_true, y_pred):
    return alpha * global_info_loss + beta * local_info_loss + gamma * prior_kl_loss


# 用来训练的模型
# model_train = Model(x_in, [z_z_1_scores, z_z_2_scores, z_f_1_scores, z_f_2_scores])
model_train = Model(x_in, z_mean)
# model_train.add_loss(alpha * global_info_loss + beta * local_info_loss + gamma * prior_kl_loss)


model_train.compile(optimizer=Adam(lr),
                    loss=dim_loss)
load_model = 0

if load_model:
    model_train.load_weights('total_model.cifar10.weights')
else:
    checkpointer = keras.callbacks.ModelCheckpoint(filepath='ckpt/dim.ckpt', verbose=1, save_best_only=False,
                                                   save_weights_only=True, period=1)
    # model_train.fit(x_train, epochs=epochs, batch_size=128, verbose=2, callbacks=[checkpointer])
    # model_train.save_weights('total_model.cifar10.weights')

    history = model_train.fit_generator(train_generator,
                                        epochs=20,
                                        steps_per_epoch=5,
                                        shuffle=True,
                                        validation_data=val_generator,
                                        validation_steps=5,
                                        verbose=2,
                                        callbacks=[checkpointer])

# 输出编码器的特征
zs = encoder.predict(x_train, verbose=True)
print("prior info", "mean", zs.mean(), "std", zs.std())
# if 0:
#     np.save("x_train", zs)
#     # np.save("y_train", y_train)
#     # zs_test = encoder.predict(x_test)
#     # np.save("x_test", zs_test)
#     # np.save("y_test", y_test)

# zs.mean()  # 查看均值（简单观察先验分布有没有达到效果）
# zs.std()  # 查看方差（简单观察先验分布有没有达到效果）
