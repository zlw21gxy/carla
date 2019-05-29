# import cv2
#
# vidcap = cv2.VideoCapture('carla_demo.mp4')
# success = True
# names = {"train", "val"}
# for name in names:
#     count = 0
#     while count < 51:
#         for i in range(2):
#             success, image = vidcap.read()
#         cv2.imwrite("/home/gu/project/predictive-filter-flow/mgPFF_video/data/DAVIS_videos/%s/carla_demo/%05d.jpg" % (name, count), image)  # save frame as JPEG file
#
#         print('Read a new frame: ', success)
#         count += 1
from vae_unit import load_mnist_data, plot_image_rows, load_carla_data
import numpy as np
import scipy.ndimage
(x_train, _), (x_test, _) = load_mnist_data(normalize=False)
# factor = 128/28
# data_1 = scipy.ndimage.zoom(x_train[:128], (1, factor, factor, 1), mode="nearest")
#
# # data_2 = scipy.ndimage.zoom(x_test, (1, factor, factor, 1))
# np.save("data_train", data_1)
# np.save("data_val", data_2)
data = x_train[:128]