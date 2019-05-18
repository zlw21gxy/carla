import cv2

vidcap = cv2.VideoCapture('carla_demo.mp4')
success = True
names = {"train", "val"}
for name in names:
    count = 0
    while count < 51:
        for i in range(2):
            success, image = vidcap.read()
        cv2.imwrite("/home/gu/project/predictive-filter-flow/mgPFF_video/data/DAVIS_videos/%s/carla_demo/%05d.jpg" % (name, count), image)  # save frame as JPEG file

        print('Read a new frame: ', success)
        count += 1
