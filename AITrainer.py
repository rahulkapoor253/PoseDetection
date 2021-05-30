import numpy as np
import time
import cv2
import PoseDetectionModule as pdm


cap = cv2.VideoCapture("TrainerVideos/TrainerVideo1.mp4")
pose_detector = pdm.PoseDetector()

while True:
    # ret_val, image = cap.read()
    # image = cv2.resize(image, (1280, 720))
    image = cv2.imread("TrainerVideos/bicep_curl.jpg")
    # to get image with drawn landmarks by default
    ret_image = pose_detector.find_pose(image)
    # to get list of landmarks
    lms_list = pose_detector.find_landmarks(ret_image)
    if len(lms_list) > 0:
        print(lms_list)

    cv2.imshow("frame", ret_image)
    cv2.waitKey(1)