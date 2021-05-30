import numpy as np
import time
import cv2
import PoseDetectionModule as pdm


cap = cv2.VideoCapture("TrainerVideos/TrainerVideo1.mp4")
pose_detector = pdm.PoseDetector()

count = 0  # to count total curls
dir = 0  # up and down for 1 curl

cTime = 0
pTime = 0

while True:
    ret_val, image = cap.read()
    image = cv2.resize(image, (1280, 720))
    # to get image with drawn landmarks by default
    ret_image = pose_detector.find_pose(image, draw=False)
    # to get list of landmarks
    lms_list = pose_detector.find_landmarks(ret_image)
    if len(lms_list) > 0:
        # For Left Arm
        angle = pose_detector.find_angle(ret_image, 11, 13, 15)
        perc = np.interp(angle, (40, 200), (0, 100))
        bar = np.interp(angle, (40, 200), (650, 100))
        # print(angle, perc)
        color = (255, 0, 255)
        if perc == 100:
            color = (0, 255, 0)
            if dir == 0:
                count += 0.5
                dir = 1
        if perc == 0:
            color = (0, 255, 0)
            if dir == 1:
                count += 0.5
                dir = 0

        # place int around perc and count to ignore 0.5 values
        cv2.rectangle(image, (1150, 100), (1175, 650), color, 3)
        cv2.rectangle(image, (1150, int(bar)), (1175, 650), color, cv2.FILLED)
        cv2.putText(image, f"{int(perc)}%", (1150, 75), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)

        cv2.putText(image, f"{int(count)}", (50, 650), cv2.FONT_HERSHEY_PLAIN, 15, (255, 0, 0), 15)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(ret_image, f"FPS {int(fps)}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0))
    cv2.imshow("frame", ret_image)
    cv2.waitKey(1)