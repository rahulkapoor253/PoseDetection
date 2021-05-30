import cv2
import PoseDetectionModule as pdm
import time


pTime = 0
cTime = 0

# downloaded a YT video and placed in ExerciseVideos folder
cap = cv2.VideoCapture("ExerciseVideos/Video1.mp4")
pose_detector = pdm.PoseDetector()

while True:
    ret_val, image = cap.read()
    # to get image with drawn landmarks
    ret_image = pose_detector.find_pose(image)
    # to get list of landmarks
    lms_list = pose_detector.find_landmarks(ret_image)
    if len(lms_list) > 0:
        print(lms_list[0])  # pointer to nose

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(ret_image, str(int(fps)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, cv2.COLOR_GRAY2BGR555)
    cv2.imshow("frame", ret_image)
    cv2.waitKey(1)