import cv2
import mediapipe as mp
import time
import math
import numpy as np


class PoseDetector:

    def __init__(self, static_image_mode=False, model_complexity=1, smooth_landmarks=True, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.image_mode = static_image_mode
        self.model_complex = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(self.image_mode, self.model_complex, self.smooth_landmarks,
                                      self.min_detection_confidence, self.min_tracking_confidence)

    def find_pose(self, image, draw=True):
        # convert image to RGB to process
        image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(image_RGB)

        if self.results.pose_landmarks:
            # print(results.pose_landmarks)
            if draw:
                self.mp_drawing.draw_landmarks(image, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        return image

    def find_landmarks(self, image):
        self.landmarks_list = []

        if self.results.pose_landmarks:
            for id, landmark in enumerate(self.results.pose_landmarks.landmark):
                # convert coordinates into pixel values
                ih, iw, _ = image.shape
                cx, cy = int(landmark.x * iw), int(landmark.y * ih)
                self.landmarks_list.append([id, cx, cy])

        return self.landmarks_list

    def find_angle(self, image, p1, p2, p3, draw=True):
        # [landmark_index, x, y]
        x1, y1 = self.landmarks_list[p1][1:]
        x2, y2 = self.landmarks_list[p2][1:]
        x3, y3 = self.landmarks_list[p3][1:]

        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        if angle <= 0:
            angle = np.absolute(angle)

        if draw:
            # draw lines between chosen landmarks
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.line(image, (x2, y2), (x3, y3), (0, 255, 0), 2)
            # highlight the chosen landmarks
            cv2.circle(image, (x1, y1), 8, (255, 0, 0), cv2.FILLED)
            cv2.circle(image, (x1, y1), 15, (255, 0, 0), 2)
            cv2.circle(image, (x2, y2), 8, (255, 0, 0), cv2.FILLED)
            cv2.circle(image, (x2, y2), 15, (255, 0, 0), 2)
            cv2.circle(image, (x3, y3), 8, (255, 0, 0), cv2.FILLED)
            cv2.circle(image, (x3, y3), 15, (255, 0, 0), 2)
            # put angle
            cv2.putText(image, str(int(angle)), (x2 - 15, y2 - 15), cv2.FONT_HERSHEY_PLAIN, 2, (200, 200, 0), 2)

        return angle


def main():
    pTime = 0
    cTime = 0

    # downloaded a YT video and placed in ExerciseVideos folder
    cap = cv2.VideoCapture("ExerciseVideos/Video1.mp4")
    pose_detector = PoseDetector()

    while True:
        ret_val, image = cap.read()

        ret_image = pose_detector.find_pose(image)
        lms_list = pose_detector.find_landmarks(ret_image)
        if len(lms_list) > 0:
            print(lms_list[0])  # pointer to nose

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(ret_image, str(int(fps)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, cv2.COLOR_GRAY2BGR555)
        cv2.imshow("frame", ret_image)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
