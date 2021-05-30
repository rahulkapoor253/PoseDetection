import cv2
import mediapipe as mp
import time


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
        landmarks_list = []

        if self.results.pose_landmarks:
            for id, landmark in enumerate(self.results.pose_landmarks.landmark):
                # convert coordinates into pixel values
                ih, iw, _ = image.shape
                cx, cy = int(landmark.x * iw), int(landmark.y * ih)
                landmarks_list.append([id, cx, cy])
                cv2.circle(image, (cx, cy), 5, cv2.COLOR_GRAY2BGR, cv2.FILLED)

        return landmarks_list


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
