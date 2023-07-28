import cv2
import mediapipe as mp
import numpy as np


class PoseDetector:

    def __init__(self, static_image_mode=False, model_complexity=1, smooth_landmarks=True,
                 enable_segmentation=False, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5,):

        self.mode = static_image_mode
        self.model = model_complexity
        self.segmentation = enable_segmentation
        self.smooth = smooth_landmarks
        self.detection = min_detection_confidence
        self.tracking = min_tracking_confidence

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(self.mode, self.model, self.segmentation,
                                      self.smooth, self.detection, self.tracking)
        self.stage = "up"
        self.count = 0
        
    #نمایش محل قرار گیری اندام‌های بدن بر روی تصویر
    def find_pose(self, img, draw=True):
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image, (640, 480))
        img_color = cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR)
        img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(image_resized)
        if self.results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(img_color, self.results.pose_landmarks,
                                            self.mp_pose.POSE_CONNECTIONS)
        return img_color

    # محاسبه محل قرار گیری اندام‌های بدن بر روی تصویر
    def find_position(self, img_color, draw=True):
        self.landmark_list = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                lm = self.results.pose_landmarks.landmark[id]
                h, w, c = img_color.shape

                cx, cy = int(lm.x * w), int(lm.y * h)
                self.landmark_list.append([id,cx,cy])

                if draw:
                    cv2.circle(img_color, (cx, cy), 2, (255, 0, 0), cv2.FILLED)

        return self.landmark_list

    # محاسبه زوایای مفاصل بدن با استفاده از محل قرار گیری اندام‌های بدن
    def find_angle(self, img,p1, p2, p3, draw=True):

        x1, y1 = self.landmark_list[p1][1:]
        x2, y2 = self.landmark_list[p2][1:]
        x3, y3 = self.landmark_list[p3][1:]
        lm_id = self.landmark_list[p2][0]
         # a = np.array(a)  # First
        # b = np.array(b)  # Mid
        # c = np.array(c)  # End
        radians = np.arctan2(y3 - y2, x3 - x2) - np.arctan2(y1 - y2, x1 - x2)
        angle = np.abs(radians * 180.0 / np.pi)
        angle = int(angle)
        if angle > 180:
            angle = 360 - angle

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 5)

            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 5)
           
            cv2.circle(img, (x1, y1), 5, (0, 0, 255), 5)
   
            cv2.circle(img, (x2, y2), 5, (0, 0, 255), 5)

            cv2.circle(img, (x3, y3), 5, (0, 0, 255), 5)

            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            return angle

    # نمایش تعداد حرکات صحیح بر روی تصویر
    def put_text(self, img, dict_a):
        for key in dict_a.keys():  
            cv2.putText(img, str(key), (10,int(dict_a[key][8])*40+60),
            cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                
            cv2.putText(img, str(dict_a[key][6]), (250,int(dict_a[key][8])*40+60),
            cv2.FONT_HERSHEY_PLAIN, 2, (156, 3, 3), 2)
            
            cv2.putText(img, str(dict_a[key][5]), (340,int(dict_a[key][8])*40+60),
            cv2.FONT_HERSHEY_PLAIN, 2, (8, 117, 19), 2)
            
        return img