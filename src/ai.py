import cv2
import numpy as np
from .mediapipe_helper import MediaPipeHelper as MPH
from tensorflow.keras.models import load_model  # type: ignore
from .configuration import Configuration
from typing import Literal
from .constants import BEGINNER_CONFIG, INTERMEDIATE_CONFIG
from time import time


class AI:
    def __init__(self, config: Configuration):
        self.config = config

    _colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]
    _threshold = 0.30
    _seconds_before_timeout = 10

    @staticmethod
    def _prob_viz(res, actions, input_frame):
        output_frame = input_frame.copy()
        _y = 17
        for num, prob in enumerate(res):
            cv2.rectangle(
                output_frame,
                (0, 75 + num * _y),
                (int(prob * 100), 90 + num * _y),
                AI._colors[0],
                -1,
            )
            cv2.putText(
                output_frame,
                actions[num],
                (0, 85 + num * _y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return output_frame

    #! Start video
    def _start(self, mode: Literal["VIEW", "PREDICT"], action_index=0):
        sequence = []
        model = load_model(f"model_{self.config.level}.keras")

        cap = cv2.VideoCapture(1)
        # Set mediapipe model
        with MPH.get_holistic() as holistic:
            start_time = time()
            print("Starting...")
            while cap.isOpened():
                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = MPH.detect(frame, holistic)

                # Draw landmarks
                MPH.draw_landmarks(image, results)

                # 2. Prediction logic
                keypoints = MPH.extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-1 * self.config.frame_length :]

                if len(sequence) == self.config.frame_length:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    # print(self.config.actions[np.argmax(res)])

                    if mode == "PREDICT":
                        print(res[action_index])

                        # ? Found
                        if res[action_index] > AI._threshold:
                            return True

                        # ? Not found
                        if time() - start_time > AI._seconds_before_timeout:
                            return False

                    elif mode == "VIEW":
                        image = AI._prob_viz(res, self.config.actions, image)

                cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)

                # Show to screen
                cv2.imshow("OpenCV Feed", image)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break
            cap.release()
            cv2.destroyAllWindows()

    #! Start video checker
    def show_all(self):
        self._start("VIEW")

    #! Start video checker
    def check_if_performed(self, action: str):
        action_index = -1
        try:
            action_index = np.where(self.config.actions == action)[0][0]
        except IndexError:
            raise ValueError(f"Action {action} not found in {self.config.level} level")

        found = self._start("PREDICT", action_index=action_index)

        return found if found else False
