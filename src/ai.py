import json
from flask import Flask, request, render_template, Response, g
import cv2
from cv2 import VideoCapture
import numpy as np
from .mediapipe_helper import MediaPipeHelper as MPH
from .constants import TIMEOUT_SECONDS, THRESHOLD_NUM_MIN, filename_to_phrase
from tensorflow.keras.models import load_model  # type: ignore
from .configuration import Configuration
from typing import Literal
from time import time
import base64

model = load_model(f"model_BEGINNER.keras")


class AI:
    def __init__(self, config: Configuration):
        self.config = config

    _threshold = 0.60

    @staticmethod
    def _prob_viz(
        res,
        actions,
        input_frame,
        difficulty: str,
        target_filename: str,
    ):
        output_frame = input_frame.copy()
        _y = 17
        _width = 80 if difficulty == "BEGINNER" else 200
        text_y_offset = 85 if difficulty == "BEGINNER" else 350

        for num, prob in enumerate(res):
            filename = actions[num]
            is_target = filename == target_filename
            # Bakground
            cv2.rectangle(
                output_frame,
                (0, text_y_offset - 10 + num * _y),
                (_width, text_y_offset + 5 + num * _y),
                (146, 212, 143) if is_target else (255, 238, 192),
                -1,
            )

            # Probability bar
            cv2.rectangle(
                output_frame,
                (0, text_y_offset - 10 + num * _y),
                (int(prob * _width), text_y_offset + 5 + num * _y),
                # (205, 237, 204), # Green
                (193, 193, 253),
                -1,
            )

            # Outline
            cv2.rectangle(
                output_frame,
                (0, text_y_offset - 10 + num * _y),
                (_width, text_y_offset + 5 + num * _y),
                (0, 0, 0),
                1,
            )

            cv2.putText(
                output_frame,
                filename_to_phrase[filename],
                (10, text_y_offset + num * _y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        return output_frame

    @staticmethod
    def _draw_timer(input_frame, time_left_sec: float):
        output_frame = input_frame.copy()
        cv2.rectangle(output_frame, (0, 0), (640, 35), (192, 238, 255), -1)
        cv2.rectangle(output_frame, (0, 0), (640, 35), (0, 0, 0), 1)
        # cv2.rectangle(output_frame, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(
            output_frame,
            f"{time_left_sec:.2f}",
            (510, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        return output_frame

    @staticmethod
    def _draw_word(input_frame, word: str):
        print(word)
        output_frame = input_frame.copy()
        cv2.putText(
            output_frame,
            word,
            (20, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        return output_frame

    #! Start video
    def _start(
        self,
        mode: Literal["VIEW", "PREDICT"],
        action_index=0,
        image_base64="",
        cache=None,
    ):
        # # Remove the prefix if present
        # if image_base64.startswith("data:image/jpeg;base64,"):
        #     image_base64 = image_base64[len("data:image/jpeg;base64,") :]
        # return {"image": image_base64, "action_found": False}, 200
        # model = load_model(f"model_{self.config.difficulty}.keras")
        # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        # cv2.namedWindow("OpenCV Feed", cv2.WINDOW_FULLSCREEN)
        # cv2.setWindowProperty("OpenCV Feed", cv2.WND_PROP_TOPMOST, 1)
        # cv2.setWindowProperty(
        #     "OpenCV Feed", cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_FULLSCREEN
        # )
        threshold_num = 0
        sequence = cache.get("sequence") or []
        # Set mediapipe model
        with MPH.get_holistic() as holistic:
            start_time = time()
            print("Starting...")
            # loop_start_time = time()
            # Read feed
            # ret, image = cap.read()

            #! NEWLY ADDED
            # Remove the prefix if present
            if image_base64.startswith("data:image/jpeg;base64,"):
                image_base64 = image_base64[len("data:image/jpeg;base64,") :]

            # print("11111111111111111111111111111111111111111111111111111")
            # print(image_base64)

            image_bytes = base64.b64decode(image_base64)
            # print("222222222222222222222222222222222222222222222222222")
            # print(image_bytes)

            im_arr = np.frombuffer(
                image_bytes, dtype=np.uint8
            )  # im_arr is one-dim Numpy array
            # print("333333333333333333333333333333333333333333333333")
            # print(im_arr)
            image = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
            # image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
            # print("4444444444444444444444444444444444444444444444444444444")
            # print(image)

            height, width, channels = image.shape
            print(f"Length: {len(sequence)}")
            print("image resolution:", width, "x", height)

            # # Make detections
            # image, results = MPH.detect(image, holistic)

            # # Draw landmarks
            # MPH.draw_landmarks(image, results)

            #!!!
            # Convert the CV2 image to a bytes object
            retval, buffer = cv2.imencode(".jpg", image)
            image_bytes = buffer.tobytes()

            # Encode the bytes as base64
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")

            return {"image": image_base64, "action_found": False}, 200
            #!!!

            # 2. Prediction logic
            keypoints = MPH.extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-1 * self.config.frame_length :]

            if len(sequence) == self.config.frame_length:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]

                print(self.config.actions[np.argmax(res)])

                if mode == "PREDICT":
                    print(
                        f"cv2.getWindowImageRect('Frame'): {cv2.getWindowImageRect('OpenCV Feed')}"
                    )
                    time_left = TIMEOUT_SECONDS - (time() - start_time)
                    image = AI._draw_timer(image, time_left)
                    image = AI._draw_word(
                        image, filename_to_phrase[self.config.actions[action_index]]
                    )
                    image = AI._prob_viz(
                        res,
                        self.config.actions,
                        image,
                        self.config.difficulty,
                        self.config.actions[action_index],
                    )
                    # print(res[action_index])

                    # ? Found
                    if res[action_index] > AI._threshold:
                        threshold_num += 1

                    if threshold_num > THRESHOLD_NUM_MIN:
                        cv2.destroyAllWindows()
                        return {"action_found": True}

                    # ? Not found
                    if time_left < 0:
                        cv2.destroyAllWindows()
                        return {"action_found": False}

                elif mode == "VIEW":
                    image = AI._prob_viz(res, self.config.actions, image)

            # cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)

            # Show to screen

            cv2.imshow("OpenCV Feed", image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord("q") or cv2.waitKey(10) & 0xFF == 27:
                if mode == "PREDICT":
                    cv2.destroyAllWindows()
                    return {"action_found": False}

            # Convert the CV2 image to a bytes object
            retval, buffer = cv2.imencode(".jpg", image)
            image_bytes = buffer.tobytes()

            # Encode the bytes as base64
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")

            cache.set("sequence", sequence)
            print(f"Length: {len(sequence)}")

            return {"image": image_base64, "action_found": False}, 200

            # Encode the frame to JPEG
            # ret, jpeg = cv2.imencode(".jpg", image)
            # image_bytes = jpeg.tobytes()

            # Yield the frame for streaming
            # yield (
            #     b"--frame\r\n"
            #     b"Content-Type: image/jpeg\r\n\r\n" + image_bytes + b"\r\n\r\n"
            # )
            # additional_info = {
            #     "action_found": True,
            #     "prediction": {"class": "example"},
            # }
            # yield (
            #     b"--frame\r\n"
            #     b"Content-Type: image/jpeg\r\n\r\n"
            #     + frame_bytes
            #     + b"\r\n\r\n"
            #     + b"Content-Type: application/json\r\n\r\n"
            #     + json.dumps(additional_info).encode("utf-8")
            #     + b"\r\n\r\n"
            # )

            # loop_end_time = time()
            # print(f"Loop time: {loop_end_time - loop_start_time}")

    #! Start video checker
    def show_all(self):
        self._start("VIEW")

    #! Start video checker
    def check_if_performed(self, action: str, image_base64: str, cache):
        action_index = -1
        try:
            action_index = np.where(self.config.actions == action)[0][0]
        except IndexError:
            raise ValueError(
                f"Action {action} not found in {self.config.difficulty} level"
            )

        return self._start(
            "PREDICT", action_index=action_index, image_base64=image_base64, cache=cache
        )
        # found = self._start("VIEW", action_index=action_index)

        # return found if found else False
