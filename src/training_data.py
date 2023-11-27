import cv2
import math
import numpy as np
import os
from .constants import DATA_PATH
from .mediapipe_helper import MediaPipeHelper as MPH
from tensorflow.keras.utils import to_categorical  # type: ignore
from sklearn.model_selection import train_test_split
from .configuration import Configuration
from multiprocessing import Pool
from functools import partial


class TrainingData:
    def __init__(self, config: Configuration):
        self.config = config

    def _setup_folder_for_collection(self):
        for action in self.config.actions:
            for sequence in range(1, self.config.video_length + 1):
                try:
                    os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
                except:
                    pass

    def _load_sequence(self, sequence, action):
        window = []
        for frame_num in range(self.config.frame_length):
            res = np.load(
                os.path.join(
                    DATA_PATH,
                    action,
                    str(sequence),
                    "{}.npy".format(frame_num),
                )
            )
            window.append(res)
        return window

    def get(self, categorical=False):
        label_map = {label: num for num, label in enumerate(self.config.actions)}

        sequences, labels = [], []
        _i = 0

        with Pool(
            processes=4
        ) as pool:  # Adjust the number of processes based on your machine's capabilities
            for action in self.config.actions:
                _i += 1
                print(f"Gathering data {_i}/{len(self.config.actions)} - {action}...")

                # Create a partial function with the 'action' parameter fixed
                partial_load_sequence = partial(self._load_sequence, action=action)

                sequences_batch = pool.map(
                    partial_load_sequence,
                    np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int),
                )
                labels_batch = [label_map[action]] * len(sequences_batch)

                sequences.extend(sequences_batch)
                labels.extend(labels_batch)

        # for action in self.config.actions:
        #     _i += 1
        #     print(f"Gathering data {_i}/{len(self.config.actions)} - {action}...")
        #     for sequence in np.array(
        #         os.listdir(os.path.join(DATA_PATH, action))
        #     ).astype(int):
        #         window = []
        #         for frame_num in range(self.config.frame_length):
        #             res = np.load(
        #                 os.path.join(
        #                     DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)
        #                 )
        #             )
        #             window.append(res)
        #         sequences.append(window)
        #         labels.append(label_map[action])

        X = np.array(sequences)
        y = np.array(labels)

        if categorical:
            y = to_categorical(y).astype(int)

        return X, y

    def get_split(self, categorical=False):
        X, y = self.get(categorical)
        return train_test_split(X, y, test_size=0.2)

    def create(self):
        self._setup_folder_for_collection()
        cap = cv2.VideoCapture(1)
        # Set mediapipe model
        with MPH.get_holistic() as holistic:
            # NEW LOOP
            # Loop through actions
            for action in self.config.actions:
                # Loop through sequences aka videos
                for video_i in range(
                    self.config.start_folder,
                    self.config.start_folder + self.config.video_length,
                ):
                    # Loop through video length aka sequence length
                    for frame_i in range(self.config.frame_length):
                        # Read feed
                        ret, frame = cap.read()

                        # Make detections
                        image, results = MPH.detect(frame, holistic)

                        # Draw landmarks
                        MPH.draw_landmarks(image, results)

                        # Draw frame percent
                        _percent = math.floor(frame_i * 100 / self.config.frame_length)
                        cv2.rectangle(image, (0, 0), (200, 10), (0, 0, 0), -1)
                        cv2.rectangle(
                            image,
                            (0, 0),
                            (math.floor((200 / 100) * _percent), 10),
                            (0, 250, 0),
                            -1,
                        )

                        # Draw videos percent
                        _percent = math.floor(
                            video_i * 100 / self.config.video_length
                            + self.config.start_folder
                        )
                        cv2.rectangle(image, (400, 0), (600, 10), (0, 0, 0), -1)
                        cv2.rectangle(
                            image,
                            (400, 0),
                            (400 + math.floor((200 / 100) * _percent), 10),
                            (100, 100, 255),
                            -1,
                        )

                        # Show action at start and watit for 3 seconds
                        if video_i == self.config.start_folder and frame_i == 0:
                            cv2.putText(
                                image,
                                f"GET READY: {action}",
                                (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 255, 0),
                                4,
                                cv2.LINE_AA,
                            )

                            cv2.imshow("OpenCV Feed", image)
                            cv2.waitKey(3000)
                        else:
                            cv2.imshow("OpenCV Feed", image)

                        # Export keypoints
                        keypoints = MPH.extract_keypoints(results)
                        npy_path = os.path.join(
                            DATA_PATH, action, str(video_i), str(frame_i)
                        )
                        np.save(npy_path, keypoints)

                        # Break gracefully
                        if cv2.waitKey(10) & 0xFF == ord("q"):
                            break

            cap.release()
            cv2.destroyAllWindows()
