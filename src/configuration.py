import numpy as np
from typing import Literal


class Configuration:
    def __init__(
        self,
        difficulty: Literal["BEGINNER", "INTERMEDIATE"],
        actions: np.ndarray,
        video_length=50,
        frame_length=30,
        start_folder=1,
    ):
        self.difficulty = difficulty
        self.actions = actions
        self.video_length = video_length
        self.frame_length = frame_length
        self.start_folder = start_folder
