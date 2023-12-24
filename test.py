from src.ai import AI, cv2, VideoCapture
from typing import Literal
from src.constants import (
    BEGINNER_CONFIG,
    INTERMEDIATE_1_CONFIG,
    INTERMEDIATE_2_CONFIG,
    INTERMEDIATE_3_CONFIG,
    INTERMEDIATE_4_CONFIG,
)


def check_if_performed(action: str):
    config = (
        BEGINNER_CONFIG
        if action in BEGINNER_CONFIG.actions
        else INTERMEDIATE_1_CONFIG
        if action in INTERMEDIATE_1_CONFIG.actions
        else INTERMEDIATE_2_CONFIG
        if action in INTERMEDIATE_2_CONFIG.actions
        else INTERMEDIATE_3_CONFIG
        if action in INTERMEDIATE_3_CONFIG.actions
        else INTERMEDIATE_4_CONFIG
        if action in INTERMEDIATE_4_CONFIG.actions
        else None
    )

    if config is None:
        raise Exception("Action not found")

    ai = AI(config)

    result = ai.check_if_performed(action)  #! START - predict
    print(f"Result: {result}")
    return result


def show_all(
    level: Literal[
        "BEGINNER",
        "INTERMEDIATE_1",
        "INTERMEDIATE_2",
        "INTERMEDIATE_3",
        "INTERMEDIATE_4",
    ] = "INTERMEDIATE_1",
):
    config = (
        BEGINNER_CONFIG
        if level == "BEGINNER"
        else INTERMEDIATE_1_CONFIG
        if level == "INTERMEDIATE_1"
        else INTERMEDIATE_2_CONFIG
        if level == "INTERMEDIATE_2"
        else INTERMEDIATE_3_CONFIG
        if level == "INTERMEDIATE_3"
        else INTERMEDIATE_4_CONFIG
        if level == "INTERMEDIATE_4"
        else None
    )

    if config is None:
        raise Exception("Action not found")

    ai = AI(config)

    ai.show_all()  #! START - show all


if __name__ == "__main__":
    show_all()
    # check_if_performed("18_grandpa")
