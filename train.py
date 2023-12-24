import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from src.training_data import TrainingData
from src.hyperparameters import HyperParameters
from src.my_model import MyModel
from typing import Literal
from src.ai import AI
from src.constants import (
    BEGINNER_CONFIG,
    INTERMEDIATE_1_CONFIG,
    INTERMEDIATE_2_CONFIG,
    INTERMEDIATE_3_CONFIG,
    INTERMEDIATE_4_CONFIG,
)


def main():
    # ? WHAT TO DO
    # level = "BEGINNER"
    level: Literal[
        "BEGINNER",
        "INTERMEDIATE_1",
        "INTERMEDIATE_2",
        "INTERMEDIATE_3",
        "INTERMEDIATE_4",
    ] = "INTERMEDIATE_1"

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

    training_data = TrainingData(config)
    my_model = MyModel(config, training_data)
    hyperparameters = HyperParameters(my_model, training_data)

    # training_data.create()  #! CREATE DATA
    # hyperparameters.find()  #! HYPERPARAMETERS
    my_model.build(200)  #! BUILD


if __name__ == "__main__":
    main()
