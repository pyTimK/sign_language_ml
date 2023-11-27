import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from src.training_data import TrainingData
from src.hyperparameters import HyperParameters
from src.my_model import MyModel
from src.ai import AI
from src.constants import BEGINNER_CONFIG, INTERMEDIATE_CONFIG


def main():
    # ? WHAT TO DO
    level = "INTERMEDIATE"
    config = BEGINNER_CONFIG if level == "BEGINNER" else INTERMEDIATE_CONFIG
    training_data = TrainingData(config)
    my_model = MyModel(config, training_data)
    hyperparameters = HyperParameters(my_model, training_data)
    ai = AI(config)

    # training_data.create()  #! CREATE DATA
    hyperparameters.find()  #! HYPERPARAMETERS
    # my_model.build(350)  #! BUILD
    # ai.show_all()  #! START - show all
    # result = ai.check_if_performed("7_cr")  #! START - predict
    # print(f"Result: {result}")


if __name__ == "__main__":
    main()
