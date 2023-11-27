from sklearn.model_selection import GridSearchCV

# from sklearn.model_selection import StratifiedKFold
from scikeras.wrappers import KerasClassifier
from .training_data import TrainingData
from .my_model import MyModel
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


class HyperParameters:
    def __init__(self, my_model: MyModel, training_data: TrainingData):
        self.my_model = my_model
        self.training_data = training_data

    def get(self):
        return {
            "optimizer": "adam",
            "activation": "relu",
            "dropout_rate": 0.5,
            "kernel_size": 3,
            "pool_size": 2,
            "lstm_units": 64,
            "num_dense_units": 64,
        }

    def find(self):
        print("Finding hyperparameters...")
        print(self.my_model)
        print(self.my_model.create())

        def build_fn(
            pool_size=2,
        ):
            return self.my_model.create(pool_size=pool_size)

        # Wrapper function for KerasClassifier
        model = KerasClassifier(
            model=build_fn,
            epochs=130,
            batch_size=5,
            verbose=0,
        )

        # Hyperparameters to tune
        param_grid = {
            "optimizer": ["adam", "sgd", "rmsprop"],
            "activation": ["relu", "tanh", "sigmoid"],
            "dropout_rate": [0.0, 0.2, 0.4],
            "lstm_units": [64, 128],
            "num_dense_units": [64, 128],
        }

        print("Defining StratifiedKFold cross-validator...")

        # Define the StratifiedKFold cross-validator
        kfold = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        print("Creating and configuring GridSearchCV...")

        # Create and configure GridSearchCV
        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring="accuracy",
            verbose=3,
            n_jobs=-1,
            error_score="raise",
            cv=kfold,
        )

        print("Getting training data...")

        # Get training data
        X, y = self.training_data.get(categorical=True)

        print(f"X: {X.shape}")
        print(f"y: {y.shape}")

        print("Running GridSearchCV...")

        # Run GridSearchCV
        grid_result = grid.fit(X, y)

        # Print results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_["mean_test_score"]
        stds = grid_result.cv_results_["std_test_score"]
        params = grid_result.cv_results_["params"]
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
