from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D  # type: ignore
from .training_data import TrainingData
from .configuration import Configuration
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint  # type: ignore
import os


class MyModel:
    def __init__(self, config: Configuration, training_data: TrainingData):
        self.config = config
        self.training_data = training_data

    def create(
        self,
        optimizer="adam",
        activation="relu",
        dropout_rate=0.5,
        kernel_size=3,
        pool_size=2,
        lstm_units=64,
        num_dense_units=64,
    ):
        model = Sequential()

        # Convolutional layers
        model.add(
            Conv1D(
                64,
                kernel_size=kernel_size,
                activation=activation,
                input_shape=(self.config.frame_length, 1662),
            )
        )
        model.add(MaxPooling1D(pool_size=pool_size))
        model.add(Conv1D(128, kernel_size=kernel_size, activation=activation))
        model.add(MaxPooling1D(pool_size=pool_size))

        # LSTM layers
        model.add(LSTM(lstm_units, return_sequences=True, activation=activation))
        model.add(LSTM(lstm_units, return_sequences=False, activation=activation))

        # Fully connected layers
        model.add(Dense(num_dense_units, activation=activation))
        model.add(Dropout(dropout_rate))
        model.add(Dense(num_dense_units // 2, activation=activation))
        model.add(Dropout(dropout_rate))

        # Output layer
        model.add(Dense(len(self.config.actions), activation="softmax"))

        # Compile model
        model.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
        )

        return model

    def build(self, epochs: int):
        X_train, X_test, y_train, y_test = self.training_data.get_split(
            categorical=True
        )

        log_dir = os.path.join("Logs")
        tb_callback = TensorBoard(log_dir=log_dir)

        model = self.create()

        checkpoint = ModelCheckpoint(
            "model.keras",
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        )

        model.fit(
            X_train,
            y_train,
            epochs=epochs,
            callbacks=[tb_callback, checkpoint],
            validation_data=(X_test, y_test),
        )

        model.save("model_final.keras")
