import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import RMSprop,Adam
from tensorflow.keras.utils import plot_model
from problem_2.load_data import load_data
from problem_2.parameters import *
from matplotlib import pyplot as plt

# DNN Classifier
def problem_2_DNN():
    try:
        os.remove('./results/problem_2_DNN.txt')
    except OSError:
        pass

    noise_level = [0, 10, 25]
    for noise in noise_level:
        # Load data
        X_train, Y_train, X_test, Y_test = load_data(f'/Board/board_data_{noise}.txt')

        # Define model
        model = Sequential()
        model.add(Input(shape=DIMENSIONS))
        # model.add(Dense(units=64, activation='relu'))
        model.add(Dense(units=20, activation='relu'))
        # model.add(Dense(units=32, activation='relu'))
        model.add(Dense(units=NUM_CLASS, activation='softmax'))

        # Define optimizer
        optimizer = RMSprop(learning_rate=LEARNING_RATE)
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        # Train the model
        history = model.fit(x=X_train,
                            y=Y_train,
                            epochs=EPOCHS,
                            batch_size=BATCH_SIZE,
                            validation_split=0.1)

        plt.plot(history.history['loss'], label="train")
        plt.plot(history.history['val_loss'], label="validation")
        plt.xlabel("No. of epochs")
        plt.ylabel("loss value")
        plt.title(f"Noise-{noise}%")
        plt.legend()
        plt.show()
        # Evaluate the model
        loss, accuracy = model.evaluate(x=X_test, y=Y_test)

        print(f"The DNN classification accuracy {noise}% label noise is: {accuracy * 100:0.2f}%")
        with open('./results/problem_2_DNN.txt', "a") as file:
            file.write(f"The DNN classification accuracy {noise}% label noise is: {accuracy * 100:0.2f}%\n")
