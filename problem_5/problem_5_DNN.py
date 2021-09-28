import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from problem_5.load_data import load_data
from problem_5.parameters import *
from tensorflow.keras import regularizers
from tensorflow.keras import activations
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.initializers import he_normal


# DNN Classifier
def problem_5_DNN():
    try:
        os.remove('./results/problem_5_DNN.txt')
    except OSError:
        pass

    # Load data
    X_train, Y_train, X_test, Y_test = load_data('/EEG/eeg_data.csv')

    S = StandardScaler()
    X_train = S.fit_transform(X_train)
    X_test = S.transform(X_test)

    INPUT_SHAPE = X_train.shape[1:]
    init = he_normal()

    # Define model
    model = Sequential()
    # model.add(Flatten(input_shape=INPUT_SHAPE))
    model.add(Dense(128, input_dim=178, kernel_initializer=init, kernel_regularizer = regularizers.l2(0.01), activation='relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(256, kernel_initializer=init, kernel_regularizer = regularizers.l2(0.01), activation='relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(32, kernel_initializer=init, kernel_regularizer = regularizers.l2(0.01), activation='relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    # model.summary()

    #Set the optimizer values
    optimizer = RMSprop(lr=LEARNING_RATE,
                        rho=RHO,
                        epsilon=EPSILON,
                        decay=DECAY)

    # ReduceLROnPlateau callback to reduce learning rate when the validation accuracy plateaus
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                                patience=PATIENCE,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)

    # Early stopping callback to stop training if we are not making any positive progress
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=PATIENCE)

    callbacks = [learning_rate_reduction, early_stopping]

    # Train the model
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer="adam",
                  metrics=['accuracy'])

    model.fit(x=X_train,
              y=Y_train,
              epochs=50,
              batch_size=BATCH_SIZE,
              validation_split=VALIDATION_SPLIT)

    # Evaluate the model
    loss, accuracy = model.evaluate(x=X_test, y=Y_test)

    print(f"The DNN classification accuracy  is: {accuracy * 100:0.2f}%")
    with open('./results/problem_5_DNN.txt', "a") as file:
        file.write(f"The DNN classification accuracy is: {accuracy * 100:0.2f}%\n")
