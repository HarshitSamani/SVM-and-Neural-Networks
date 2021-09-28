from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from problem_4.load_data import load_data_SVC


# SVM Classifier
def problem_4_SVM():
    try:
        os.remove('./results/problem_4_SVM.txt')
    except OSError:
        pass

    # Load data
    # X_train, Y_train, X_test, Y_test = load_data_SVC('Neuro_dataset')
    X = np.load("aal.npy",allow_pickle=True)
    Y = np.array(34*[1]+47*[0])
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=0)

    X_train, Y_train = shuffle(X_train, Y_train, random_state=0)
    X_test, Y_test = shuffle(X_test, Y_test, random_state=0)

    kernels = ['linear', 'poly', 'rbf', 'sigmoid']

    for kernel in kernels:
        clf = make_pipeline(StandardScaler(), SVC(kernel=kernel, gamma='scale'))
        clf.fit(X_train, Y_train)
        accuracy = clf.score(X_test, Y_test)

        print(
            f"The SVM classification accuracy with {kernel} Kernel is: {accuracy * 100:0.2f}%")
        with open('./results/problem_4_SVM.txt', "a") as file:
            file.write(
                f"The SVM classification accuracy with {kernel} Kernel is: {accuracy * 100:0.2f}%\n")
