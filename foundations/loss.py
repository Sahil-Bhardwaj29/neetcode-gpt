import numpy as np
from numpy.typing import NDArray
from math import log


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        for i in range(len(y_pred)):
            y_pred[i] += 1e-7
        L = 0
        sm = 0
        for i in range(len(y_true)):
            sm += y_true[i]*log(y_pred[i]) + (1 - y_true[i])*log(1 - y_pred[i])
        L = -sm/(len(y_true))
        return round(L,4)
        pass

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        for i in range(len(y_pred)):
            y_pred[i] += 1e-7
        L = 0
        sm = 0
        for i in range(len(y_true)):
            for j in range(len(y_true[0])):
                sm += y_true[i][j]*log(y_pred[i][j])
        L = -sm/(len(y_true))
        return round(L,4)
        pass
