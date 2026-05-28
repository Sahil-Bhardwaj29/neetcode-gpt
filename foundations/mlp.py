import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        h = x
        z =0
        for i in range(len(weights)-1):
            z = np.dot(h,weights[i]) + biases[i]
            h = np.maximum(0,z)
        z = np.maximum(0,z)
        y = np.dot(h,weights[len(weights)-1])+biases[len(biases)-1]
        return np.round(y,5)

        pass
