import numpy as np
from numpy.typing import NDArray
from math import exp

class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        mx = np.max(z)
        for i in range(len(z)):
            z[i] -= mx
        sm = 0
        for i in range(len(z)):
            sm +=  exp(z[i])

        for i in range(len(z)):
            z[i] = exp(z[i]) / sm
        return np.round(z,4)
        pass