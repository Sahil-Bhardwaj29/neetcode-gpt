import numpy as np
from numpy.typing import NDArray
from math import exp

class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        res = z.astype(np.float64).copy()
        mx = np.max(res)
        for i in range(len(res)):
            res[i] -= mx
        sm = 0
        for i in range(len(res)):
            sm +=  exp(res[i])

        for i in range(len(res)):
            res[i] = exp(res[i]) / sm
        return np.round(res,4)
        pass