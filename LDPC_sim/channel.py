import numpy as np
from numba import jitclass
from numba import jitclass, types, typed
from numba import jit

@jitclass([('threshold', types.double),
           ('message_len', types.int64)])
class channel:

    def __init__(self,threshold,message_len):
        self.threshold = threshold
        self.message_len = message_len

    def AWGN_channel(self):
        noise = np.random.normal(0,self.threshold,self.message_len)
        return noise

    def BEC_channel(self,message):
        threshold = self.threshold
        threshold = 1 - self.threshold
        for x_axis in range(self.message_len):
            random_num = np.random.rand()
            random_num2 = np.random.rand()
            if random_num > threshold:
                message[x_axis] = np.random.rand()
                if random_num2 > 0.5:
                    message[x_axis] = message[x_axis] * -1

        return message
                

if __name__ == "__main__":
    # message = [1,1,1,1,1,1,-1,-1,-1,-1,-1]
    test = channel(0.1,10)
    noise = test.AWGN_channel()
    print(noise)
