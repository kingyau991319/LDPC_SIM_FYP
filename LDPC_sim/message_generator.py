import numpy as np
import random


# LDPCMessage : for constructing a message code here(randomly or fixed)
class LDPCMessage:

    def __init__(self,messagelen):
        self.messagelen = messagelen

    # printing message_code here
    def printCode(self):
        print("length:" , self.messagelen)

    def randomCode(self):
        message_code = np.random.randint(2, size = self.messagelen)
        return message_code

    def zeroCode(self):
        message_code = np.zeros((1,self.messagelen))[0]
        return message_code

    def fullCode(self):
        message_code = np.ones((1,self.messagelen))[0]
        return message_code