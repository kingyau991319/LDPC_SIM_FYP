import numpy as np
import random
import message_generator
import codegenerator

class LinearCodeGenerator:

    def __init__(self,parity_check_matrix,input_m_xlen,input_m_ylen,message_len):
        # if(input_m_ylen != message_len):
        #     print("error","for input_matrix cols is not equal to message cols")
        #     return -1
        self.input_matrix = input_matrix
        self.input_m_xlen = input_m_xlen
        self.input_m_ylen = input_m_ylen
        self.message_len = message_len

    def build_parity_check_matrix(self):
        
        input_matrix = self.input_matrix
        xlen = self.input_m_xlen
        ylen = self.input_m_ylen
        m_len = self.message_len

        check_m_len = xlen + m_len
        check_m_col = ylen

        parity_check_matrix = np.zeros((check_m_col,check_m_len))
        input_matrix = input_matrix.transpose()

        for y_axis in range(check_m_col):
            parity_check_matrix[y_axis][y_axis + message_len] = 1
            for x_axis in range(xlen):
                parity_check_matrix[y_axis][x_axis] = input_matrix[y_axis][x_axis]

        return parity_check_matrix

    def build_generator_matrix(self):
        
        input_matrix = self.input_matrix
        xlen = self.input_m_xlen
        ylen = self.input_m_ylen
        m_len = self.message_len

        generator_m_len = xlen + m_len
        generator_m_col = ylen
        print(xlen)
        print(generator_m_col)
        generator_martix = np.zeros((generator_m_col,generator_m_len))

        for y_axis in range(generator_m_col):
            generator_martix[y_axis][y_axis] = 1
            for x_axis in range(xlen):
                generator_martix[y_axis][x_axis + message_len] = input_matrix[y_axis][x_axis]

        return generator_martix

if __name__ == "__main__":
    message_len = 8
    input_matrix = []
    wc = 4
    wr = 2
    ylen = 8
    xlen = 8

    # messageBuilder = message_generator.LDPCMessage(message,message_len)
    matrixBuilder = codegenerator.CodeGenerator(wc,wr,xlen,ylen)
    input_matrix = matrixBuilder.matrixGeneratorForRegularLDPC()
    LDPCEncoder_test = LinearCodeGenerator(input_matrix,xlen,ylen,message_len)
    generator_martix = LDPCEncoder_test.build_generator_matrix()
    print(generator_martix)
