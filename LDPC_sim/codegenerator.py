import numpy as np
import matplotlib.pyplot as plt 
import random
import time
import message_generator
import channel
import decoding
import warnings
import math

# what I want to do here
# to construct a parity-check matrix and generator matrix
class CodeGenerator:

    def __init__(self,wc,wr,xlen,ylen):
        self.wc = wc
        self.wr = wr # for regular matrix
        self.xlen = xlen
        self.ylen = ylen

    def shiftIdentityMatrix(self,len,shift):
        output_matrix = np.identity(len)
        shift = int(shift)
        for x_axis in range(len):
            output_matrix[x_axis] = np.roll(output_matrix[x_axis], shift)
        return output_matrix

    '''
        mode_flag = -1: shift_tmp = shift_tmp + 1
        mode_flag = 1: shift_tmp = input_matrix[y_axis][x_axis]
        mode_flag = 0: shift_tmp = (y_axis + x_axis) % identity_len
    '''

    def matrixGeneratorForQC_LDPC2(self,input_matrix,identity_len,mode_flag = -1):
        wr = self.wr
        wc = self.wc
        xlen = self.xlen
        ylen = self.ylen
        # input_matrix = self.matrixGeneratorForRegularLDPC()

        output_matrix = np.zeros((ylen * identity_len,xlen * identity_len))

        for y_axis in range(ylen):
            shift_tmp = -1
            for x_axis in range(xlen):  
                if input_matrix[y_axis][x_axis] >= 1:
                    if mode_flag == 0:
                        shift_tmp = (y_axis + x_axis) % identity_len
                    elif mode_flag == 1:
                        shift_tmp = input_matrix[y_axis][x_axis] - 1
                    else:
                        shift_tmp = shift_tmp + 1
                    tmp_matrix = self.shiftIdentityMatrix(identity_len,shift_tmp)
                    # shift_tmp = shift_tmp + 1
                    for y_axis2 in range(identity_len):
                        for x_axis2 in range(identity_len):
                            y_index = int(y_axis * identity_len + y_axis2)
                            x_index = int(x_axis * identity_len + x_axis2)
                            output_matrix[y_index][x_index] = int(tmp_matrix[y_axis2][x_axis2])

        self.wr = wr * identity_len
        self.wc = wc * identity_len
        self.xlen = xlen * identity_len
        self.ylen = ylen * identity_len

        return output_matrix

    #LDGM2
    #add row to make identity matrix for mlen in LDGM by LDPC
    def outputLDGMMatrix2(self,output_matrix):
        xlen = self.xlen
        ylen = self.ylen
        output_martix_copy = np.copy(output_matrix)

        tmp3_matrix = np.zeros((ylen,xlen))

        tmp_ylen = int((ylen * ylen - ylen) / 2)
        tmp_matrix = np.zeros((tmp_ylen,xlen))
        tmp_index_count = 0
        half_ylen = int(ylen / 2)

        for y_axis1 in range(half_ylen):
            for y_axis2 in range(ylen):
                for x_axis in range(xlen):
                    if y_axis1 != y_axis2:
                        tmp_matrix[tmp_index_count][x_axis] = (output_martix_copy[y_axis1][x_axis] + output_martix_copy[y_axis2][x_axis]) % 2

        tmp_index_count = ylen * half_ylen - half_ylen

        tmp2_matrix = np.zeros(xlen)
        success = np.zeros(ylen)

        for y_axis1 in range(ylen):
            for y_axis2 in range(tmp_index_count):
                success[y_axis1] = 0
                for x_axis in range(xlen):
                    tmp2_matrix[x_axis] = (tmp_matrix[y_axis2][x_axis] + output_martix_copy[y_axis1][x_axis]) % 2
                    if tmp2_matrix[x_axis] == 1 and x_axis >= ylen:
                        success[y_axis1] = success[y_axis1] + 1
                if success[y_axis1] == 1:
                    index = 0
                    for x_axis in range(ylen):
                        if tmp2_matrix[x_axis+ylen] == 1:
                            index = x_axis
                    for x_axis in range(xlen):
                        tmp3_matrix[index][x_axis] = tmp2_matrix[x_axis]
                    continue

        generator_matrix = np.zeros((ylen,xlen))
        for y_axis in range(ylen):
            generator_matrix[y_axis][y_axis] = 1
            for x_axis in range(ylen):
                generator_matrix[y_axis][ylen+x_axis] = tmp3_matrix[y_axis][x_axis]

        return generator_matrix

    def matrixGeneratorForRegularLDPC(self):

        wr = self.wr
        wc = self.wc
        xlen = self.xlen
        ylen = self.ylen
        half_ylen = int(ylen/2)

        # construct a zero matrix
        output_matrix = np.zeros((ylen,xlen))
        wr_count = np.zeros(ylen)
        wc_count = np.zeros(xlen)
        tmp = 0

        for y_axis in range(half_ylen):
            for x_axis in range(xlen):
                if wr_count[y_axis] < wc:
                    output_matrix[y_axis][x_axis + tmp] = 1
                    wr_count[y_axis] = wr_count[y_axis] + 1
            tmp = (tmp + wc) % xlen
        
        for x_axis in range(xlen):
            wc_count[x_axis] = 1

        tmp3 = 0
        for y_axis in range(half_ylen):
            tmp2 = tmp3
            for x_axis in range(wr):
                output_matrix[y_axis + half_ylen][tmp2] = 1
                wr_count[y_axis + half_ylen] = wr_count[y_axis + half_ylen] + 1
                wc_count[tmp2] = wc_count[tmp2] + 1
                tmp2 = (tmp2 + wc) % xlen
            tmp3 = tmp3 + 1

        return output_matrix

    def randomGeneratorCode(self):
        output_matrix = np.random.randint(2,size=(self.ylen,self.xlen))
        return output_matrix
    
    def oneGeneratorCode(self):
        output_martix = np.ones([self.ylen,self.xlen])

        output_martix[0][0] = 0
        output_martix[self.ylen-1][self.xlen-1] = 0

        return output_martix

    def oneGeneratorCodeNoTailHead(self):
        output_martix = np.ones([self.ylen,self.xlen])
        output_martix[0][0] = 0
        output_martix[self.ylen-1][self.xlen-1] = 0

        return output_martix

    def anotherSpecificCodewordTest(self,mult_num):
        input_matrix = np.ones([6,12])
        
        for y_axis in range(6):
            for x_axis in range(12):
                pass


        self.wr = 6
        self.wc = 3
        self.xlen = 8
        self.ylen = 4

        # print(mult_num)

        # if mult_num > 1:
        #     input_matrix = self.matrixGeneratorForQC_LDPC2(input_matrix,mult_num, 0)

        self.xlen = 8 * mult_num
        self.ylen = 4 * mult_num

        return input_matrix

    def specific_codeword_test(self,mult_num):
        input_matrix = np.zeros([7,14])

        for y_axis in range(7):
            for x_axis in range(14):
                input_matrix[y_axis][y_axis] = 2
                if x_axis < 7:
                    input_matrix[y_axis][7 + y_axis] = 1
                    input_matrix[(y_axis + 1) % 7][7 + y_axis] = 1
                    input_matrix[y_axis][(y_axis + 2)% 7] = 3
                    input_matrix[y_axis][(y_axis + 3)% 7] = 4

        identity_len = mult_num
        output_matrix = input_matrix

        self.wr = 5
        self.wc = 3
        self.xlen = 14
        self.ylen = 7

        input_matrix[0][13] = 0

        output_matrix = self.matrixGeneratorForQC_LDPC2(input_matrix,identity_len, 1)
        
        pre_generator_matrix = np.where(output_matrix==0,output_matrix,1)
        for y_axis in range(7*mult_num - mult_num):
            pre_generator_matrix[y_axis + mult_num] = np.add(pre_generator_matrix[y_axis + mult_num], pre_generator_matrix[y_axis]) % 2

        pre_generator_matrix = pre_generator_matrix.transpose()

        generator_matrix = np.zeros((7*mult_num,14*mult_num))

        for y_axis in range(7*mult_num):
            generator_matrix[y_axis][y_axis] = 1
            for x_axis in range(7*mult_num):
                generator_matrix[y_axis][x_axis+7*mult_num] = pre_generator_matrix[y_axis][x_axis]

        self.xlen = 14 * identity_len
        self.ylen = 7 * identity_len

        return output_matrix,generator_matrix

def regularLDPC_wr(wc,xlen,ylen):
    return int(wc * xlen / ylen)


if __name__ == "__main__":

    mult_num = 2
    wr = 4 * mult_num
    ylen = 4 * mult_num
    xlen = 8 * mult_num
    wc = 2 * mult_num

    codeGenerator = CodeGenerator(wc,wr,xlen,ylen)

    matrix = codeGenerator.anotherSpecificCodewordTest(mult_num)

    print("parity_check_matrix")
    print(matrix)