import warnings
import math
import time

import numpy as np
from scipy import special
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from numba import jit

import codegenerator
import decoding
import channel
import message_generator

class Init:

    '''
        channel_mode_flag = 0 : AWGN channel with sigma(noise_set)
        channel_mode_flag = 1 : BEC channel with sigma(noise_set)
    '''

    def __init__(self,noise_set,mult_num,LDPC_type,channel_mode_flag,wr,ylen,xlen,wc,iteration,clip_num,filename=""):
        self.noise_set = noise_set
        self.len_mult_num = mult_num
        self.LDPC_type = LDPC_type
        self.channel_mode_flag = channel_mode_flag
        self.wr = wr
        self.ylen = ylen
        self.xlen = xlen
        self.wc = wc
        self.clip_num = clip_num
        sigma =  2 * (self.noise_set ** 2)
        SNRc = 1 / sigma
        self.SNRc = SNRc
        self.iteration = iteration
        self.filename = filename


    # for single case and print the result
    def signal_tranmission(self):
        self.signal_tranmission_repeat(1,1)

    def _printCodeSetMatrix(self,xlen,ylen,generator_matrix,parity_check_matrix):

            print("xlen",xlen)
            print("ylen",ylen)
            print("generator_matrix")
            print(generator_matrix)
            print("parity-check-matrix")
            print(parity_check_matrix)
            print("--------------------------------------------------------------------")
            print()

    def _printCodeInfo(self,generator_matrix,message,message_copy,final_message,hamming_distance,message_lam,ylen):

        print("random generated message")
        print(message_copy)
        print("--------------------------------------------------------------------")
        print("receiving message")
        print(message)
        print("message_lam")
        print(message_lam)
        print("final message with checking")
        print(final_message)
        print("final message")
        print(final_message[:ylen])
        print("original message")
        message_copy = np.matmul(message_copy,generator_matrix) % 2
        print(message_copy)
        print("hamming_distance")
        print(hamming_distance)
        print()


    def _primaryLDPC(self,f_name):

        def _gf2elim(M):

            m,n = M.shape

            i=0
            j=0

            while i < m and j < n:
                # find value and index of largest element in remainder of column j
                k = np.argmax(M[i:, j]) +i

                # swap rows
                #M[[k, i]] = M[[i, k]] this doesn't work with numba
                temp = np.copy(M[k])
                M[k] = M[i]
                M[i] = temp


                aijn = M[i, j:]

                col = np.copy(M[:, j]) #make a copy otherwise M will be directly affected

                col[i] = 0 #avoid xoring pivot row with itself

                flip = np.outer(col, aijn)

                M[:, j:] = M[:, j:] ^ flip

                i +=1
                j +=1

            return M

        f = open(f_name,"r")

        str = f.read()
        str = str.split()
        str = np.array(str)
        str_len = len(str)
        input_num = str.astype(np.int)

        xlen = input_num[0]
        ylen = input_num[1]
        code_rate = ylen / xlen
        wc = input_num[2]
        wr = input_num[3]
        wc_num = input_num[xlen+4:xlen+ylen+4]

        lower_bound_of_array = str_len - np.sum(wc_num)

        input_num = input_num[lower_bound_of_array:]
        parity_check_matrix = np.zeros((ylen,xlen))
        x_index_index = 0

        for y_axis in range(ylen):
            for x_axis in range(wc_num[y_axis]):
                x_index = input_num[x_index_index] - 1
                parity_check_matrix[y_axis][x_index] = 1
                x_index_index = x_index_index + 1

        parity_check_matrix = parity_check_matrix.astype(np.uint8)
        parity_check_matrix_copy = np.zeros((ylen,xlen))
        generator_matrix = np.zeros((ylen,xlen))

        # what i need to do first
        # 1. change the parity check matrix
        # 2. apply gf2elim
        # 3. transpose and make generator matrix

        for y_axis in range(ylen):
            for x_axis in range(ylen):
                parity_check_matrix_copy[y_axis][x_axis] = int(parity_check_matrix[y_axis][x_axis+ylen])
                parity_check_matrix_copy[y_axis][x_axis+ylen] = int(parity_check_matrix[y_axis][x_axis])

        parity_check_matrix_copy = parity_check_matrix_copy.astype(np.uint8)

        M = _gf2elim(parity_check_matrix_copy)

        M2 = np.zeros((ylen,ylen))

        for y_axis in range(ylen):
            for x_axis in range(ylen):
                M2[y_axis][x_axis] = M[y_axis][x_axis+ylen]

        M2 = np.transpose(M2)

        for y_axis in range(ylen):
            generator_matrix[y_axis][y_axis] = 1
            for x_axis in range(ylen):
                generator_matrix[y_axis][x_axis+ylen] = M2[y_axis][x_axis]

        return wr,ylen,xlen,wc,parity_check_matrix,generator_matrix

    def _secondLDPC(self,len_mult_num):
        pass


    def _thridLDPC(self,len_mult_num):
        self.wr,self.wc,self.xlen,self.ylen = 4,3,14,7
        wr,wc,xlen,ylen = 4,3,14,7
        codegenerator_sim = codegenerator.CodeGenerator(wc,wr,xlen,ylen)
        parity_check_matrix,generator_matrix = codegenerator_sim.specific_codeword_test(len_mult_num)
        wr,wc,xlen,ylen = 4,3,14*len_mult_num,7*len_mult_num

        return wr,ylen,xlen,wc,parity_check_matrix,generator_matrix

    def _message_to_LLR(self,rece_message):
        mess_to_LLR = (2 / (self.noise_set**2)) * rece_message
        mess_to_LLR = np.clip(mess_to_LLR,-self.clip_num,self.clip_num)
        return mess_to_LLR

    def signal_tranmission_repeat(self,message_sending_time,print_flag):

        warnings.filterwarnings('ignore')
        len_mult_num,noise_set = self.len_mult_num,self.noise_set

        ylen,xlen = self.ylen,self.xlen

        #generator matrix and parity-check matrix
        if self.LDPC_type == 0:
            wr,ylen,xlen,wc,parity_check_matrix,generator_matrix = self._primaryLDPC(self.filename)
        elif self.LDPC_type == 1:
            wr,ylen,xlen,wc,parity_check_matrix,generator_matrix = self._secondLDPC(len_mult_num)
        elif self.LDPC_type == 2:
            wr,ylen,xlen,wc,parity_check_matrix,generator_matrix = self._thridLDPC(len_mult_num)

        #channel
        message_channel = channel.channel(noise_set,xlen)

        #decoding_tool
        decoding_tool = decoding.decoding_algorithm(parity_check_matrix,ylen,xlen,clip_num)

        #return result
        average_probaility_error_summation = 0
        bit_error_flag_sum = 0
        undetected_block_error_num = 0
        undetected_block_hamming_dist = 0
        detected_block_error_num = 0
        detected_block_hamming_dist = 0


        if print_flag == 1:
            self._printCodeSetMatrix(xlen,ylen,generator_matrix,parity_check_matrix)
        #undetected error = 0
        for k in range(message_sending_time):
    
            codeword_hamming_dist,no_bit_error_flag,no_bit_type_flag,codeword_hard_hamming_dist = self.message_sending_simulation(ylen,generator_matrix,parity_check_matrix,message_channel,decoding_tool,print_flag,self.iteration)
            # hamming_distance_out,no_bit_error_flag = self.message_sending_simulation2(noise_set,ylen,xlen,generator_matrix,parity_check_matrix,message_channel,decoding_tool,print_flag)
            average_probaility_error_summation = average_probaility_error_summation + codeword_hamming_dist
            bit_error_flag_sum = bit_error_flag_sum + no_bit_error_flag
            if no_bit_type_flag == 0:
                detected_block_error_num = detected_block_error_num + 1
                detected_block_hamming_dist = detected_block_hamming_dist + codeword_hamming_dist
                print("iteration:", k, "| hamming_distance:" , codeword_hamming_dist," | detected_error"," | codeword_hard_hamming_dist:",codeword_hard_hamming_dist)
            elif no_bit_type_flag == 1:
                undetected_block_error_num = undetected_block_error_num + 1
                undetected_block_hamming_dist = undetected_block_hamming_dist + codeword_hamming_dist
                print("iteration:", k, "| hamming_distance:" , codeword_hamming_dist," | undetected_error"," | codeword_hard_hamming_dist:",codeword_hard_hamming_dist)

        probaility_block_error = bit_error_flag_sum / message_sending_time
        prob_detected_block_error = detected_block_error_num / message_sending_time
        prob_undetected_block_error = undetected_block_error_num / message_sending_time
        
        prob_BER = average_probaility_error_summation / (message_sending_time * ylen)
        prob_detected_BER = detected_block_hamming_dist / (message_sending_time * ylen)
        prob_undetected_BER = undetected_block_hamming_dist / (message_sending_time * ylen)

        return prob_BER,probaility_block_error,prob_detected_block_error,prob_detected_BER,prob_undetected_block_error,prob_undetected_BER

    '''
        for running product code
        1.message -> the passing message
        2.message_copy -> for compare the final_message and output the hamming distance
        3.out_message -> hard_decision work
    '''

    def message_sending_simulation2(self,noise_set,ylen,xlen,generator_matrix,parity_check_matrix,message_channel,decoding_tool,print_flag):

        # message generator
        message = np.random.randint(2,size=(ylen,ylen))
        message_copy = message.copy() # to backup and compare the result

        # mult it and transpose and matmul one more time
        message = np.matmul(message,generator_matrix)
        message = np.transpose(message)
        message = np.matmul(message,generator_matrix) % 2
        print("sending message")
        print(message)
        message[message == 1] = -1
        message[message == 0] = 1

        # noise with AWGN
        message = message + np.random.normal(0,noise_set,(xlen,xlen))

        # '''
        #     Encoding part 
        #     ------------------------------------------------------------
        #     Decoding part
        #     1.hard_decision and output the H*(X^T), 
        #     if the row and cols is equal to (0^T),then just end.

        #     2.otherwise, do the SPA alogrithm to the cols and rows, using hamming distance to count it,

        #     3.If hamming distance(C,R) == 0, then output, 
        #     hamming distance(C,R) != 0 and < t, then
        #     SPA again, If > t, then declearing failure.

        # '''

        # hard_decision use_later
        out_message_hard_decision = message.copy()
        out_message_hard_decision[out_message_hard_decision >=0] = 0
        out_message_hard_decision[out_message_hard_decision < 0] = 1
        out_message_hard_decision_transpose_col = out_message_hard_decision.copy().transpose()

        out_message_hard_decision_row = out_message_hard_decision[:ylen,:xlen]
        out_message_hard_decision_col = out_message_hard_decision_transpose_col[:ylen,:xlen]

        # I need to do the SPA for both
        message_that_LLR = message.copy()
        message_that_LLR = self._message_to_LLR(message_that_LLR)

        message_that_LLR_row = message_that_LLR[:ylen,:xlen]
        message_that_LLR_col = message_that_LLR.copy().transpose()
        message_that_LLR_col = message_that_LLR_col[:ylen,:xlen]

        decoding_message_row = np.array([])
        decoding_message_col = np.array([])

        for k in range(ylen):
            decoding_message_row = np.append(decoding_message_row,decoding_tool.sumProductAlgorithmWithIteration(message_that_LLR_row[k],self.iteration),axis=0)
            decoding_message_col = np.append(decoding_message_col,decoding_tool.sumProductAlgorithmWithIteration(message_that_LLR_col[k],self.iteration),axis=0)

        decoding_message_row = np.resize(decoding_message_row,(ylen,xlen))
        decoding_message_col = np.resize(decoding_message_col,(ylen,xlen))

        hamming_distance_row = np.zeros(ylen)
        hamming_distance_col = np.zeros(ylen)

        col_message = decoding_message_col[:ylen,:ylen].copy()
        row_message = decoding_message_row.copy().transpose()
        row_message = row_message[:ylen,:ylen]

        for k in range(ylen):
            hamming_distance_row[k] = np.count_nonzero(out_message_hard_decision_row[k]!=decoding_message_row[k])
            hamming_distance_col[k] = np.count_nonzero(out_message_hard_decision_col[k]!=decoding_message_col[k])
            hamming_distance_row_message = np.count_nonzero(row_message!=message_copy)
            hamming_distance_col_message = np.count_nonzero(col_message!=message_copy)


        if print_flag == 1:
            print("out_message_hard_decision_row")
            print(out_message_hard_decision_row)
            print("out_message_hard_decision_col")
            print(out_message_hard_decision_col)
            print("decoding_message_row")
            print(decoding_message_row)
            print("decoding_message_col")
            print(decoding_message_col)
            print("hamming_distance_row_for_hard_decision_and_after_SPADecoding")
            print(hamming_distance_row)
            print("hamming_distance_col_for_hard_decision_and_after_SPADecoding")
            print(hamming_distance_col)
            print("t")
            print(ylen/2)
            print("message_copy")
            print(message_copy)
            print("row_message")
            print(row_message)
            print("hamming_distance_row_message")
            print(hamming_distance_row_message)
            print("col_message")
            print(col_message)
            print("hamming_distance_col_message")
            print(hamming_distance_col_message)


        # message_hamming_distance = np.count_nonzero(message_copy!=out_message[:ylen])

        # hamming distance count between hard-decision and final summation
        # if count > t (ylen/2) , that is with IBDD....

        return 0,0

    def message_sending_simulation(self,ylen,generator_matrix,parity_check_matrix,message_channel,decoding_tool,print_flag,iteration):

        # message generator
        # message = np.random.randint(2, size = ylen)
        message = np.zeros(ylen)
        message_copy = np.copy(message)

        # building_sending_message
        codeword = np.matmul(message,generator_matrix) % 2

        codeword[codeword == 1] = -1
        codeword[codeword == 0] = 1

        rece_codeword = np.add(codeword,message_channel.AWGN_channel())

        # for hard decision check to avoid the use of SPA
        codeword_hard_decision = rece_codeword.copy()
        codeword_hard_decision[codeword_hard_decision > 0] = 0
        codeword_hard_decision[codeword_hard_decision < 0] = 1

        hard_decision_check = np.matmul(parity_check_matrix,codeword_hard_decision) % 2
        hard_decision_mult_sum = np.sum(hard_decision_check)
        if (hard_decision_mult_sum == 0):
            return 0,0,-1,0

        message_LLR = self._message_to_LLR(rece_codeword)
        out_message = decoding_tool.sumProductAlgorithmWithIteration(message_LLR,iteration)
        message_hamming_dist = np.count_nonzero(message_copy!=out_message[:ylen])
        codeword_hard_hamming_dist = np.count_nonzero(message_copy!=codeword_hard_decision[:ylen])
        out_message = np.matmul(parity_check_matrix,out_message) % 2
        out_message_flag = np.sum(out_message)

        block_error_flag = (message_hamming_dist == 0)
        block_error_type_flag = -1

        if message_hamming_dist != 0 and out_message_flag > 0:
            block_error_type_flag = 0 # undetected error
        elif message_hamming_dist != 0:
            block_error_type_flag = 1 # detected error

        if print_flag == 1:
            self._printCodeInfo(generator_matrix,message,message_copy,out_message,message_hamming_dist,message_LLR,ylen)

        return message_hamming_dist,block_error_flag,block_error_type_flag,codeword_hard_hamming_dist

    def cal_SNR(self):

        if self.noise_set <= 0:
            return -1
        sigma =  2 * (self.noise_set ** 2)
        SNRc = 1 / sigma
        SNRb = 2 * SNRc
        SNRcDB = 10 * np.log10(SNRc)
        SNRbDB = 10 * np.log10(SNRb)
        return round(SNRbDB,4),round(SNRcDB,4)

    def get_block_error(self,prob_block_right,message_sending_time):
        return 1 - prob_block_right

    def getNoiseBySNR(self,snr):
        snr = 4 * (10 ** (snr / 10))
        sigma = 1/snr
        sigma = sigma ** (1/2)
        return sigma

def third_LDPC_code_running(xlen,ylen,len_mult_num,noise_set,message_sending_time,iteration,name,LDPC_type,clip_num,filename=""):

    time_start = time.time()
    wr = 10
    wc = 5

    init = Init(noise_set,len_mult_num,LDPC_type,0,wr,ylen,xlen,wc,iteration,clip_num,filename)
    prob_BER,prob_block_right,prob_detected_block_error,prob_detected_BER,prob_undetected_block_error,prob_undetected_BER = init.signal_tranmission_repeat(message_sending_time,print_flag = 0)
    SNRbDB,SNRcDB = init.cal_SNR()
    print("probaility_bit_error",prob_BER * 100, "%")

    print("noise_set",init.noise_set)
    print("SNRbDB",SNRbDB)
    print("SNRcDB",SNRcDB)

    probaility_block_error = init.get_block_error(prob_block_right,message_sending_time)
    prob_undetected_block_error = init.get_block_error(prob_undetected_block_error,message_sending_time)
    prob_undetected_block_error = init.get_block_error(prob_undetected_block_error,message_sending_time)

    print("probility_block_error",probaility_block_error)
    print("message_sending_time",message_sending_time)

    time_end = time.time()
    count_time = time_end - time_start
    count_time = round(count_time,2)
    print("use_time",count_time)
    ylen = int(ylen)
    xlen = int(xlen)

    result_list = [(name,xlen,ylen,message_sending_time,noise_set,prob_BER,SNRcDB,SNRbDB,probaility_block_error,prob_detected_block_error,prob_detected_BER,prob_undetected_block_error,prob_undetected_BER,count_time)]

    column = ['LDPC_code','xlen','ylen','message_sending_time','noise_set','average_probaility_error','SNRcDB','SNRbDB','Prob_block_error','detected_block_error','detected_bit_error','undetected_block_error','undetected_bit_error','count_time']

    # if the csv file is empty
    # df = pd.DataFrame(result_list, columns = column)
    # df.to_csv('sim_result.csv')

    df = pd.DataFrame(result_list, columns = column)
    df.to_csv('sim_result.csv', mode='a', header=False)


def runOnlyOneTimeLDPC():

    len_mult_num = 1
    wr = 6
    ylen = 7 * len_mult_num
    xlen = 14 * len_mult_num
    wc = 3

    noise_set = 1
    print_flag = 1
    iteration = 10
    LDPC_type = 2


    init = Init(noise_set,len_mult_num,LDPC_type,0,wr,ylen,xlen,wc,iteration)
    init.signal_tranmission()

if __name__ == "__main__":


    # runOnlyOneTimeLDPC()
    # name = "third_LDPC_code_1792_896"
    # name = "third_LDPC_code_896_448"
    # name = "third_LDPC_code_448_224"
    # name = "third_LDPC_code_224_112"
    # name = "third_LDPC_code_112_56"
    # name = "third_LDPC_code_56_28"
    # name = "MacKeyCode_816.55.178"
    
    clip_num = 25

    name = "MacKeyCode_96.3.963" + "clip_num" + str(clip_num)
    filename = "test_code_816.55.178"


    # 0 -> primary_LDPC_code, 1 -> second_LDPC_code, 2 -> third_LDPC_code
    LDPC_type = 0
    # name = "MacKeyCode_408.3.854_clip" + str(clip_num)

    len_mult_num = 1

    iteration = 20
    xlen,ylen = 96,48
    filename = "test_code_96.3.963"

    clip_num = 8
    name = "MacKeyCode_96.3.963" + "clip_num" + str(clip_num)
    noise_set,message_sending_time = 0.5,10000
    third_LDPC_code_running(xlen,ylen,len_mult_num,noise_set,message_sending_time,iteration,name,LDPC_type,clip_num,filename)
