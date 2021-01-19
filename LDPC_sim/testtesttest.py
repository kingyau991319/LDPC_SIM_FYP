import warnings
import math
import time
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import special
import pandas as pd
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt

import codegenerator
import decoding
import channel
import message_generator

class Init:

    '''
        channel_mode_flag = 0 : AWGN channel with sigma(noise_set)
        channel_mode_flag = 1 : BEC channel with sigma(noise_set)
    '''

    def __init__(self,noise_set,mult_num,LDPC_type,channel_mode_flag,wr,ylen,xlen,wc):
        self.noise_set = noise_set
        self.mult_num = mult_num
        self.LDPC_type = LDPC_type
        self.channel_mode_flag = channel_mode_flag
        self.wr = wr
        self.ylen = ylen
        self.xlen = xlen
        self.wc = wc

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

    def _printCodeInfo(self,generator_matrix,message_k,message,message_copy,sending_message,noise,final_message,bit_error_rate,message_lam,ylen):

        print("random generated message")
        print(message_copy)
        print("sending message")
        print(sending_message)
        print("--------------------------------------------------------------------")
        print("noise! ヽ(●´∀`●)ﾉ")
        print(noise)
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
        print("bit error rate")
        print((bit_error_rate * 100) / (3) ,"%")
        print("message_hard_decision")
        print(message_k)
        print("message_hard_decision and message bit hamming")
        print(np.count_nonzero(final_message!=message_k))
        print()


    def _primaryLDPC(self,mult_num):

        wr = 4 
        ylen = 3
        xlen = 5
        wc = 2

        parity_check_matrix = np.array([[1.0,1.0,0.0,1.0,0.0],[1.0,0.0,1.0,1.0,1.0]])
        generator_matrix = np.array([[1.0,0.0,0.0,1.0,0.0],[0.0,1.0,0.0,1.0,1.0],[0.0,0.0,1.0,0.0,1.0]])

        '''
            parity_check_matrix
            [1,1,0,1,0]
            [1,0,1,1,1]
            generator_matrix
            [1,0,0,1,0]
            [0,1,0,1,1]
            [0,0,1,0,1]
            message_len:3
            parity_check_matrix:2
        '''

        return wr,ylen,xlen,wc,parity_check_matrix,generator_matrix

    def _message_to_LLR(self,rece_message):
        # mess_to_LLR = np.clip(rece_message, -0.9,0.9)
        # mess_to_LLR = np.where(mess_to_LLR < 0,(1 + np.absolute(mess_to_LLR)) / ( 1 - np.absolute(mess_to_LLR)),(1 - mess_to_LLR) / (1 + mess_to_LLR)) 
        # mess_to_LLR = np.log(mess_to_LLR)
        mess_to_LLR = (- 2 / (self.noise_set**2)) * rece_message
        return mess_to_LLR

    def signal_tranmission_repeat(self,message_sending_time,print_flag):

        warnings.filterwarnings('ignore')
        mult_num,noise_set = self.mult_num,self.noise_set

        ylen,xlen = self.ylen,self.xlen

        #generator matrix and parity-check matrix
        wr,ylen,xlen,wc,parity_check_matrix,generator_matrix = self._primaryLDPC(mult_num)

        #channel
        message_channel = channel.channel(noise_set,xlen)

        #decoding_tool
        decoding_tool = decoding.decoding_algorithm(parity_check_matrix,2,5)

        #return result
        average_probaility_error = 0
        average_probaility_error_summation = 0
        no_bit_error_flag_sum = 0
        iteration = 10

        if print_flag == 1:
            self._printCodeSetMatrix(xlen,ylen,generator_matrix,parity_check_matrix)

        for k in range(message_sending_time):
            # self.message_sneding_simulation2(noise_set,ylen,xlen,generator_matrix,parity_check_matrix,message_channel,decoding_tool,print_flag,iteration)
            right_rate,no_bit_error_flag = self.message_sending_simulation(noise_set,ylen,generator_matrix,parity_check_matrix,message_channel,decoding_tool,print_flag,iteration)
            average_probaility_error_summation = average_probaility_error_summation + right_rate
            no_bit_error_flag_sum = no_bit_error_flag_sum + no_bit_error_flag
            if k % 100 == 0:
                print(k)

        prob_block_right = no_bit_error_flag_sum / message_sending_time
        average_probaility_error = average_probaility_error_summation / (message_sending_time * 3)

        return average_probaility_error,prob_block_right

    def message_sending_simulation(self,noise_set,ylen,generator_matrix,parity_check_matrix,message_channel,decoding_tool,print_flag,iteration):

            # message generator
            message = np.random.randint(2, size = ylen)
            message_copy = np.copy(message)

            # building_sending_message
            message = np.matmul(message,generator_matrix) % 2
            message[message == 0] = -1

            # add noise
            message = np.add(message,message_channel.AWGN_channel())

            # receive the code and hard decision and count
            message_hard_decision = np.where(message > 0,1,0)
            message_hard_decision = np.matmul(parity_check_matrix,message_hard_decision) % 2

            if message_hard_decision.sum() != 0:
                message_LLR = self._message_to_LLR(message)
                out_message = decoding_tool.sumProductAlgorithmWithIteration(message_LLR,iteration)
                hamming_distance_for_in_out_mess = np.count_nonzero(message_copy!=out_message[:ylen])
            else:
                hamming_distance_for_in_out_mess = 0

            return hamming_distance_for_in_out_mess,(hamming_distance_for_in_out_mess == 0)

    def cal_SNR(self):
        if self.noise_set <= 0:
            return -1
        sigma =  2 * (self.noise_set ** 2)
        SNRc = 1 / sigma
        SNRb = (5/3) * SNRc
        SNRcDB = 10 * np.log10(SNRc)
        SNRbDB = 10 * np.log10(SNRb)
        return round(SNRbDB,4),round(SNRcDB,4)


    def get_block_data(self,prob_block_right,message_sending_time):
        return 1 - prob_block_right

    def getNoiseSetBySNR(self,snr):
        snr = 4 * (10 ** (snr / 10))
        sigma = 1/snr
        sigma = sigma ** (1/2)
        return sigma

def third_LDPC_code_running(noise_set,message_sending_time):

    time_start = time.time()
    mult_num = 1
    wr = 3
    ylen = 3 * mult_num
    xlen = 5 * mult_num
    wc = 2

    print_flag = 0
    name = "test_code_AWGN"

    init = Init(noise_set,mult_num,0,0,wr,ylen,xlen,wc)
    average_probaility_error,prob_block_right = init.signal_tranmission_repeat(message_sending_time,print_flag)

    SNRbDB,SNRcDB = init.cal_SNR()
    print("average_probaility_error",average_probaility_error * 100, "%")

    print("noise_set",init.noise_set)
    print("SNRbDB",SNRbDB)
    print("SNRcDB",SNRcDB)

    expection = init.get_block_data(prob_block_right,message_sending_time)
    print("expection",expection)
    print("message_sending_time",message_sending_time)

    time_end = time.time()
    count_time = time_end - time_start
    count_time = round(count_time,2)
    print("use_time",count_time)

    result_list = [(name,xlen,ylen,message_sending_time,noise_set,average_probaility_error,SNRcDB,SNRbDB,expection,count_time)]

    column = ['LDPC_code','xlen','ylen','message_sending_time','noise_set','average_probaility_error','SNRcDB','SNRbDB','Prob_block_error','count_time']
    
    # if the csv file is empty
    # df = pd.DataFrame(result_list, columns = column)
    # df.to_csv('sim_result.csv')

    df = pd.read_csv('sim_result.csv',index_col=0)
    df_new = pd.DataFrame(result_list, columns = column)
    df = df.append(df_new)
    df.to_csv('sim_result.csv')

if __name__ == "__main__":

    noise_set = 0.7
    message_sending_time = 100000
    third_LDPC_code_running(noise_set,message_sending_time)

    # mult_num = 1
    # wr = 3
    # ylen = 3
    # xlen = 5
    # wc = 2

    # noise_set = 1
    # print_flag = 0

    # init = Init(noise_set,mult_num,0,0,wr,ylen,xlen,wc)
    # init.signal_tranmission()
