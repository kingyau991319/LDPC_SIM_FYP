import numpy as np
from numpy.polynomial import Polynomial
import json
import time
import pandas as pd
import os

class BCH_code:

    def __init__(self,noise_set):

        self.noise_set = noise_set

        parity_check_matrix = np.array([
            [1,0,0,0,1,0,0,1,1,0,1,0,1,1,1],
            [0,1,0,0,1,1,0,1,0,1,1,1,1,0,0],
            [0,0,1,0,0,1,1,0,1,0,1,1,1,1,0],
            [0,0,0,1,0,0,1,1,0,1,0,1,1,1,1],
            [1,0,0,0,1,1,0,0,0,1,1,0,0,0,1],
            [0,0,0,1,1,0,0,0,1,1,0,0,0,1,1],
            [0,0,1,0,1,0,0,1,0,1,0,0,1,0,1],
            [0,1,1,1,1,0,1,1,1,1,0,1,1,1,1]
        ])

        self.parity_check_matrix = np.transpose(parity_check_matrix)
        self.err_corr = []

        # 120 for 15C1 + 15C2
        for x in range(15):
            for y in range(15):
                if x > y:
                    continue
                new_arr = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                new_arr[x] = 1
                new_arr[y] = 1
                self.err_corr.append(new_arr)


    # (15,7) BCH code
    def BCH_15_7_codeword_generator(self, msg):

        msg = np.poly1d(msg)

        generator_matrix = np.poly1d(np.array([1,1,1,0,1,0,0,0,1]))
        codeword = msg * generator_matrix

        codeword = codeword.coeffs
        codeword = np.array(codeword) % 2
        codeword = codeword[::-1]
        while len(codeword) < 15:
            codeword = np.append(codeword,0)

        return codeword

    def BCH_15_7_codeword_decoding(self, msg):

        decoding_code = 0
        decoding_result = np.matmul(msg,self.parity_check_matrix) % 2

        # violant method to do with the decoding
        if np.sum(decoding_result) != 0:
            for x in range(120):
                msg_copy = msg.copy()
                msg_copy = np.add(msg_copy,self.err_corr[x]) % 2
                decoding_result2 = np.matmul(msg_copy,self.parity_check_matrix) % 2
                sum_of_result = np.sum(decoding_result2)
                if sum_of_result == 0:
                    return msg_copy

        return msg

    def linear_BCH_code_process(self):
        # generate message
        msg = np.random.randint(2, size = 7)

        # generator the codeword.
        codeword = self.BCH_15_7_codeword_generator(msg)
        msg_copy = np.copy(codeword)
        # BPSK
        codeword[codeword == 1] = -1
        codeword[codeword == 0] = 1

        # add noise
        recvcode = codeword + np.random.normal(0,self.noise_set,15)
        # recvcode_copy = np.copy(recvcode)
        hard_dec = np.copy(recvcode)
        # decoding
        hard_dec[hard_dec > 0] = 0
        hard_dec[hard_dec < 0] = 1
        hard_dec_deocding = self.BCH_15_7_codeword_decoding(hard_dec)

        # compare the original message and mark the result
        hamming_dist_msg = np.count_nonzero(hard_dec_deocding!=msg_copy)
        block_err = (hamming_dist_msg != 0)

        return hamming_dist_msg,block_err


    def cal_SNR(self):

        if self.noise_set <= 0:
            return -1
        sigma =  2 * (self.noise_set ** 2)
        SNRc = 1 / sigma

        SNRb = 15 / 7 * SNRc

        SNRcDB = 10 * np.log10(SNRc)
        SNRbDB = 10 * np.log10(SNRb)
        return round(SNRbDB,4),round(SNRcDB,4)

def code_run_process(noise_set,message_sending_time):

    code = BCH_code(noise_set)
    time_start = time.time()
    hamming_dist_msg_count = 0
    probaility_block_error = 0
    for i in range(message_sending_time):
        tmp_msg_count,block_count = code.linear_BCH_code_process()
        hamming_dist_msg_count = hamming_dist_msg_count + tmp_msg_count
        probaility_block_error = probaility_block_error + block_count
        print(i)
    count_time = time.time() - time_start
    prob_BER = hamming_dist_msg_count / (15 * message_sending_time)
    probaility_block_error = probaility_block_error / message_sending_time

    name = "BCH_15_7"
    SNRcDB,SNRbDB = code.cal_SNR()

    print("name:",name)
    print("BLER",probaility_block_error)
    print("BER:",prob_BER)
    print("Count_time:",count_time)
    print("Message sending time:",message_sending_time)
    print("SNRcDB",SNRcDB)
    print("SNRbDB",SNRbDB)


    ylen = 7
    xlen = 15

    result_list = [(name,xlen,ylen,message_sending_time,noise_set,prob_BER,SNRcDB,SNRbDB,probaility_block_error,0,0,0,0,count_time)]
    column = ['LDPC_code','xlen','ylen','message_sending_time','noise_set','average_probaility_error','SNRcDB','SNRbDB','Prob_block_error','detected_block_error','detected_bit_error','undetected_block_error','undetected_bit_error','count_time']

    df = pd.DataFrame(result_list, columns = column)

    if os.path.exists('sim_result.csv') == False:
        df.to_csv('sim_result.csv', mode='a', header=True)
    else:
        df.to_csv('sim_result.csv', mode='a', header=False)


if __name__ == "__main__":

    # test_val
    noise_set = 1
    message_sending_time = 10000
    code_run_process(noise_set,message_sending_time)

    noise_set = 0.9
    message_sending_time = 10000
    code_run_process(noise_set,message_sending_time)

    noise_set = 0.8
    message_sending_time = 10000
    code_run_process(noise_set,message_sending_time)

    noise_set = 0.75
    message_sending_time = 10000
    code_run_process(noise_set,message_sending_time)

    noise_set = 0.6
    message_sending_time = 10000
    code_run_process(noise_set,message_sending_time)

    noise_set = 0.5
    message_sending_time = 10000
    code_run_process(noise_set,message_sending_time)

    noise_set = 0.45
    message_sending_time = 100000
    code_run_process(noise_set,message_sending_time)

    noise_set = 0.4
    message_sending_time = 1000000
    code_run_process(noise_set,message_sending_time)

    # for k in range(100):

