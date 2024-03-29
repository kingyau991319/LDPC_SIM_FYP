import numpy as np
from numpy.polynomial import Polynomial
import json
import time
import pandas as pd
import os

class BCH_code:

    def __init__(self,noise_set,message_sending_time,BCH_type,iteration,clip_num):

        self.noise_set = noise_set
        self.message_sending_time = message_sending_time
        self.BCH_type = BCH_type
        self.iteration = iteration
        self.clip_num = clip_num

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

    def _message_to_LLRQAM(self, encode_out_msg,message_len):
        mess_to_LLR = np.array([])
        mult_form = 0.5 / (self.noise_set**2)

        for k in np.arange(message_len):
            if encode_out_msg[2*k] <= -2:
                mess_to_LLR = np.append(mess_to_LLR, 8 * encode_out_msg[2*k] + 8)
                mess_to_LLR = np.append(mess_to_LLR, 4 * encode_out_msg[2*k] + 8)
            elif (encode_out_msg[2*k] > -2) and (encode_out_msg[2*k] <= 0):
                mess_to_LLR = np.append(mess_to_LLR, 4 * encode_out_msg[2*k])
                mess_to_LLR = np.append(mess_to_LLR, 4 * encode_out_msg[2*k] + 8)

            elif (encode_out_msg[2*k] > 0) and (encode_out_msg[2*k] <= 2):
                mess_to_LLR = np.append(mess_to_LLR, 4 * encode_out_msg[2*k])
                mess_to_LLR = np.append(mess_to_LLR, 8 - 4 * encode_out_msg[2*k])
            else:
                mess_to_LLR = np.append(mess_to_LLR, 8 * encode_out_msg[2*k] - 8)
                mess_to_LLR = np.append(mess_to_LLR, 8 - 4 * encode_out_msg[2*k])

            if encode_out_msg[2*k+1] <= -2:
                mess_to_LLR = np.append(mess_to_LLR, 8 * encode_out_msg[2*k+1] + 8)
                mess_to_LLR = np.append(mess_to_LLR, 4 * encode_out_msg[2*k+1] + 8)
            elif (encode_out_msg[2*k+1] > -2) and (encode_out_msg[2*k+1] <= 0):
                mess_to_LLR = np.append(mess_to_LLR, 4 * encode_out_msg[2*k+1])
                mess_to_LLR = np.append(mess_to_LLR, 4 * encode_out_msg[2*k+1] + 8)
            elif (encode_out_msg[2*k+1] > 0) and (encode_out_msg[2*k+1] <= 2):
                mess_to_LLR = np.append(mess_to_LLR, 4 * encode_out_msg[2*k+1])
                mess_to_LLR = np.append(mess_to_LLR, 8 - 4 * encode_out_msg[2*k+1])
            else:
                mess_to_LLR = np.append(mess_to_LLR, 8 * encode_out_msg[2*k+1] - 8)
                mess_to_LLR = np.append(mess_to_LLR, 8 - 4 * encode_out_msg[2*k+1])

        mess_to_LLR = mult_form * mess_to_LLR
        mess_to_LLR = np.clip(mess_to_LLR,-self.clip_num,self.clip_num)
        # print(mess_to_LLR)

        return mess_to_LLR

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

    # product code(normal product code) with QAM 16

    def linear_BCH_code_process2(self):
        # generate message
        msg = np.random.randint(2, size = (7,7))
        prod_codeword = []

        # generator the codeword.
        for k in range(7):
            codeword = self.BCH_15_7_codeword_generator(msg[k])
            prod_codeword = np.append(prod_codeword,codeword)


        prod_codeword = np.resize(prod_codeword, (7,15))
        prod_codeword = prod_codeword.transpose()

        prod_codeword_col = np.array([])
        for k in range(15):
            codeword = self.BCH_15_7_codeword_generator(prod_codeword[k])
            prod_codeword_col = np.append(prod_codeword_col,codeword)

        # print("len(prod_codeword_col)",len(prod_codeword_col))

        codeword = prod_codeword_col
        codeword_copy = np.copy(prod_codeword_col)
        codeword_copy = np.resize(codeword_copy,(15,15))

        # add three padding to cross the QAM-16
        codeword = np.append(codeword,0)
        codeword = np.append(codeword,0)
        codeword = np.append(codeword,0)

        # encoder
        encode_out_msg = np.array([])
        # print("codeword_copy")
        # print(codeword_copy)

        for k in np.arange(57):

            if (codeword[4*k] == 0) and (codeword[4*k+1] == 0):
                encode_out_msg = np.append(encode_out_msg,-3)
            elif (codeword[4*k] == 0) and (codeword[4*k+1] == 1):
                encode_out_msg = np.append(encode_out_msg,-1)
            elif (codeword[4*k] == 1) and (codeword[4*k+1] == 0):
                encode_out_msg = np.append(encode_out_msg,3)
            else:
                encode_out_msg = np.append(encode_out_msg,1)

            if (codeword[4*k+2] == 0) and (codeword[4*k+3] == 0):
                encode_out_msg = np.append(encode_out_msg,-3)
            elif (codeword[4*k+2] == 0) and (codeword[4*k+3] == 1):
                encode_out_msg = np.append(encode_out_msg,-1)
            elif (codeword[4*k+2] == 1) and (codeword[4*k+3] == 0):
                encode_out_msg = np.append(encode_out_msg,3)
            else:
                encode_out_msg = np.append(encode_out_msg,1)

        # print("len(encode_out_msg)",len(encode_out_msg))
        # add noises
        rece_codeword = encode_out_msg + np.random.normal(0,self.noise_set,114)
        # decoding for QAM-16
        codeword_hard_decision = np.array([])
        
        for k in np.arange(57):
            if rece_codeword[2*k] <= -2:
                codeword_hard_decision = np.append(codeword_hard_decision, 0)
                codeword_hard_decision = np.append(codeword_hard_decision, 0)
            elif (rece_codeword[2*k] > -2) and (rece_codeword[2*k] <= 0):
                codeword_hard_decision = np.append(codeword_hard_decision, 0)
                codeword_hard_decision = np.append(codeword_hard_decision, 1)
            elif (rece_codeword[2*k] > 0) and (rece_codeword[2*k] <= 2):
                codeword_hard_decision = np.append(codeword_hard_decision, 1)
                codeword_hard_decision = np.append(codeword_hard_decision, 1)
            else:
                codeword_hard_decision = np.append(codeword_hard_decision, 1)
                codeword_hard_decision = np.append(codeword_hard_decision, 0)

            if rece_codeword[2*k+1] <= -2:
                codeword_hard_decision = np.append(codeword_hard_decision, 0)
                codeword_hard_decision = np.append(codeword_hard_decision, 0)
            elif (rece_codeword[2*k+1] > -2) and (rece_codeword[2*k+1] <= 0):
                codeword_hard_decision = np.append(codeword_hard_decision, 0)
                codeword_hard_decision = np.append(codeword_hard_decision, 1)
            elif (rece_codeword[2*k+1] > 0) and (rece_codeword[2*k+1] <= 2):
                codeword_hard_decision = np.append(codeword_hard_decision, 1)
                codeword_hard_decision = np.append(codeword_hard_decision, 1)
            else:
                codeword_hard_decision = np.append(codeword_hard_decision, 1)
                codeword_hard_decision = np.append(codeword_hard_decision, 0)

        recv_code_LLR = self._message_to_LLRQAM(rece_codeword,57)
        recv_code_LLR = recv_code_LLR[:224]
        recv_code_LLR = np.resize(recv_code_LLR,(15,15))
        # recv_code_LLR = np.clip(recv_code_LLR,-self.clip_num,self.clip_num)

        codeword_hard_decision = codeword_hard_decision[:224]
        codeword_hard_decision = np.resize(codeword_hard_decision,(15,15))

        # deocding part
        decoding_code = codeword_hard_decision.copy()

        for n in range(self.iteration):

            new_row_arr = np.array([])
            new_col_arr = np.array([])

            for k in range(15):
                new_row_arr = np.append(new_row_arr,self.BCH_15_7_codeword_decoding(decoding_code[k]))

            new_row_arr = np.resize(new_row_arr,(15,15))
            new_row_arr = new_row_arr.transpose()

            for k in range(7):
                new_col_arr = np.append(new_col_arr,self.BCH_15_7_codeword_decoding(new_row_arr[k]))
            for k in range(8):
                new_col_arr = np.append(new_col_arr,new_row_arr[k+7],axis=0)

            new_col_arr = np.resize(new_col_arr,(15,15))
            new_col_arr = new_col_arr.transpose()

            new_col_arr[new_col_arr == 1] = -1
            new_col_arr[new_col_arr == 0] = 1
            # print(recv_code_LLR)
            recv_code_LLR = np.add(recv_code_LLR,new_col_arr)

            # representation of Product Code
            decoding_code = recv_code_LLR.copy()
            decoding_code[decoding_code > 0] = 0
            decoding_code[decoding_code < 0] = 1


        recv_code_LLR[recv_code_LLR > 0] = 1
        recv_code_LLR[recv_code_LLR < 0] = 0
        # compare the original message and mark the result
        hamming_dist_msg = np.count_nonzero(recv_code_LLR!=codeword_copy)
        # print("hamming_dist_msg",hamming_dist_msg)
        block_err = (hamming_dist_msg != 0)

        return hamming_dist_msg,block_err

    def cal_SNR(self):

        if self.noise_set <= 0:
            return -1
        sigma =  2 * (self.noise_set ** 2)
        SNRc = 1 / sigma

        if self.BCH_type == 0:
            SNRb = 15 / 7 * SNRc
        # elif self.BCH_type == 1:
        else:
            SNRb = 225 / 49 * SNRc

        SNRcDB = 10 * np.log10(SNRc)
        SNRbDB = 10 * np.log10(SNRb)
        return round(SNRbDB,4),round(SNRcDB,4)

    def code_run_process_prepare(self):
        time_start = time.time()
        hamming_dist_msg_count = 0
        probaility_block_error = 0
        for i in range(self.message_sending_time):
            if BCH_type == 0:
                tmp_msg_count,block_count = self.linear_BCH_code_process()
            elif BCH_type == 1:
                tmp_msg_count,block_count = self.linear_BCH_code_process2()
            hamming_dist_msg_count = hamming_dist_msg_count + tmp_msg_count
            probaility_block_error = probaility_block_error + block_count
            print(i)
        count_time = time.time() - time_start
        if BCH_type == 0:
            prob_BER = hamming_dist_msg_count / (15 * self.message_sending_time)
        elif BCH_type == 1:
            prob_BER = hamming_dist_msg_count / (225 * self.message_sending_time)
        
        probaility_block_error = probaility_block_error / self.message_sending_time
        return count_time,prob_BER,probaility_block_error


def code_run_process(noise_set,message_sending_time,BCH_type,name,iteration=5,clip_num=3):

    code = BCH_code(noise_set,message_sending_time,BCH_type,iteration,clip_num)
    count_time,prob_BER,probaility_block_error = code.code_run_process_prepare()
    SNRcDB,SNRbDB = code.cal_SNR()

    print("name:",name)
    print("clip_num",clip_num)
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
    message_sending_time = 1
    # BCH_type -> 0 : BPSK 1 : QAM 16
    BCH_type = 1
    name = "BCH_product_code"

    noise_set = 0.75
    clip_num = 5
    message_sending_time = 100
    iteration = 100
    code_run_process(noise_set,message_sending_time,BCH_type,name,iteration,clip_num)
