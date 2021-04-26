import numpy as np
from numpy.polynomial import Polynomial
import json
import time
import pandas as pd
import os

import numpy.polynomial.polynomial as pp
from numpy.polynomial import Polynomial as P

import lookup_table_for_IBBR as funcLUC

class BCH_code:

    def __init__(self,noise_set,message_sending_time,BCH_type,iteration):

        self.noise_set = noise_set
        self.message_sending_time = message_sending_time
        self.BCH_type = BCH_type
        self.iteration = iteration

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

        syndrome_mapping_matrix = np.array([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1],
            [1,1,0,0],
            [0,1,1,0],
            [0,0,1,1],
            [1,1,0,1],
            [1,0,1,0],
            [0,1,0,1],
            [1,1,1,0],
            [0,1,1,1],
            [1,1,1,1],
            [1,0,1,1],
            [1,0,0,1],
        ])

        self.syndrome_mapping_matrix = syndrome_mapping_matrix

        self.parity_check_matrix = np.transpose(parity_check_matrix)

        self.ref = pd.read_csv("decoding_ref.csv")

    def change_type_ref(self,msg):
        msg_ref = 0
        for k in range(8):
            msg_ref = msg_ref + (int(msg[k]) << k)
        return msg_ref

    def BCH_15_7_codeword_decoding(self, codeword):

        def poly_to_numpy_arr(poly, length):
            arr = poly.coeffs
            arr = arr[::-1]
            arr = np.abs(arr) % 2
            while len(arr) < length:
                arr = np.append(arr, 0)
            return arr

        # test case
        # codeword = np.array([0,0,0,1,0,1,0,0,0,0,0,0,1,0,0])
        codeword = codeword[::-1]


        minimal_polynomials1 = np.poly1d(np.array([1,0,0,1,1]))
        minimal_polynomials2 = np.poly1d(np.array([1,1,1,1,1]))

        syndrome = np.matmul(codeword,self.parity_check_matrix) % 2

        if np.sum(syndrome) != 0:

            # reverse the code for polynomial showing in numpy mode
            codeword_poly= np.poly1d(codeword)
            codeword_poly1 = codeword_poly / minimal_polynomials1
            codeword_poly2 = codeword_poly / minimal_polynomials2
            
            codeword_poly1,codeword_poly2 = poly_to_numpy_arr(codeword_poly1[1],4),poly_to_numpy_arr(codeword_poly2[1],4)
            
            # poly for power
            syndrome1 = codeword_poly1.copy()
            syndrome2 = codeword_poly1.copy()
            syndrome3 = codeword_poly2.copy()
            syndrome4 = codeword_poly1.copy()

            while len(syndrome1) < 15:
                syndrome1 = np.append(syndrome1, 0)
                syndrome2 = np.append(syndrome2, 0)
                syndrome3 = np.append(syndrome3, 0)
                syndrome4 = np.append(syndrome4, 0)

            for k in np.arange(4):
                if k == 0:
                    continue
                if int(syndrome2[k]) == 1:
                    syndrome2[k] = 0
                    syndrome2[k * 2] = 1
                if int(syndrome3[k]) == 1:
                    syndrome3[k] = 0
                    syndrome3[k * 3] = 1
                if int(syndrome4[k]) == 1:
                    syndrome4[k] = 0
                    syndrome4[k * 4] = 1

            # mult the ref list and output the final result of the syndrome
            syndrome1 = np.matmul(syndrome1,self.syndrome_mapping_matrix) % 2
            syndrome2 = np.matmul(syndrome2,self.syndrome_mapping_matrix) % 2
            syndrome3 = np.matmul(syndrome3,self.syndrome_mapping_matrix) % 2
            syndrome4 = np.matmul(syndrome4,self.syndrome_mapping_matrix) % 2

            for k in np.arange(14):
                if np.array_equal(syndrome1,self.syndrome_mapping_matrix[k]):
                    syndrome1 = np.zeros(15)
                    syndrome1[k] = 1
                if np.array_equal(syndrome2,self.syndrome_mapping_matrix[k]):
                    syndrome2 = np.zeros(15)
                    syndrome2[k] = 1
                if np.array_equal(syndrome3,self.syndrome_mapping_matrix[k]):
                    syndrome3 = np.zeros(15)
                    syndrome3[k] = 1
                if np.array_equal(syndrome4,self.syndrome_mapping_matrix[k]):
                    syndrome4 = np.zeros(15)
                    syndrome4[k] = 1

            syndrome1 = syndrome1[::-1]
            syndrome2 = syndrome2[::-1]
            syndrome3 = syndrome3[::-1]
            syndrome4 = syndrome4[::-1]

            syndrome1_poly = np.poly1d(syndrome1)
            syndrome2_poly = np.poly1d(syndrome2)
            syndrome3_poly = np.poly1d(syndrome3)
            syndrome4_poly = np.poly1d(syndrome4)

            # part2. building the elementary symmetric function
            u,p,lu0 = -1,-1,0

            d0 = np.poly1d([1])
            simga0 = np.poly1d([1])
            lu1 = 0
            
            
            
            print(elementarySymmetricFunc1)

            


            # elementarySymmetricFunc2 = syndrome1_poly * syndrome1_poly + syndrome2_poly
            # elementarySymmetricFunc2 = elementarySymmetricFunc2
            # elementarySymmetricFunc3 = syndrome3_poly + syndrome2_poly * elementarySymmetricFunc1 + elementarySymmetricFunc2 * syndrome1_poly
            # elementarySymmetricFunc3 = elementarySymmetricFunc3
            # elementarySymmetricFunc4 = (elementarySymmetricFunc1 ** 2) * syndrome2_poly

            # elementarySymmetricFunc = elementarySymmetricFunc1 + elementarySymmetricFunc2 * np.poly1d(np.array([1,0])) + elementarySymmetricFunc3 * np.poly1d(np.array([1,0,0])) + elementarySymmetricFunc4 * np.poly1d(np.array([1,0,0,0]))
            # elementarySymmetricFunc = poly_to_numpy_arr(elementarySymmetricFunc, 15) % 2
            # elementarySymmetricFunc = np.poly1d(elementarySymmetricFunc[::-1])

            # print("elementarySymmetricFunc1")
            # print(elementarySymmetricFunc1)
            # print("elementarySymmetricFunc2")
            # print(elementarySymmetricFunc2)
            # print("elementarySymmetricFunc3")
            # print(elementarySymmetricFunc3)
            # print("elementarySymmetricFunc4")
            # print(elementarySymmetricFunc4)

            # print(elementarySymmetricFunc)


        return codeword

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
        mess_to_LLR = np.clip(mess_to_LLR,-1,1)

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


    def encode_channel_with_QAM16(self):

        # generate message
        # msg = np.zeros((7,7))
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

        codeword = prod_codeword_col
        codeword_copy = np.copy(prod_codeword_col)
        codeword_copy = np.resize(codeword_copy,(15,15))

        # add three padding to cross the QAM-16
        codeword = np.append(codeword,0)
        codeword = np.append(codeword,0)
        codeword = np.append(codeword,0)

        # encoder
        encode_out_msg = np.array([])

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

        codeword_hard_decision = codeword_hard_decision[:224]
        codeword_hard_decision = np.resize(codeword_hard_decision,(15,15))

        return codeword_copy,recv_code_LLR,codeword_hard_decision


    # linear_BCH_code_process2 is built for the IBBR-CR
    def linear_BCH_code_process2(self):

        # deocding part
        codeword_copy,recv_code_LLR,codeword_hard_decision = self.encode_channel_with_QAM16()
        decoding_code = codeword_hard_decision.copy()
        recv_code_LLR_transpose = recv_code_LLR.copy().transpose()

        # 1. decode row first
        # 2. tranpose and keep the LLR and row
        #   2.5 change the n-th row word
        # 3. change to decode column
        #   3.5 change the n-th columns word
        # iteration and n + 1

        for n in range(self.iteration):
            
            # j is the j-th row word and j-columns word
            for j in range(15): 
                new_row_arr = np.array([])
                new_col_arr = np.array([])

                for k in range(15):
                    tmp_row_array = self.BCH_15_7_codeword_decoding(decoding_code[k])
                    new_row_arr = np.append(new_row_arr,tmp_row_array)
                    if recv_code_LLR[k][j] > 0:
                        recv_code_LLR[k][j] = 1
                    else:
                        recv_code_LLR[k][j] = -1
                    decoding_code[k][j] = recv_code_LLR[k][j] + funcLUC.LookUpTable_row(codeword_hard_decision[k][j], recv_code_LLR[k][j],14,3,decoding_code[k][j])
                decoding_code = decoding_code.transpose()

                for k in range(7):
                    tmp_col_array = self.BCH_15_7_codeword_decoding(decoding_code[k])
                    new_col_arr = np.append(new_col_arr,tmp_col_array)
                    if recv_code_LLR[k][j] > 0:
                        recv_code_LLR[k][j] = 1
                    else:
                        recv_code_LLR[k][j] = -1
                    decoding_code[k][j] = recv_code_LLR[k][j] + funcLUC.LookUpTable_col(codeword_hard_decision[k][j], recv_code_LLR[k][j],14,3,decoding_code[k][j])

                for k in range(8):
                    new_col_arr = np.append(new_col_arr,decoding_code[k+7],axis=0)

                decoding_code = decoding_code.transpose()

        # compare the original message and mark the result
        hamming_dist_msg = np.count_nonzero(decoding_code!=codeword_copy)
        block_err = (hamming_dist_msg != 0)

        return hamming_dist_msg,block_err

    def linear_BCH_code_process3(self):

        codeword_copy,recv_code_LLR,codeword_hard_decision = self.encode_channel_with_QAM16()
        # return codeword_hard_decision rece_codeword

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
            recv_code_LLR = np.add(recv_code_LLR,new_col_arr)
            recv_code_LLR = np.clip(recv_code_LLR,-1,1)

            # representation of Product Code
            decoding_code = recv_code_LLR.copy()
            decoding_code[decoding_code > 0] = 0
            decoding_code[decoding_code < 0] = 1


        recv_code_LLR[recv_code_LLR > 0] = 1
        recv_code_LLR[recv_code_LLR < 0] = 0

        hamming_dist_msg = np.count_nonzero(recv_code_LLR!=codeword_copy)
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
            elif BCH_type == 2:
                tmp_msg_count,block_count = self.linear_BCH_code_process3()

            hamming_dist_msg_count = hamming_dist_msg_count + tmp_msg_count
            probaility_block_error = probaility_block_error + block_count
            if tmp_msg_count > 0:
                print("message: ", i, " | hamming dist: ", tmp_msg_count, " | block err")
        count_time = time.time() - time_start
        if BCH_type == 0:
            prob_BER = hamming_dist_msg_count / (15 * self.message_sending_time)
        elif BCH_type == 1:
            prob_BER = hamming_dist_msg_count / (225 * self.message_sending_time)
        elif BCH_type == 2:
            prob_BER = hamming_dist_msg_count / (225 * self.message_sending_time)

        probaility_block_error = probaility_block_error / self.message_sending_time
        return count_time,prob_BER,probaility_block_error


def code_run_process(noise_set,message_sending_time,BCH_type,name,iteration=5):

    code = BCH_code(noise_set,message_sending_time,BCH_type,iteration)
    count_time,prob_BER,probaility_block_error = code.code_run_process_prepare()
    SNRcDB,SNRbDB = code.cal_SNR()

    print("name:",name)
    print("BLER",probaility_block_error)
    print("BER:",prob_BER)
    print("Count_time:",count_time)
    print("Message sending time:",message_sending_time)
    print("SNRcDB",SNRcDB)
    print("SNRbDB",SNRbDB)


    # ylen = 7
    # xlen = 15

    # result_list = [(name,xlen,ylen,message_sending_time,noise_set,prob_BER,SNRcDB,SNRbDB,probaility_block_error,0,0,0,0,count_time)]
    # column = ['LDPC_code','xlen','ylen','message_sending_time','noise_set','average_probaility_error','SNRcDB','SNRbDB','Prob_block_error','detected_block_error','detected_bit_error','undetected_block_error','undetected_bit_error','count_time']

    # df = pd.DataFrame(result_list, columns = column)

    # if os.path.exists('sim_result.csv') == False:
    #     df.to_csv('sim_result.csv', mode='a', header=True)
    # else:
    #     df.to_csv('sim_result.csv', mode='a', header=False)


if __name__ == "__main__":

    # test_val
    message_sending_time = 1
    # BCH_type -> 0 : BPSK 1 : iBBD-CR 2 : product code
    BCH_type = 0

    noise_set = 1
    # 0.49 -> 1 , 100
    # 0.0012 -> 2 , 100
    # 0.0009 -> 3 , 100
    # 0.00013 -> 5 , 100
    # 0.019 0.0002 -> 5 , 1000
    # 0.020 0.0002 -> 6 , 1000
    # 0.018 0.0002 -> 7 , 1000
    # 0.018 0.0003 -> 7 , 1000
    # 0.017 0.00025 -> 10, 1000
    iteration = 1
    name = "BCH_product_code"

    noise_set, message_sending_time = 0.6, 1
    code_run_process(noise_set,message_sending_time,BCH_type,name,iteration)

    # noise_set, message_sending_time = 0.8, 100
    # code_run_process(noise_set,message_sending_time,BCH_type,name,iteration,clip_num)

    # noise_set, message_sending_time = 0.75, 1000
    # code_run_process(noise_set,message_sending_time,BCH_type,name,iteration,clip_num)

    # noise_set, message_sending_time = 0.7, 1000
    # code_run_process(noise_set,message_sending_time,BCH_type,name,iteration,clip_num)

    # noise_set, message_sending_time = 0.6, 1000
    # code_run_process(noise_set,message_sending_time,BCH_type,name,iteration,clip_num)

    # noise_set, message_sending_time = 0.5, 10000
    # code_run_process(noise_set,message_sending_time,BCH_type,name,iteration,clip_num)

