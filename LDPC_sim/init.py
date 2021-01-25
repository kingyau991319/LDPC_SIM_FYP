import warnings
import time

import numpy as np
import pandas as pd
import os

import codegenerator
import decoding

class Init:

    def __init__(self,noise_set,mult_num,LDPC_code,iteration,clip_num,f_name="",LDPC_type=0):

        self.noise_set = noise_set
        sigma =  2 * (self.noise_set ** 2)
        self.SNRc = 1 / sigma

        self.clip_num = clip_num
        self.iteration = iteration

        self.wr = 0
        self.wc = 0
        self.ylen = 0
        self.xlen = 0

        self.len_mult_num = mult_num
        self.LDPC_code = LDPC_code
        self.LDPC_type = LDPC_type
        self.f_name = f_name

        #generator matrix and parity-check matrix
        self.codegenerator = codegenerator.CodeGenerator()
        if self.LDPC_code == 0:
            self._primaryLDPC(self.f_name)
        elif self.LDPC_code == 2:
            self._thridLDPC()

        #decoding_tool
        self.decoding_tool = decoding.decoding_algorithm(self.parity_check_matrix,self.ylen,self.xlen,self.clip_num)

    def _primaryLDPC(self,f_name):
        self.parity_check_matrix,self.generator_matrix = self.codegenerator.inputMacKayCode(f_name)
        self.wr,self.wc,self.xlen,self.ylen = self.codegenerator.wr,self.codegenerator.wc,self.codegenerator.xlen,self.codegenerator.ylen

    def _thridLDPC(self):
        self.parity_check_matrix,self.generator_matrix = self.codegenerator.QCLDPCCode(self.len_mult_num)
        self.wr,self.wc,self.xlen,self.ylen = 4,3,14*self.len_mult_num,7*self.len_mult_num

    # for single case and print the result
    def signal_tranmission(self):
        self.signal_tranmission_repeat(1,1)

    def _printCodeSetMatrix(self):

            print("xlen",self.xlen)
            print("ylen",self.ylen)
            print("generator_matrix")
            print(self.generator_matrix)
            print("parity-check-matrix")
            print(self.parity_check_matrix)
            print("--------------------------------------------------------------------")
            print()

    def _printCodeInfo(self,generator_matrix,msg,msg_copy,final_msg,hamming_distance,msg_lam,ylen):

        print("random generated message")
        print(msg_copy)
        print("--------------------------------------------------------------------")
        print("receiving message")
        print(msg)
        print("message_lam")
        print(msg_lam)
        print("final message with checking")
        print(final_msg)
        print("final message")
        print(final_msg[:ylen])
        print("original message")
        message_copy = np.matmul(msg_copy,self.generator_matrix) % 2
        print(msg)
        print("hamming_distance")
        print(hamming_distance)
        print()

    def _message_to_LLR(self,rece_message):
        mess_to_LLR = (2 / (self.noise_set**2)) * rece_message
        mess_to_LLR = np.clip(mess_to_LLR,-self.clip_num,self.clip_num)
        return mess_to_LLR

    def signal_tranmission_repeat(self,message_sending_time,print_flag):

        def signalProcessingStatsMark(detected_block_hamming_dist,undetected_block_hamming_dist,detected_block_err_num,undetected_block_err_num,avg_prob_err_sum,codeword_hamming_dist,bit_error_flag_sum,no_bit_error_flag):
            avg_prob_err_sum = avg_prob_err_sum + codeword_hamming_dist
            bit_error_flag_sum = bit_error_flag_sum + no_bit_error_flag
            if no_bit_type_flag == 0:
                detected_block_err_num = detected_block_err_num + 1
                detected_block_hamming_dist = detected_block_hamming_dist + codeword_hamming_dist
                print("iteration:", k, "| hamming_distance:" , codeword_hamming_dist," | detected_error"," | codeword_hard_hamming_dist:",codeword_hard_hamming_dist)
            elif no_bit_type_flag == 1:
                undetected_block_err_num = undetected_block_err_num + 1
                undetected_block_hamming_dist = undetected_block_hamming_dist + codeword_hamming_dist
                print("iteration:", k, "| hamming_distance:" , codeword_hamming_dist," | undetected_error"," | codeword_hard_hamming_dist:",codeword_hard_hamming_dist)

            return avg_prob_err_sum,bit_error_flag_sum,detected_block_err_num,detected_block_hamming_dist,undetected_block_err_num,undetected_block_hamming_dist

        warnings.filterwarnings('ignore')

        #return result
        avg_prob_err_sum = 0
        bit_error_flag_sum = 0
        undetected_block_err_num, detected_block_err_num = 0, 0
        undetected_block_hamming_dist, detected_block_hamming_dist = 0, 0

        if print_flag == 1:
            self._printCodeSetMatrix()

        # 1. LDPC code
        # 2. Product code A
        # 3. product code B

        if self.LDPC_type == 0:
            for k in range(message_sending_time):
                codeword_hamming_dist,no_bit_error_flag,no_bit_type_flag,codeword_hard_hamming_dist = self.messageSendingSimulation(print_flag)
                avg_prob_err_sum,bit_error_flag_sum,detected_block_err_num,detected_block_hamming_dist,undetected_block_err_num,undetected_block_hamming_dist = signalProcessingStatsMark(detected_block_hamming_dist,undetected_block_hamming_dist,detected_block_err_num,undetected_block_err_num,avg_prob_err_sum,codeword_hamming_dist,bit_error_flag_sum,no_bit_error_flag)

        elif self.LDPC_type == 1:
            for k in range(message_sending_time):
                codeword_hamming_dist,no_bit_error_flag,no_bit_type_flag,codeword_hard_hamming_dist = self.messageSendingSimulation1(print_flag)
                avg_prob_err_sum,bit_error_flag_sum,detected_block_err_num,detected_block_hamming_dist,undetected_block_err_num,undetected_block_hamming_dist = signalProcessingStatsMark(detected_block_hamming_dist,undetected_block_hamming_dist,detected_block_err_num,undetected_block_err_num,avg_prob_err_sum,codeword_hamming_dist,bit_error_flag_sum,no_bit_error_flag)

        elif self.LDPC_type == 2:
            for k in range(message_sending_time):
                codeword_hamming_dist,no_bit_error_flag,no_bit_type_flag,codeword_hard_hamming_dist = self.messageSendingSimulation2(print_flag)
                avg_prob_err_sum,bit_error_flag_sum,detected_block_err_num,detected_block_hamming_dist,undetected_block_err_num,undetected_block_hamming_dist = signalProcessingStatsMark(detected_block_hamming_dist,undetected_block_hamming_dist,detected_block_err_num,undetected_block_err_num,avg_prob_err_sum,codeword_hamming_dist,bit_error_flag_sum,no_bit_error_flag)

        probaility_block_error = bit_error_flag_sum / message_sending_time
        prob_detected_block_error = detected_block_err_num / message_sending_time
        prob_undetected_block_error = undetected_block_err_num / message_sending_time
        
        prob_BER = avg_prob_err_sum / (message_sending_time * self.ylen)
        prob_detected_BER = detected_block_hamming_dist / (message_sending_time * self.ylen)
        prob_undetected_BER = undetected_block_hamming_dist / (message_sending_time * self.ylen)

        return prob_BER,probaility_block_error,prob_detected_block_error,prob_detected_BER,prob_undetected_block_error,prob_undetected_BER

    '''
        for running product code
        1.message -> the passing message
        2.message_copy -> for compare the final_message and output the hamming distance
        3.out_message -> hard_decision work
    '''


        # if print_flag == 1:
        #     print("out_message_hard_decision_row")
        #     print(out_msg_hard_decision_row)
        #     print("out_message_hard_decision_col")
        #     print(out_msg_hard_decision_col)
        #     print("decoding_message_row")
        #     print(decoding_msg_row)
        #     print("decoding_message_col")
        #     print(decoding_msg_col)
        #     print("hamming_distance_row_for_hard_decision_and_after_SPADecoding")
        #     hamming_dist_hard_row_sum = int(hamming_dist_row.sum())
        #     print(hamming_dist_row,hamming_dist_hard_row_sum)
        #     print("hamming_distance_col_for_hard_decision_and_after_SPADecoding")
        #     hamming_dist_hard_col_sum = int(hamming_dist_col.sum())
        #     print(hamming_dist_col,hamming_dist_hard_col_sum)
        #     print("t")
        #     print(ylen/2)
        #     print("message_copy")
        #     print(msg_copy)
        #     print("row_message")
        #     print(row_msg)
        #     print("hamming_distance_row_message")
        #     print(hamming_dist_row_msg)
        #     print("col_message")
        #     print(col_msg)
        #     print("hamming_distance_col_message")
        #     print(hamming_dist_col_msg)

    def messageSendingSimulation(self,print_flag):

        xlen, ylen = self.xlen, self.ylen
        parity_check_matrix, generator_matrix = self.parity_check_matrix, self.generator_matrix

        # message generator
        message = np.random.randint(2, size = ylen)
        # message = np.zeros(ylen)
        message_copy = np.copy(message)

        # building_sending_message
        codeword = np.matmul(message,self.generator_matrix) % 2

        codeword[codeword == 1] = -1
        codeword[codeword == 0] = 1

        rece_codeword = np.add(codeword,np.random.normal(0,self.noise_set,self.xlen))

        # for hard decision check to avoid the use of SPA
        codeword_hard_decision = rece_codeword.copy()
        codeword_hard_decision[codeword_hard_decision > 0] = 0
        codeword_hard_decision[codeword_hard_decision < 0] = 1

        hard_decision_check = np.matmul(parity_check_matrix,codeword_hard_decision) % 2
        hard_decision_mult_sum = np.sum(hard_decision_check)
        if (hard_decision_mult_sum == 0):
            return 0,0,-1,0

        message_LLR = self._message_to_LLR(rece_codeword)
        out_message = self.decoding_tool.sumProductAlgorithmWithIteration(message_LLR,self.iteration)
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
            self._printCodeInfo(self.generator_matrix,message,message_copy,out_message,message_hamming_dist,message_LLR,ylen)

        return message_hamming_dist,block_error_flag,block_error_type_flag,codeword_hard_hamming_dist

    def messageSendingSimulation1(self,print_flag):

        # message generator
        xlen, ylen = self.xlen, self.ylen
        msg = np.random.randint(2,size=(ylen,ylen))

        msg_copy = msg.copy() # to backup and compare the result

        # mult it and transpose and matmul one more time
        msg = np.matmul(msg,self.generator_matrix)
        msg = np.transpose(msg)
        msg = np.matmul(msg,self.generator_matrix) % 2
        msg[msg == 1] = -1
        msg[msg == 0] = 1

        # noise with AWGN
        msg = msg + np.random.normal(0,self.noise_set,(xlen,xlen))

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

        # hard-decision
        hard_decision_output = np.where(msg > 0, 0, 1)

        # # I need to do the SPA for both
        msg_that_LLR = msg.copy()
        msg_that_LLR = self._message_to_LLR(msg_that_LLR)

        msg_that_LLR_row = msg_that_LLR[:ylen,:xlen]
        decoding_msg_row = np.array([])
        for k in range(ylen):
            decoding_msg_row = np.append(decoding_msg_row,self.decoding_tool.sumProductAlgorithmWithIterationForPC(msg_that_LLR_row[k],self.iteration)[1],axis=0)
        decoding_msg_row = np.resize(decoding_msg_row,(ylen,xlen))
        decoding_msg_row = np.clip(decoding_msg_row,-self.clip_num,self.clip_num)

        ### I use the row_result and do it again for the cols

        msg_that_LLR_col = msg_that_LLR.copy()
        msg_that_LLR_col = msg_that_LLR_col[:xlen,:ylen].transpose()

        decoding_msg_row = decoding_msg_row[:ylen,:ylen].transpose()

        for y in np.arange(self.ylen):
            for x in np.arange(self.ylen):
                msg_that_LLR_col[y][x] = decoding_msg_row[y][x]

        decoding_msg_col = np.array([])
        for k in np.arange(ylen):
            decoding_msg_col = np.append(decoding_msg_col,self.decoding_tool.sumProductAlgorithmWithIterationForPC(msg_that_LLR_col[k],self.iteration)[1],axis=0)
        decoding_msg_col = np.resize(decoding_msg_col,(ylen,xlen))

        output_msg = decoding_msg_col[:ylen,:ylen]
        decoded_output = np.where(decoding_msg_col > 0,0,1)

        hamming_dist_msg = np.count_nonzero(msg_copy != decoded_output[:ylen,:ylen])
        hamming_dist_msg = hamming_dist_msg / self.ylen

        if hamming_dist_msg == 0:
            return 0,0,-1,0
        else:
            codeword_hard_hamming_dist = np.count_nonzero(hard_decision_output[:ylen,:ylen]!=decoded_output[:ylen,:ylen])
            hard_decision_check = np.matmul(self.parity_check_matrix,decoded_output.transpose()) % 2
            isDetectedError = np.sum(hard_decision_check)
            if isDetectedError == 0:
                isDetectedError = 1
            else:
                isDetectedError = 0

                #TODO blcok_error

            return hamming_dist_msg,1,isDetectedError,codeword_hard_hamming_dist/self.ylen


    def messageSendingSimulation2(self,print_flag):

        # message generator
        xlen, ylen = self.xlen, self.ylen
        msg = np.random.randint(2,size=(ylen,ylen))

        msg_copy = msg.copy() # to backup and compare the result

        # mult it and transpose and matmul one more time
        msg = np.matmul(msg,self.generator_matrix)
        msg = np.transpose(msg)
        msg = np.matmul(msg,self.generator_matrix) % 2
        msg[msg == 1] = -1
        msg[msg == 0] = 1

        # noise with AWGN
        msg = msg + np.random.normal(0,self.noise_set,(xlen,xlen))

        # hard-decision
        hard_decision_output = np.where(msg > 0, 0, 1)

        # # I need to do the SPA for both
        msg_that_LLR = msg.copy()
        msg_that_LLR = self._message_to_LLR(msg_that_LLR)

        decoding_msg_row = np.array([])

        for k in range(xlen):
            decoding_msg_row = np.append(decoding_msg_row,self.decoding_tool.sumProductAlgorithmWithIterationForPC(msg_that_LLR[k],self.iteration)[1],axis=0)
        decoding_msg_row = np.resize(decoding_msg_row,(xlen,xlen))
        decoding_msg_row = np.clip(decoding_msg_row,-self.clip_num,self.clip_num)

        # I use the row_result and do it again for the cols
        decoding_msg_row = decoding_msg_row[:xlen,:ylen].transpose()
        decoding_msg_col = np.array([])

        for k in np.arange(ylen):
            decoding_msg_col = np.append(decoding_msg_col,self.decoding_tool.sumProductAlgorithmWithIterationForPC(decoding_msg_row[k],self.iteration)[1],axis=0)
        decoding_msg_col = np.resize(decoding_msg_col,(ylen,xlen))

        output_msg = decoding_msg_col[:ylen,:ylen]
        decoded_output = np.where(decoding_msg_col > 0,0,1)

        hamming_dist_msg = np.count_nonzero(msg_copy != decoded_output[:ylen,:ylen])

        if hamming_dist_msg == 0:
            return 0,0,-1,0
        else:
            codeword_hard_hamming_dist = np.count_nonzero(hard_decision_output[:ylen,:ylen]!=decoded_output[:ylen,:ylen])
            hard_decision_check = np.matmul(self.parity_check_matrix,decoded_output.transpose()) % 2
            is_detected_rrr = np.sum(hard_decision_check)
            if is_detected_rrr == 0:
                is_detected_rrr = 1
            else:
                is_detected_rrr = 0

                #TODO blcok_error

            return hamming_dist_msg / self.ylen,1,is_detected_rrr,codeword_hard_hamming_dist/self.ylen

    def cal_SNR(self):

        if self.noise_set <= 0:
            return -1
        sigma =  2 * (self.noise_set ** 2)
        SNRc = 1 / sigma

        if self.LDPC_type == 0:
            SNRb = 2 * SNRc
        elif self.LDPC_type == 1:
            SNRb = 3 * SNRc
        elif self.LDPC_type == 2:
            SNRb = 4 * SNRc

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

def LDPCCode_running(len_mult_num,noise_set,message_sending_time,iteration,name,LDPC_code,clip_num,filename,LDPC_type=1):

    time_start = time.time()
    wr = 10
    wc = 5

    init = Init(noise_set,len_mult_num,LDPC_code,iteration,clip_num,filename,LDPC_type)
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
    ylen = int(init.ylen)
    xlen = int(init.xlen)

    result_list = [(name,xlen,ylen,message_sending_time,noise_set,prob_BER,SNRcDB,SNRbDB,probaility_block_error,prob_detected_block_error,prob_detected_BER,prob_undetected_block_error,prob_undetected_BER,count_time)]
    column = ['LDPC_code','xlen','ylen','message_sending_time','noise_set','average_probaility_error','SNRcDB','SNRbDB','Prob_block_error','detected_block_error','detected_bit_error','undetected_block_error','undetected_bit_error','count_time']

    df = pd.DataFrame(result_list, columns = column)

    if os.path.exists('sim_result.csv') == False:
        df.to_csv('sim_result.csv', mode='a', header=True)
    else:
        df.to_csv('sim_result.csv', mode='a', header=False)

def runOnlyOneTimeLDPC():

    len_mult_num = 1

    noise_set = 0.1
    print_flag = 0
    iteration = 10
    clip_num = 7
    LDPC_code = 2
    LDPC_type = 2

    init = Init(noise_set,len_mult_num,LDPC_code,iteration,7,"",LDPC_type)

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

    # 0 -> primary_LDPC_code, 1 -> second_LDPC_code, 2 -> third_LDPC_code
    LDPC_code = 0
    # 0 -> linear LDPC, 1 -> Product A LDPC, 2 -> Product B LDPC
    LDPC_type = 2

    len_mult_num = 1

    clip_num = 8
    iteration = 20
    name = "MacKeyCode_96.33.964"
    name = "Product_codeB_96.33.964"
    filename = "test_code_96.33.964"

    noise_set,message_sending_time = 0.85,100
    LDPCCode_running(len_mult_num,noise_set,message_sending_time,iteration,name,LDPC_code,clip_num,filename,LDPC_type)

    noise_set,message_sending_time = 0.8,100
    LDPCCode_running(len_mult_num,noise_set,message_sending_time,iteration,name,LDPC_code,clip_num,filename,LDPC_type)