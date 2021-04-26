import warnings
import time

import numpy as np
import pandas as pd
import os

import codegenerator
import decoding

import BCH_code as BCH

class Init:

    def __init__(self,noise_set,mult_num,LDPC_code,iteration,clip_num,f_name="",LDPC_type=0,iter_decod=3):

        self.noise_set = noise_set
        sigma =  2 * (self.noise_set ** 2)
        self.SNRc = 1 / sigma

        self.clip_num = clip_num
        self.iteration = iteration

        self.iter_decod = iter_decod

        self.wr = 0
        self.wc = 0
        self.ylen = 0
        self.xlen = 0

        self.len_mult_num = mult_num
        self.LDPC_code = LDPC_code
        self.LDPC_type = LDPC_type
        self.f_name = f_name
        self.BCH_code = BCH.BCH_code(self.noise_set,0,0,self.iteration,self.clip_num)

        #generator matrix and parity-check matrix
        self.codegenerator = codegenerator.CodeGenerator()
        if self.LDPC_code == 0:
            self.parity_check_matrix,self.generator_matrix = self.codegenerator.inputMacKayCode(f_name)
            self.wr,self.wc,self.xlen,self.ylen = self.codegenerator.wr,self.codegenerator.wc,self.codegenerator.xlen,self.codegenerator.ylen
        elif self.LDPC_code == 2:
            self.parity_check_matrix,self.generator_matrix = self.codegenerator.QCLDPCCode(self.len_mult_num)
            self.wr,self.wc,self.xlen,self.ylen = 4,3,14*self.len_mult_num,7*self.len_mult_num

        #decoding_tool
        self.decoding_tool = decoding.decoding_algorithm(self.parity_check_matrix,self.ylen,self.xlen,self.clip_num)

    def _message_to_LLR(self,rece_message):
        mess_to_LLR = (2 / (self.noise_set**2)) * rece_message
        mess_to_LLR = np.clip(mess_to_LLR,-self.clip_num,self.clip_num)
        return mess_to_LLR

    def signal_tranmission_repeat(self,message_sending_time,print_flag):

        def signalProcessingStatsMark(detected_block_hamming_dist,undetected_block_hamming_dist,detected_block_err_num,undetected_block_err_num,avg_prob_err_sum,codeword_hamming_dist,bit_error_flag_sum,no_bit_error_flag):
            avg_prob_err_sum = avg_prob_err_sum + codeword_hamming_dist
            bit_error_flag_sum = bit_error_flag_sum + no_bit_error_flag
            detected_block_err_num = detected_block_err_num + 1
            detected_block_hamming_dist = detected_block_hamming_dist + codeword_hamming_dist
            print("iteration:", k, "| hamming_distance:" , codeword_hamming_dist," | detected_error"," | codeword_hard_hamming_dist:",codeword_hard_hamming_dist)

            return avg_prob_err_sum,bit_error_flag_sum,detected_block_err_num,detected_block_hamming_dist,undetected_block_err_num,undetected_block_hamming_dist

        warnings.filterwarnings('ignore')

        #return result
        avg_prob_err_sum = 0
        bit_error_flag_sum = 0
        undetected_block_err_num, detected_block_err_num = 0, 0
        undetected_block_hamming_dist, detected_block_hamming_dist = 0, 0

        if self.LDPC_type == 2:
            for k in range(message_sending_time):
                codeword_hamming_dist,no_bit_error_flag,no_bit_type_flag,codeword_hard_hamming_dist = self.messageSendingSimulation2(print_flag)
                avg_prob_err_sum,bit_error_flag_sum,detected_block_err_num,detected_block_hamming_dist,undetected_block_err_num,undetected_block_hamming_dist = signalProcessingStatsMark(detected_block_hamming_dist,undetected_block_hamming_dist,detected_block_err_num,undetected_block_err_num,avg_prob_err_sum,codeword_hamming_dist,bit_error_flag_sum,no_bit_error_flag)
        else:
            for k in range(message_sending_time):
                codeword_hamming_dist,no_bit_error_flag,no_bit_type_flag,codeword_hard_hamming_dist = self.messageSendingSimulation1(print_flag)
                avg_prob_err_sum,bit_error_flag_sum,detected_block_err_num,detected_block_hamming_dist,undetected_block_err_num,undetected_block_hamming_dist = signalProcessingStatsMark(detected_block_hamming_dist,undetected_block_hamming_dist,detected_block_err_num,undetected_block_err_num,avg_prob_err_sum,codeword_hamming_dist,bit_error_flag_sum,no_bit_error_flag)


        probaility_block_error = bit_error_flag_sum / message_sending_time
        prob_detected_block_error = detected_block_err_num / message_sending_time
        prob_undetected_block_error = undetected_block_err_num / message_sending_time
        
        prob_BER = avg_prob_err_sum / (message_sending_time * self.ylen)
        prob_detected_BER = detected_block_hamming_dist / (message_sending_time * self.ylen)
        prob_undetected_BER = undetected_block_hamming_dist / (message_sending_time * self.ylen)

        return prob_BER,probaility_block_error,prob_detected_block_error,prob_detected_BER,prob_undetected_block_error,prob_undetected_BER

    # 1. Product code A
    def messageSendingSimulation1(self,print_flag):
        BCH_code = self.BCH_code
        xlen, ylen = self.xlen, self.ylen
        msg = np.random.randint(2, size=(7,ylen))

        msg_copy = msg.copy() # to backup and compare the result
        msg = np.transpose(msg)
        codeword = np.array([])
        for k in range(27):
            codeword = np.append(codeword,BCH_code.BCH_15_7_codeword_generator(msg[k]))
        codeword = np.resize(codeword,(ylen,15))
        # print("codeword")
        # print(codeword[0])
        # print(BCH_code.BCH_15_7_codeword_decoding(codeword[0]))

        codeword = np.transpose(codeword)
        codeword = np.matmul(codeword,self.generator_matrix) % 2
        codeword_copy = codeword.copy()
        codeword[codeword == 1] = -1
        codeword[codeword == 0] = 1

        # noise with AWGN
        codeword = codeword + np.random.normal(0,self.noise_set,(15,xlen))

        # hard-decision
        hard_decision_output = np.where(codeword > 0, 0, 1)
        msg_that_LLR = codeword.copy()
        msg_that_LLR = self._message_to_LLR(msg_that_LLR)

        for iteration in range(self.iter_decod):
            decoding_msg_row = np.array([])
            decoding_msg_col = np.array([])

            for k in range(15):
                decoding_msg_row = np.append(decoding_msg_row,self.decoding_tool.sumProductAlgorithmWithIterationForPC(msg_that_LLR[k],self.iteration),axis=0)
            decoding_msg_row = np.resize(decoding_msg_row,(15,xlen))
            decoding_msg_row = np.clip(decoding_msg_row,-self.clip_num,self.clip_num)

            decoding_msg_row_hard_dec = np.where(decoding_msg_row > 0, 0, 1)

            decoding_msg_row_hard_dec = decoding_msg_row_hard_dec.transpose()

            for k in np.arange(ylen):
                decoding_msg_col = np.append(decoding_msg_col,BCH_code.BCH_15_7_codeword_decoding(decoding_msg_row_hard_dec[k]))

            for k in np.arange(ylen):
                decoding_msg_col = np.append(decoding_msg_col,decoding_msg_row_hard_dec[ylen+k],axis=0)

            decoding_msg_col = np.resize(decoding_msg_col,(xlen,15))

            decoding_msg_col = decoding_msg_col * (self.clip_num / 2)
            decoding_msg_col = decoding_msg_col.transpose()
            msg_that_LLR = np.add(decoding_msg_col,decoding_msg_row)
            msg_that_LLR = np.clip(decoding_msg_col,-self.clip_num,self.clip_num)

        decoded_output = np.where(msg_that_LLR > 0,0,1)

        hamming_dist_msg = np.count_nonzero(codeword_copy[7,:ylen] != decoded_output[7,:ylen])

        if hamming_dist_msg == 0:
            return 0,0,-1,0
        else:
            codeword_hard_hamming_dist = np.count_nonzero(codeword_copy[7, :ylen] != hard_decision_output[7, :ylen])
            is_detected_err = 1

            return hamming_dist_msg / 7,1,is_detected_err,codeword_hard_hamming_dist / 7


    # 2 -> Product Code B
    def messageSendingSimulation2(self,print_flag):

        # message generator
        xlen, ylen = self.xlen, self.ylen
        msg = np.random.randint(2,size=(ylen,ylen))

        msg_copy = msg.copy() # to backup and compare the result
        # msg_copy = msg_copy.transpose()

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


        # for iteration in range(self.iter_decod):
        for iteration in range(self.iter_decod):

            decoding_msg_row = np.array([])
            decoding_msg_col = np.array([])

            for k in range(xlen):
                decoding_msg_row = np.append(decoding_msg_row,self.decoding_tool.sumProductAlgorithmWithIterationForPC(msg_that_LLR[k],self.iteration),axis=0)
            decoding_msg_row = np.resize(decoding_msg_row,(xlen,xlen))
            decoding_msg_row = np.clip(decoding_msg_row,-self.clip_num,self.clip_num)

            # I use the row_result and do it again for the cols
            decoding_msg_row = decoding_msg_row.transpose()

            for k in np.arange(ylen):
                decoding_msg_col = np.append(decoding_msg_col,self.decoding_tool.sumProductAlgorithmWithIterationForPC(decoding_msg_row[k],self.iteration),axis=0)
            for k in np.arange(ylen):
                decoding_msg_col = np.append(decoding_msg_col,decoding_msg_row[ylen+k],axis=0)

            decoding_msg_col = np.resize(decoding_msg_col,(xlen,xlen))
            decoding_msg_col = np.clip(decoding_msg_col,-self.clip_num,self.clip_num)

            decoding_msg_col = decoding_msg_col.transpose()
            msg_that_LLR = decoding_msg_col

        msg_that_LLR = msg_that_LLR.transpose()

        decoded_output = np.where(msg_that_LLR > 0,0,1)

        hamming_dist_msg = np.count_nonzero(msg_copy != decoded_output[:ylen,:ylen])

        if hamming_dist_msg == 0:
            return 0,0,-1,0
        else:
            codeword_hard_hamming_dist = np.count_nonzero(hard_decision_output[:ylen,:ylen]!=msg_copy[:ylen,:ylen])
            hard_decision_check = np.matmul(self.parity_check_matrix,decoded_output.transpose()) % 2
            is_detected_err = np.sum(hard_decision_check)
            if is_detected_err == 0:
                is_detected_err = 1
            else:
                is_detected_err = 0

                #TODO blcok_error

            return hamming_dist_msg / self.ylen,1,is_detected_err,codeword_hard_hamming_dist/self.ylen

    def cal_SNR(self):

        if self.noise_set <= 0:
            return -1
        sigma =  2 * (self.noise_set ** 2)
        SNRc = 1 / sigma

        if self.LDPC_type == 1:
            SNRb = 15/7 * 2 * SNRc
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

def LDPCCode_running(len_mult_num,noise_set,message_sending_time,iteration,name,LDPC_code,clip_num,filename,LDPC_type):

    time_start = time.time()

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

    # df = pd.DataFrame(result_list, columns = column)

    # if os.path.exists('sim_result.csv') == False:
    #     df.to_csv('sim_result.csv', mode='a', header=True)
    # else:
    #     df.to_csv('sim_result.csv', mode='a', header=False)

if __name__ == "__main__":

    # runOnlyOneTimeLDPC()

    # 0 -> MacKayLDPC, 2 -> third_LDPC_code
    LDPC_code = 0
    # 1 -> Product A LDPC, 2 -> Product B LDPC
    LDPC_type = 1

    len_mult_num = 4

    clip_num = 5
    iteration = 10
    filename = "test_code_96.33.964"
    # name = "TestCodeProductB"
    # filename = ""
    name = "MacKeyCode_96.33.964"
    # name = "QAM16_96.33.964"
    # name = "Test_code_ProductCode"

    #28*28

    noise_set,message_sending_time = 0.01,100
    LDPCCode_running(len_mult_num,noise_set,message_sending_time,iteration,name,LDPC_code,clip_num,filename,LDPC_type)
    # noise_set,message_sending_time = 0.01,1
    # LDPCCode_running(len_mult_num,noise_set,message_sending_time,iteration,name,LDPC_code,clip_num,filename,LDPC_type)
    # noise_set,message_sending_time = 0.01,1
    # LDPCCode_running(len_mult_num,noise_set,message_sending_time,iteration,name,LDPC_code,clip_num,filename,LDPC_type)
    # noise_set,message_sending_time = 0.01,1
    # LDPCCode_running(len_mult_num,noise_set,message_sending_time,iteration,name,LDPC_code,clip_num,filename,LDPC_type)
