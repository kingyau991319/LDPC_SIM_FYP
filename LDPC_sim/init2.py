import warnings
import time

import numpy as np
import pandas as pd
import os

import codegenerator
import decoding

import BCH_code as BCH
import hamming_code

class Init:

    def __init__(self,noise_set,mult_num,LDPC_code,iteration,clip_num,f_name="",LDPC_type=0,iter_decod=3,f_name2=""):
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
        self.f_name2 = f_name2
        self.BCH_code = BCH.BCH_code(self.noise_set,0,0,self.iteration,self.clip_num)
        self.hamming_code = hamming_code

        #generator matrix and parity-check matrix
        self.codegenerator = codegenerator.CodeGenerator()
        if self.LDPC_code == 0:
            self.parity_check_matrix,self.generator_matrix = self.codegenerator.inputMacKayCode(f_name)
            self.wr,self.wc,self.xlen,self.ylen = self.codegenerator.wr,self.codegenerator.wc,self.codegenerator.xlen,self.codegenerator.ylen

            # second LDPC matrix
            if self.LDPC_type == 3:
                self.parity_check_matrix2,self.generator_matrix2 = self.codegenerator.inputMacKayCode(f_name2)
                self.wr2,self.wc2,self.xlen2,self.ylen2 = self.codegenerator.wr,self.codegenerator.wc,self.codegenerator.xlen,self.codegenerator.ylen

        elif self.LDPC_code == 2:
            self.parity_check_matrix,self.generator_matrix = self.codegenerator.QCLDPCCode(self.len_mult_num)
            self.wr,self.wc,self.xlen,self.ylen = 4,3,14*self.len_mult_num,7*self.len_mult_num

        #decoding_tool
        self.decoding_tool = decoding.decoding_algorithm(self.parity_check_matrix,self.ylen,self.xlen,self.clip_num)

        if self.LDPC_type == 3:
            self.decoding_tool2 = decoding.decoding_algorithm(self.parity_check_matrix2,self.ylen2,self.xlen2,self.clip_num)

    def _message_to_LLR(self,rece_message):
        mess_to_LLR = (2 / (self.noise_set**2)) * rece_message
        mess_to_LLR = np.clip(mess_to_LLR,-self.clip_num,self.clip_num)
        return mess_to_LLR

    def signal_tranmission_repeat(self,message_sending_time):

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
                codeword_hamming_dist,no_bit_error_flag,no_bit_type_flag,codeword_hard_hamming_dist = self.messageSendingSimulation2()
                avg_prob_err_sum,bit_error_flag_sum,detected_block_err_num,detected_block_hamming_dist,undetected_block_err_num,undetected_block_hamming_dist = signalProcessingStatsMark(detected_block_hamming_dist,undetected_block_hamming_dist,detected_block_err_num,undetected_block_err_num,avg_prob_err_sum,codeword_hamming_dist,bit_error_flag_sum,no_bit_error_flag)
        elif self.LDPC_type == 1:
            for k in range(message_sending_time):
                codeword_hamming_dist,no_bit_error_flag,no_bit_type_flag,codeword_hard_hamming_dist = self.messageSendingSimulation1()
                avg_prob_err_sum,bit_error_flag_sum,detected_block_err_num,detected_block_hamming_dist,undetected_block_err_num,undetected_block_hamming_dist = signalProcessingStatsMark(detected_block_hamming_dist,undetected_block_hamming_dist,detected_block_err_num,undetected_block_err_num,avg_prob_err_sum,codeword_hamming_dist,bit_error_flag_sum,no_bit_error_flag)
        elif self.LDPC_type == 3:
            for k in range(message_sending_time):
                codeword_hamming_dist,no_bit_error_flag,no_bit_type_flag,codeword_hard_hamming_dist = self.messageSendingSimulation3()
                avg_prob_err_sum,bit_error_flag_sum,detected_block_err_num,detected_block_hamming_dist,undetected_block_err_num,undetected_block_hamming_dist = signalProcessingStatsMark(detected_block_hamming_dist,undetected_block_hamming_dist,detected_block_err_num,undetected_block_err_num,avg_prob_err_sum,codeword_hamming_dist,bit_error_flag_sum,no_bit_error_flag)
        elif self.LDPC_type == 4:
            for k in range(message_sending_time):
                codeword_hamming_dist,no_bit_error_flag,no_bit_type_flag,codeword_hard_hamming_dist = self.messageSendingSimulation4()
                avg_prob_err_sum,bit_error_flag_sum,detected_block_err_num,detected_block_hamming_dist,undetected_block_err_num,undetected_block_hamming_dist = signalProcessingStatsMark(detected_block_hamming_dist,undetected_block_hamming_dist,detected_block_err_num,undetected_block_err_num,avg_prob_err_sum,codeword_hamming_dist,bit_error_flag_sum,no_bit_error_flag)


        probaility_block_error = bit_error_flag_sum / message_sending_time
        prob_detected_block_error = detected_block_err_num / message_sending_time
        prob_undetected_block_error = undetected_block_err_num / message_sending_time
        
        prob_BER = avg_prob_err_sum / (message_sending_time * self.ylen)
        prob_detected_BER = detected_block_hamming_dist / (message_sending_time * self.ylen)
        prob_undetected_BER = undetected_block_hamming_dist / (message_sending_time * self.ylen)

        return prob_BER,probaility_block_error,prob_detected_block_error,prob_detected_BER,prob_undetected_block_error,prob_undetected_BER

    # 1. Product code A
    def messageSendingSimulation1(self):
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
    def messageSendingSimulation2(self):

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

    # product code C
    ##################################################################
    # for my own idea?
    # I want to build tow LDPC matrix and encode it together
    # such that I can take the different LDPC code advantages
    ### encoder
    # -> 1.encode row by fisrt-LDPC
    # -> 2.encode col by seoncd-LDPC
    ### channel
    # -> adding AWGN noise
    ### decoding
    # -> 1.decode by the row via SPA methods for first matrix
    # -> 2.decode by the col via SPA methods for seoncd matrix
    # -> 3.appending for part col to keep that data amount
    # ? options:
    # there are different variable change that I can choose
    # 1. only change the sinle entry of the codeowrds
    # 2. for all row or col
    

    # when should I start to do:
    # 8 th May, 2021, I want to show this idea at my ppt quicky
    # -> one day for building code, 3 days for running the code and testing the result
    

    # in: LDPC:1 LDPC:2
    # out: BER, BLER, undetected error flag | detected error flag
    def messageSendingSimulation3(self):

        # message generator
        xlen, ylen = self.xlen, self.ylen
        # msg = np.random.randint(2,size=(ylen,ylen))
        msg = np.zeros((ylen,ylen))

        msg_copy = msg.copy() # to backup and compare the result
        # msg_copy = msg_copy.transpose()

        # mult it and transpose and matmul one more time
        msg = np.matmul(msg,self.generator_matrix)
        msg = np.transpose(msg)
        msg = np.matmul(msg,self.generator_matrix2) % 2
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
                decoding_msg_row = np.append(decoding_msg_row,self.decoding_tool2.sumProductAlgorithmWithIterationForPC(msg_that_LLR[k],self.iteration),axis=0)
            decoding_msg_row = np.resize(decoding_msg_row,(xlen,xlen))
            decoding_msg_row = np.clip(decoding_msg_row,-self.clip_num,self.clip_num)

            # I use the row_result and do it again for the cols
            decoding_msg_row = decoding_msg_row.transpose()

            # That is for another LDPC code
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

            return hamming_dist_msg / self.ylen,1,is_detected_err,codeword_hard_hamming_dist/self.ylen

    # 4. Product code D with hamming code
    def messageSendingSimulation4(self):
        BCH_code = self.BCH_code
        xlen, ylen = self.xlen, self.ylen
        msg = np.random.randint(2, size=(4,ylen))
        msg = np.zeros((4,ylen))

        msg = np.transpose(msg)
        msg_copy = msg.copy()
        codeword = np.array([])

        # generate hamming code
        for k in np.arange(0,ylen):
            codeword = np.append(codeword,self.hamming_code.data_encode(msg[k]))
        codeword = np.resize(codeword,(ylen,7))
        codeword = np.transpose(codeword)

        # generate LDPC code
        codeword = np.matmul(codeword,self.generator_matrix) % 2

        codeword_copy = codeword.copy()
        codeword[codeword == 1] = -1
        codeword[codeword == 0] = 1

        # # noise with AWGN
        codeword = codeword + np.random.normal(0,self.noise_set,(7,xlen))

        # # hard-decision
        hard_decision_output = np.where(codeword > 0, 0, 1)
        msg_that_LLR = codeword.copy()
        msg_that_LLR = self._message_to_LLR(msg_that_LLR)

        for iteration in range(self.iter_decod):
            decoding_msg_row = np.array([])
            decoding_msg_col = np.array([])

            for k in range(7):
                decoding_msg_row = np.append(decoding_msg_row,self.decoding_tool.sumProductAlgorithmWithIterationForPC(msg_that_LLR[k],self.iteration),axis=0)
            decoding_msg_row = np.resize(decoding_msg_row,(7,xlen))
            decoding_msg_row = np.clip(decoding_msg_row,-self.clip_num,self.clip_num)

            decoding_msg_row_hard_dec = np.where(decoding_msg_row > 0, 0, 1)
            decoding_msg_row_hard_dec = decoding_msg_row_hard_dec.transpose()

            for k in np.arange(ylen):
                decoding_msg_col = np.append(decoding_msg_col,self.hamming_code.data_decode(decoding_msg_row_hard_dec[k]))

            for k in np.arange(ylen):
                decoding_msg_col = np.append(decoding_msg_col,decoding_msg_row_hard_dec[ylen+k],axis=0)


            decoding_msg_col = np.resize(decoding_msg_col,(xlen,7))

            decoding_msg_col = decoding_msg_col * self.clip_num
            decoding_msg_col = decoding_msg_col.transpose()
            msg_that_LLR = np.add(decoding_msg_col,decoding_msg_row)
            msg_that_LLR = np.clip(decoding_msg_col,-self.clip_num,self.clip_num)

        decoded_output = np.where(msg_that_LLR > 0,1,0)
        decoded_output = decoded_output.transpose()
        decoded_output_result = np.array([])


        for k in range(ylen):
            decoded_output_result = np.append(decoded_output_result ,self.hamming_code.outputmsg(decoded_output[k]))
        decoded_output_result = np.resize(decoded_output_result,(ylen,4))

        # decoded_output_result = decoded_output_result.astype(int)
        # print(msg_copy[0])
        # print(decoded_output_result[0])

        hamming_dist_msg = np.count_nonzero(msg != decoded_output_result)

        if hamming_dist_msg == 0:
            return 0,0,-1,0
        else:
            codeword_hard_hamming_dist = np.count_nonzero(codeword_copy[4, :ylen] != hard_decision_output[4, :ylen])
            is_detected_err = 1

            return hamming_dist_msg / 4,1,is_detected_err,codeword_hard_hamming_dist / 4

    def cal_SNR(self):

        if self.noise_set <= 0:
            return -1
        sigma =  2 * (self.noise_set ** 2)
        SNRc = 1 / sigma

        if self.LDPC_type == 1:
            SNRb = 15/7 * 2 * SNRc
        elif self.LDPC_type == 2:
            SNRb = 4 * SNRc
        elif self.LDPC_type == 3:
            coderate = (self.ylen * self.ylen2) / (self.xlen * self.xlen2)
            SNRb = (1/coderate) * SNRc
        elif self.LDPC_type == 4:
            coderate = (self.ylen * 4) / (self.xlen * 7)
            SNRb = (1/coderate) * SNRc

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

def LDPCCode_running(len_mult_num,noise_set,message_sending_time,iteration,name,LDPC_code,clip_num,filename,LDPC_type,iter_decod=5,filename2=""):

    time_start = time.time()
    init = Init(noise_set,len_mult_num,LDPC_code,iteration,clip_num,filename,LDPC_type,iter_decod,filename2)
    prob_BER,prob_block_right,prob_detected_block_error,prob_detected_BER,prob_undetected_block_error,prob_undetected_BER = init.signal_tranmission_repeat(message_sending_time)
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

if __name__ == "__main__":

    # runOnlyOneTimeLDPC()

    # 0 -> MacKayLDPC, 2 -> third_LDPC_code
    LDPC_code = 0
    # 1 -> Product A LDPC with BCH code, 2 -> Product B LDPC, 3 -> Product C LDPC with other LDPC code
    # 4 -< Product D LDPC with hamming code
    LDPC_type = 4

    len_mult_num = 4

    clip_num = 5
    iteration = 10
    iter_decod = 1
    # filename = "test_code_96.33.964"
    filename = "test_code_204.33.484"

    filename2 = ""
    # name = "TestCodeProductB"
    # filename = ""
    name = "LDPC_test"
    # name = "QAM16_96.33.964"
    # name = "Test_code_ProductCode"

    #28*28

    noise_set,message_sending_time = 1,10
    LDPCCode_running(len_mult_num,noise_set,message_sending_time,iteration,name,LDPC_code,clip_num,filename,LDPC_type,iter_decod)
    noise_set,message_sending_time = 0.9,10
    LDPCCode_running(len_mult_num,noise_set,message_sending_time,iteration,name,LDPC_code,clip_num,filename,LDPC_type,iter_decod)
    noise_set,message_sending_time = 0.8,100
    LDPCCode_running(len_mult_num,noise_set,message_sending_time,iteration,name,LDPC_code,clip_num,filename,LDPC_type,iter_decod)
    noise_set,message_sending_time = 0.7,1000
    LDPCCode_running(len_mult_num,noise_set,message_sending_time,iteration,name,LDPC_code,clip_num,filename,LDPC_type,iter_decod)
    noise_set,message_sending_time = 0.6,1000
    LDPCCode_running(len_mult_num,noise_set,message_sending_time,iteration,name,LDPC_code,clip_num,filename,LDPC_type,iter_decod)
    noise_set,message_sending_time = 0.55,1000
    LDPCCode_running(len_mult_num,noise_set,message_sending_time,iteration,name,LDPC_code,clip_num,filename,LDPC_type,iter_decod)
