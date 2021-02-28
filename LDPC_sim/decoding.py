import numpy as np
import warnings

class decoding_algorithm(object):

    def __init__(self,parity_check_matrix,parity_check_matrix_col,parity_check_matrix_len,clip_num):
        '''
        Decoding the LDPC code by different decoding methods

        Parameters
        ----------
        parity_check_matrix: the length of the identity matrix.
        p_col: The number of places by which elements are shifted.
        p_len: The length of .
        clip_num: to clip the bound of the LLR by SPA each time.

        LastUpdatedDate: 1-23-2021
        '''

        self.parity_check_matrix = parity_check_matrix
        self.p_col = parity_check_matrix_col
        self.p_len = parity_check_matrix_len
        self.t = int(parity_check_matrix_col / 2)
        self.clip_num = clip_num

    def sumProductAlgorithmWithIterationForPC(self,input_matrix,iteration):
        '''
        Using Sum-Product Algorithm(SPA) for decoding a LDPC code

        Parameters
        ----------
        input_matrix: the receiving code thorugh a noise channel that expressed as log-likelihood ratio
        iteration: time to do the iteration, if iteration is too large, that is low speed for the code.

        Returns
        -------
        hard_decision_output: the hard-decision output after decoding.
        final_part_summation: the LLR output after decoding.
        '''

        parity_matrix_mult_part = np.where(self.parity_check_matrix==0,0.0,input_matrix)
        parity_matrix_copy_add_part = np.copy(parity_matrix_mult_part)

        for iteration_time in range(iteration):
            mult_summation = np.where(parity_matrix_copy_add_part==0,1,np.tanh(parity_matrix_copy_add_part/2))
            mult_summation = np.prod(mult_summation,axis=1)

            for y_axis in range(self.p_col):
                parity_matrix_mult_part[y_axis] = np.where(parity_matrix_copy_add_part[y_axis] == 0,0,2 * np.arctanh(mult_summation[y_axis] / np.tanh(parity_matrix_copy_add_part[y_axis]/2)))
            add_summation = np.sum(parity_matrix_mult_part,axis=0)

            parity_matrix_copy_add_part = np.where(parity_matrix_copy_add_part == 0,0,input_matrix + add_summation - parity_matrix_mult_part)
            parity_matrix_copy_add_part = np.clip(parity_matrix_copy_add_part,-self.clip_num,self.clip_num)
            add_part_summation = np.sum(parity_matrix_mult_part,axis=0)

            final_part_summation = np.add(input_matrix,add_part_summation)
            hard_decision_output = np.where(final_part_summation > 0,0,1)

            check_matrix = np.matmul(self.parity_check_matrix,hard_decision_output) % 2

            if int(np.sum(check_matrix)) == 0:
                break

        return final_part_summation

    def sumProductAlgorithmWithIteration(self,input_matrix,iteration):

        '''
        Using Sum-Product Algorithm(SPA) for decoding a LDPC code

        Parameters
        ----------
        input_matrix: the receiving code thorugh a noise channel that expressed as log-likelihood ratio
        iteration: time to do the iteration, if iteration is too large, that is low speed for the code.

        Returns
        -------
        hard_decision_output: the hard-decision output after decoding.
        '''

        parity_matrix_mult_part = np.where(self.parity_check_matrix==0,0.0,input_matrix)
        parity_matrix_copy_add_part = np.copy(parity_matrix_mult_part)

        for iteration_time in range(iteration):
            mult_summation = np.where(parity_matrix_copy_add_part==0,1,np.tanh(parity_matrix_copy_add_part/2))
            mult_summation = np.prod(mult_summation,axis=1)

            for y_axis in range(self.p_col):
                parity_matrix_mult_part[y_axis] = np.where(parity_matrix_copy_add_part[y_axis] == 0,0,2 * np.arctanh(mult_summation[y_axis] / np.tanh(parity_matrix_copy_add_part[y_axis]/2)))
            add_summation = np.sum(parity_matrix_mult_part,axis=0)

            parity_matrix_copy_add_part = np.where(parity_matrix_copy_add_part == 0,0,input_matrix + add_summation - parity_matrix_mult_part)
            parity_matrix_copy_add_part = np.clip(parity_matrix_copy_add_part,-self.clip_num,self.clip_num)
            add_part_summation = np.sum(parity_matrix_mult_part,axis=0)

            final_part_summation = np.add(input_matrix,add_part_summation)
            hard_decision_output = np.where(final_part_summation > 0,0,1)

            check_matrix = np.matmul(self.parity_check_matrix,hard_decision_output) % 2

            if int(np.sum(check_matrix)) == 0:
                break

        return hard_decision_output

    def Gallager_A_alg(self,message):
        #TODO
        pass

    def hamming_distance(self,a,b):
        return np.count_nonzero(a!=b)

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    parity_matrix = np.array([[1.0,1.0,1.0,0.0,0.0],[0.0,1.0,0.0,1.0,1.0]])
    message_lam = np.array([1.5,0.1,-1,0.8,-1.2])
    decoding = decoding_algorithm(parity_matrix,2,5,25)
    # print(decoding.sumProductAlgorithmWithIteration(message_lam,3))
    print(decoding.sumProductAlgorithmWithIterationForPC(message_lam,3))
