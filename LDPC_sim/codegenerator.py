import numpy as np
import MacKayCode.mackay_code as mackay_code

class CodeGenerator:

    def __init__(self):
        '''
        Building the LDPC code with its parity-check matrix and generator matrix.

        Parameters
        ----------
        ylen,xlen: the length of cols, the length of rows.
        wr,wc: the weight of the row, the weight of the cols.

        Returns
        -------
        ylen,xlen: the length of cols, the length of rows.
        wr,wc: the weight of the row, the weight of the cols.

        LastUpdatedDate: 1-23-2021
        '''

        self.wc = 0
        self.wr = 0
        self.xlen = 0
        self.ylen = 0

    def _shiftIdentityMatrix(self,len,shift):

        '''
        Return the shift matrix by inputing the original matrix.

        Parameters
        ----------
        len: the length of the identity matrix.
        shift: The number of places by which elements are shifted.

        Returns
        -------
        output_matrix: the shifted matrix.
        '''

        output_matrix = np.identity(len)
        shift = int(shift)
        for x_axis in np.arange(len):
            output_matrix[x_axis] = np.roll(output_matrix[x_axis], shift)
        return output_matrix



    def _expendQCLDPCCode(self,input_matrix,identity_len,mode_flag = -1):
        '''
        Return the matrix that expanding by Shifted matrix.

        Parameters
        ----------
        len: the length of the identity matrix.
        shift: The number of places by which elements are shifted.
        mode_flag: the mode of the function.
            -1: shift_tmp = shift_tmp + 1
            1: shift_tmp = input_matrix[y_axis][x_axis]
            0: shift_tmp = (y_axis + x_axis) % identity_len

        Returns
        -------
        expanded_matrix: the expanding matrix.
        '''

        expanded_matrix = np.zeros((self.ylen * identity_len,self.xlen * identity_len))

        for y_axis in np.arange(self.ylen):
            shift_tmp = -1
            for x_axis in np.arange(self.xlen):  
                if input_matrix[y_axis][x_axis] >= 1:
                    if mode_flag == 0:
                        shift_tmp = (y_axis + x_axis) % identity_len
                    elif mode_flag == 1:
                        shift_tmp = input_matrix[y_axis][x_axis] - 1
                    else:
                        shift_tmp = shift_tmp + 1
                    tmp_matrix = self._shiftIdentityMatrix(identity_len,shift_tmp)
                    # shift_tmp = shift_tmp + 1
                    for y_axis2 in np.arange(identity_len):
                        for x_axis2 in np.arange(identity_len):
                            y_index = int(y_axis * identity_len + y_axis2)
                            x_index = int(x_axis * identity_len + x_axis2)
                            expanded_matrix[y_index][x_index] = int(tmp_matrix[y_axis2][x_axis2])

        # updated the new variable
        self.wr,self.wc = self.wr * identity_len,self.wc * identity_len
        self.xlen,self.ylen = self.xlen * identity_len,self.ylen * identity_len

        return expanded_matrix

    def inputMacKayCode(self,filename = "test_code_96.33.964"):
        '''
        Return the generator matrix and the parity-check matrix for the MacKayCode 

        Parameters
        ----------
        filename: the file of the MacKay format code.
                example: test_code_96.33.964

        Returns
        -------
        output_matrix: the parity-check_matrix for MacKay code.
        generator_matrix: the generator_matrix for MacKay code.
        '''

        self.ylen,self.xlen,self.wr,self.wc,parity_check_matrix,generator_matrix = mackay_code.inputMacKayCode(filename)
        return parity_check_matrix, generator_matrix

    def QCLDPCCode(self,mult_num):
        '''
        Return the generator matrix and the parity-check matrix for the QC_LDPC.

        Parameters
        ----------
        mult_num: the length of the identity matrix.

        Returns
        -------
        output_matrix: the parity-check_matrix for QC_LDPC code.
        generator_matrix: the generator_matrix for QC_LDPC code.
        '''

        input_matrix = np.zeros([7,14])

        for y_axis in np.arange(7):
            for x_axis in np.arange(14):
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

        output_matrix = self._expendQCLDPCCode(input_matrix,identity_len, 1)
        
        pre_generator_matrix = np.where(output_matrix==0,output_matrix,1)
        for y_axis in np.arange(7*mult_num - mult_num):
            pre_generator_matrix[y_axis + mult_num] = np.add(pre_generator_matrix[y_axis + mult_num], pre_generator_matrix[y_axis]) % 2

        pre_generator_matrix = pre_generator_matrix.transpose()

        generator_matrix = np.zeros((7*mult_num,14*mult_num))

        for y_axis in np.arange(7*mult_num):
            generator_matrix[y_axis][y_axis] = 1
            for x_axis in np.arange(7*mult_num):
                generator_matrix[y_axis][x_axis+7*mult_num] = pre_generator_matrix[y_axis][x_axis]

        self.xlen = 14 * identity_len
        self.ylen = 7 * identity_len

        return output_matrix,generator_matrix

if __name__ == "__main__":

# test_case0----------------------------------------------------------
    codeGenerator = CodeGenerator()
    generatorMat,parityMat = codeGenerator.QCLDPCCode(1)

    print("QC_LDPC_code")

    print("xlen,ylen")
    print(codeGenerator.xlen,codeGenerator.ylen)

    print("generator matrix for QC_LDPC")
    print(generatorMat)
    print("parity-check matrix for QC_LDPC")
    print(parityMat)
    print()
    
# test_case1----------------------------------------------------------
    generatorMat,parityMat = codeGenerator.inputMacKayCode()

    print("MacKayCode with 96.33.964(default)")

    print("xlen,ylen")
    print(codeGenerator.xlen,codeGenerator.ylen)

    print("generator matrix for MacKayCode")
    print(generatorMat)
    print("parity-check matrix for MacKayCode")
    print(parityMat)
    print()

# test_case2----------------------------------------------------------
    generatorMat,parityMat = codeGenerator.inputMacKayCode("test_code_204.33.484")

    print("MacKayCode with 204.33.484(not default)")

    print("xlen,ylen")
    print(codeGenerator.xlen,codeGenerator.ylen)

    print("generator matrix for MacKayCode")
    print(generatorMat)
    print("parity-check matrix for MacKayCode")
    print(parityMat)
    print()
