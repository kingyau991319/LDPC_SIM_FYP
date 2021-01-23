import numpy as np

'''
    for generate the generator matrix and parity-check matrix 
    by inputting the file format with MacKay LDPC code in this website:  
    https://www.inference.org.uk/mackay/codes/data.html 
'''

def _gf2elim(M):

    '''
        ref by https://gist.github.com/popcornell/bc29d1b7ba37d824335ab7b6280f7fec 
        Return: generator_matrix(m) 
        Use to do the Guassian elimination for GF(2) 
    '''
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

def inputMacKayCode(f_name = "test_code_96.33.964"):
    '''
    This function is for input the MacKayCode and output the detail of the document. 

    Parameters
    ----------
    f_name: filename with the LDPC code from the file that store the MacKayCode.
            default -> 96.33.964

    Returns
    ----------
    ylen,xlen: the length of cols, the length of rows
    wr,wc: the weight of the row, the weight of the cols
    parity_check_matrix
    generator_matrix
    '''

    f_name = "./MacKayCode/" + f_name
    f = open(f_name,"r")

    str = f.read()
    str = str.split()
    str = np.array(str)
    str_len = len(str)
    input_num = str.astype(np.int)

    xlen = input_num[0]
    ylen = input_num[1]
    wc = input_num[2]
    wr = input_num[3]
    wc_num = input_num[xlen+4:xlen+ylen+4]

    lower_bound_of_array = str_len - np.sum(wc_num)

    input_num = input_num[lower_bound_of_array:]
    parity_check_matrix = np.zeros((ylen,xlen))
    x_index_index = 0

    for y_axis in np.arange(ylen):
        for x_axis in np.arange(wc_num[y_axis]):
            x_index = input_num[x_index_index] - 1
            parity_check_matrix[y_axis][x_index] = 1
            x_index_index = x_index_index + 1

    parity_check_matrix = parity_check_matrix.astype(np.uint8)
    parity_check_matrix_copy = np.zeros((ylen,xlen))
    generator_matrix = np.zeros((ylen,xlen))

    '''
    Step:
    1. change the parity check matrix
    2. apply gf2elim
    3. transpose and make generator matrix
    '''

    for y_axis in np.arange(ylen):
        for x_axis in np.arange(ylen):
            parity_check_matrix_copy[y_axis][x_axis] = int(parity_check_matrix[y_axis][x_axis+ylen])
            parity_check_matrix_copy[y_axis][x_axis+ylen] = int(parity_check_matrix[y_axis][x_axis])

    parity_check_matrix_copy = parity_check_matrix_copy.astype(np.uint8)

    M = _gf2elim(parity_check_matrix_copy)

    M2 = np.zeros((ylen,ylen))

    for y_axis in np.arange(ylen):
        for x_axis in np.arange(ylen):
            M2[y_axis][x_axis] = M[y_axis][x_axis+ylen]

    M2 = np.transpose(M2)

    for y_axis in np.arange(ylen):
        generator_matrix[y_axis][y_axis] = 1
        for x_axis in np.arange(ylen):
            generator_matrix[y_axis][x_axis+ylen] = M2[y_axis][x_axis]

    return ylen,xlen,wr,wc,parity_check_matrix,generator_matrix