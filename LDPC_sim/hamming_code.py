import numpy as np


generator_matrix = np.array([[1,1,1,0,0,0,0],[1,0,0,1,1,0,0],[0,1,0,1,0,1,0],[1,1,0,1,0,0,1]])
parity_check_matrix = np.array([[1,0,1,0,1,0,1],[0,1,1,0,0,1,1],[0,0,0,1,1,1,1]])
parity_check_matrix = parity_check_matrix.transpose()

# generator_matrix = np.array([[1,0,0,0,1,1,1,0,0,0,0,1,1,1,1],[0,1,0,0,1,0,0,1,1,0,1,0,1,1,1],[0,0,1,0,0,1,0,1,0,1,1,1,0,1,1],[0,0,0,1,0,0,1,0,1,1,1,1,1,0,1]])


def data_encode(msg):
    msg = np.matmul(msg,generator_matrix) % 2
    return msg

def data_decode_only_msg(codeword):
    error_check = np.matmul(codeword,parity_check_matrix) % 2
    error_check = error_check
    if np.sum(codeword) != 0:
        if (error_check[0] == 0) and ((error_check[1] == 0) and (error_check[2] == 1)):
            codeword = np.add(codeword,np.array([1,0,0,0,0,0,0])) % 2
        elif (error_check[1] == 0) and ((error_check[1] == 1) and (error_check[2] == 0)):
            codeword = np.add(codeword,np.array([0,1,0,0,0,0,0])) % 2
        elif (error_check[1] == 0) and ((error_check[1] == 1) and (error_check[2] == 1)):
            codeword = np.add(codeword,np.array([0,0,1,0,0,0,0])) % 2
        elif (error_check[0] == 1) and ((error_check[1] == 0) and (error_check[2] == 0)):
            codeword = np.add(codeword,np.array([0,0,0,1,0,0,0])) % 2
        elif (error_check[0] == 1) and ((error_check[1] == 0) and (error_check[2] == 1)):
            codeword = np.add(codeword,np.array([0,0,0,0,1,0,0])) % 2
        elif (error_check[0] == 1) and ((error_check[1] == 1) and (error_check[2] == 0)):
            codeword = np.add(codeword,np.array([0,0,0,0,0,1,0])) % 2
        elif (error_check[0] == 1) and ((error_check[1] == 1) and (error_check[2] == 1)):
            codeword = np.add(codeword,np.array([0,0,0,0,0,0,1])) % 2

    codeword = np.array([codeword[2],codeword[4],codeword[5],codeword[6]])

    return codeword


def data_decode(codeword):
    error_check = np.matmul(codeword,parity_check_matrix) % 2
    
    if np.sum(error_check) != 0:
        if (error_check[0] == 0) and ((error_check[1] == 0) and (error_check[2] == 1)):
            codeword = np.add(codeword,np.array([1,0,0,0,0,0,0])) % 2
        elif (error_check[1] == 0) and ((error_check[1] == 1) and (error_check[2] == 0)):
            codeword = np.add(codeword,np.array([0,1,0,0,0,0,0])) % 2
        elif (error_check[1] == 0) and ((error_check[1] == 1) and (error_check[2] == 1)):
            codeword = np.add(codeword,np.array([0,0,1,0,0,0,0])) % 2
        elif (error_check[0] == 1) and ((error_check[1] == 0) and (error_check[2] == 0)):
            codeword = np.add(codeword,np.array([0,0,0,1,0,0,0])) % 2
        elif (error_check[0] == 1) and ((error_check[1] == 0) and (error_check[2] == 1)):
            codeword = np.add(codeword,np.array([0,0,0,0,1,0,0])) % 2
        elif (error_check[0] == 1) and ((error_check[1] == 1) and (error_check[2] == 0)):
            codeword = np.add(codeword,np.array([0,0,0,0,0,1,0])) % 2
        elif (error_check[0] == 1) and ((error_check[1] == 1) and (error_check[2] == 1)):
            codeword = np.add(codeword,np.array([0,0,0,0,0,0,1])) % 2

    return codeword

def outputmsg(codeword):
    return np.array([codeword[2],codeword[4],codeword[5],codeword[6]])

def generator_test():
    msg = np.random.randint(2,size=4)
    print(msg)
    msg = data_encode(msg)
    print(msg)
    msg = data_decode(msg)
    print(msg)

if __name__ == "__main__":
    generator_test()