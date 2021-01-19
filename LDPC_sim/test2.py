import numpy as np 

f = open("test_code","r")

str = f.read()
str = str.split()
str = np.array(str)
str_len = len(str)
input_num = str.astype(np.int)

xlen = input_num[0]
ylen = input_num[1]
code_rate = ylen / xlen
wc = input_num[2]
wr = input_num[3]
wc_num = input_num[xlen+4:xlen+ylen+4]

lower_bound_of_array = str_len - np.sum(wc_num)

input_num = input_num[lower_bound_of_array:]
parity_check_matrix = np.zeros((ylen,xlen))
x_index_index = 0

for y_axis in range(ylen):
    for x_axis in range(wc_num[y_axis]):
        x_index = input_num[x_index_index] - 1
        parity_check_matrix[y_axis][x_index] = 1
        x_index_index = x_index_index + 1

# for y_axis in range(ylen):
#     for x_axis in range(ylen):
#         parity_check_matrix[y_axis][ylen+x_axis] = 0
#     parity_check_matrix[y_axis][ylen+y_axis] = 1



# for k in range(100000):


# parity_check_matrix_trans = parity_check_matrix[:ylen,:ylen].copy().transpose()
        
# generator_matrix = np.zeros((ylen,xlen))
# for y_axis in range(ylen):
#     for x_axis in range(xlen):
#         if x_axis >= ylen:
#             generator_matrix[y_axis][x_axis] = parity_check_matrix_trans[y_axis][x_axis-ylen]
#     generator_matrix[y_axis][y_axis] = 1

parity_check_matrix_to_list = parity_check_matrix.tolist()
# generator_matrix_to_list = generator_matrix.tolist()

for y_axis in range(ylen):
    for x_axis in range(xlen):
        print(int(parity_check_matrix_to_list[y_axis][x_axis]),end='')
    print()

# print()

# for y_axis in range(ylen):
#     for x_axis in range(xlen):
#         print(int(generator_matrix_to_list[y_axis][x_axis]),end='')
#     print()