import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from scipy import special
import warnings

# find the first nearest value
def find_nearest(array, nearest_num):
    array = np.asarray(array)
    idx = (np.abs(array - nearest_num)).argmin()
    value = array[idx]
    return value,idx

# find the second nearest value
def find_second_nearest(array, nearest_num):
    array = np.asarray(array)
    arr_copy = array.copy()
    arr_nearest_val,idx = find_nearest(arr_copy,nearest_num)

    if arr_nearest_val > 0 and arr_nearest_val < len(array):
        prev_value = arr_copy[idx-1]
        next_value = arr_copy[idx+1]

        if arr_nearest_val < nearest_num:
            return prev_value,idx-1

        else:
            return next_value,idx+1

    else:
        raise Exception("ArrayOutOfBound with no value close to the second nearest point.")

def plot_SNR_graph(flag = 0):

    # get the min of the y_axis so I can get the relative poisition
    y_min_value = 0
    last_plot_annotoate_BER = 0
    last_plot_annotoate_SNR = 0

    warnings.filterwarnings('ignore')
    fig, ax = plt.subplots()

    # take the data that I need
    df = pd.read_csv("sim_result.csv")

    df_name = df["LDPC_code"]
    df_name = df_name.drop_duplicates()

    # for unencoded BPSK
    unencoded_x = np.linspace(0,15,10000)
    unencoded_y = unencoded_x.copy()
    k = np.sqrt(2 * (10 ** (unencoded_y / 10) ))
    unencoded_y = 0.5 * special.erfc(k/np.sqrt(2))
    plt.plot(unencoded_x,unencoded_y,label="uncoded data transmission")

    # for each LDPC name

    for name_k in df_name:
        filter = df["LDPC_code"] == name_k
        #print(name_k)

        df_filter = df[filter]
        df_filter = df_filter[['LDPC_code','SNRbDB','average_probaility_error','noise_set']]
        df_filter = df_filter.sort_values(by=['SNRbDB'])
        print(df_filter)
        x_axis_value = df_filter['SNRbDB'].to_numpy()
        y_axis_value = df_filter['average_probaility_error'].to_numpy()
        if min(y_axis_value) < y_min_value or y_min_value == 0:
            y_min_value = min(y_axis_value)
        plt.scatter(x_axis_value,y_axis_value)
        plt.plot(x_axis_value,y_axis_value,label=name_k)


        # I plot the subtraction here
        if flag != 1:
            #1. then, plot the 1e-05 for the LDPC part ~ ndB
            first_nearest_num,idx_1 = find_nearest(y_axis_value,1e-05) # (val, idx)
            nearest_SNR1 = x_axis_value[idx_1]
            second_nearest_num,idx_2 = find_second_nearest(y_axis_value,1e-05) # (val, idx)
            nearest_SNR2 = x_axis_value[idx_2]

            if(second_nearest_num < first_nearest_num):
                linear_value = np.arange(second_nearest_num,first_nearest_num,1e-07)
                linear_value_len = len(linear_value)
                step_num = (nearest_SNR1 - nearest_SNR2) / linear_value_len
                linear_SNR = np.arange(nearest_SNR2,nearest_SNR1,step_num)

            else:
                linear_value = np.arange(first_nearest_num,second_nearest_num,1e-07)
                linear_value_len = len(linear_value)
                step_num = (nearest_SNR2 - nearest_SNR1) / linear_value_len
                linear_SNR = np.arange(nearest_SNR1,nearest_SNR2,step_num)

            fin_nearest_num,idx = find_nearest(linear_value,1e-05)
            fin_SNR = linear_SNR[idx]

            if last_plot_annotoate_BER == 0 and last_plot_annotoate_SNR == 0:
                #2. first, plot the 1e-05 for unencoded graph ~ 9.6dB
                near_unencoded_value,near_unencoded_idx = find_nearest(unencoded_y,fin_nearest_num)
                near_unencoded_SNR = unencoded_x[near_unencoded_idx]
                last_plot_annotoate_BER = near_unencoded_value
                last_plot_annotoate_SNR = near_unencoded_SNR

            #3. build the annotate
            plt.annotate(s='', xy=(fin_SNR,fin_nearest_num), xytext=(last_plot_annotoate_SNR,last_plot_annotoate_BER), arrowprops=dict(arrowstyle='<->'))

            midpoint_text_x = (last_plot_annotoate_SNR + fin_SNR) / 2 - 0.5
            midpoint_text_y = last_plot_annotoate_BER + 0.3 * last_plot_annotoate_BER
            text_str = str(last_plot_annotoate_SNR - fin_SNR)
            text_str = text_str[:3] + "dB"
            ax.text(midpoint_text_x,midpoint_text_y, text_str, style='italic')

            last_plot_annotoate_BER = fin_nearest_num
            last_plot_annotoate_SNR = fin_SNR

    # for label or title
    ax.set_yscale('log')
    plt.xlabel("E$_{b}$/N$_{0}$ (dB)")
    plt.ylabel("bit error rate(BER)")

    # print(y_min_value)
    plt.ylim(y_min_value / 1.618)
    # plt.ylim((10**-8,1))
    plt.grid(True,linestyle='-.')

    # show the graph
    plt.margins(x=0)
    plt.legend()
    plt.show()

# it consists the data of average_probaility_error,detected_bit_error,undetected_bit_error
def plot_SNR_graph2(name,ylim_num):

    warnings.filterwarnings('ignore')
    fig, ax = plt.subplots()


    df = pd.read_csv("sim_result.csv")
    filter = (df["LDPC_code"] == name)

    # encoded graph
    df1 = df[filter]
    df1 = df1[['LDPC_code','SNRbDB','average_probaility_error','detected_bit_error','undetected_bit_error','noise_set']]
    df1 = df1.sort_values(by=['SNRbDB'])
    x_axis_value = df1['SNRbDB'].to_numpy()
    average_probaility_error = df1['average_probaility_error'].to_numpy()
    detected_bit_error = df1['detected_bit_error'].to_numpy()
    undetected_bit_error = df1['undetected_bit_error'].to_numpy()

    # unencoded graph
    unencoded_x = np.linspace(0,15,1000)
    unencoded_y = unencoded_x.copy()
    k = np.sqrt(2 * (10 ** (unencoded_y / 10) ))
    unencoded_y = 0.5 * special.erfc(k/np.sqrt(2))
    plt.plot(unencoded_x,unencoded_y,label="uncoded data transmission")

    # text plot
    coded_y_compare,coded_x_compare = find_nearest(average_probaility_error,1e-04)
    coded_x_compare = x_axis_value[coded_x_compare]
    #2. first, plot the 1e-05 for unencoded graph ~ 9.6dB
    unencoded_y_compare = coded_y_compare
    unused,unencoded_x_compare = find_nearest(unencoded_y,unencoded_y_compare)
    unencoded_x_compare = unencoded_x[unencoded_x_compare]
    x_axis = [coded_x_compare,unencoded_x_compare]
    y_axis = [coded_y_compare,unencoded_y_compare]
    print(x_axis,y_axis)
    #plt.plot(x_axis,y_axis)
    plt.annotate(s='', xy=(coded_x_compare,coded_y_compare), xytext=(unencoded_x_compare,unencoded_y_compare), arrowprops=dict(arrowstyle='<->'))

    midpoint_text_x = (unencoded_x_compare + coded_x_compare) / 2 - 0.5
    midpoint_text_y = unencoded_y_compare + 0.3 * unencoded_y_compare
    text_str = str(abs(unencoded_x_compare-coded_x_compare))
    text_str = text_str[:3] + "dB"
    ax.text(midpoint_text_x,midpoint_text_y, text_str, style='italic')

    ax.set_yscale('log')

    plt.xlabel("E$_{b}$/N$_{0}$ (dB)")
    plt.ylabel("bit error rate(BER)")
    plt.ylim((ylim_num,1))

    plt.grid(True,linestyle='-.')

    plt.title(name)

    plt.scatter(x_axis_value,average_probaility_error)
    plt.plot(x_axis_value,average_probaility_error,label='average_probaility_error')

    plt.scatter(x_axis_value,detected_bit_error,marker='s')
    plt.plot(x_axis_value,detected_bit_error,label='detected_bit_error')

    plt.scatter(x_axis_value,undetected_bit_error,marker='s')
    plt.plot(x_axis_value,undetected_bit_error,label='undetected_bit_error')

    # plt.axis('tight') 
    plt.margins(x=0)

    # Q function with ecnoding plot
    plt.legend()
    plt.show()

#plot_SNR_graph("MacKeyCodeTest","MacKeyCodeTest2")

# plot_SNR_graph2("MacKeyCode_96.3.965",ylim_num)
# array = [1.16250000e-01, 1.27083333e-02, 2.50000000e-03, 6.25000000e-04, 1.35833333e-04, 1.29583333e-05, 3.29166667e-06]
plot_SNR_graph()

# array = [1,2,3,4,5,6,7]
# find_val = 1e-05

# nearest_value,index = find_nearest(array,find_val)
# second_nearest_value,index = find_second_nearest(array,find_val)
# print("nearest_value",nearest_value)
# print("second_nearest_value",second_nearest_value)