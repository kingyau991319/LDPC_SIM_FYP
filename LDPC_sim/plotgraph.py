import numpy as np
from numpy.linalg import matrix_rank
import pandas as pd
import matplotlib.pyplot as plt 
from scipy import special
import warnings

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx],idx

def plot_SNR_graph(flag = 0):

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
        print()
        x_axis_value = df_filter['SNRbDB'].to_numpy()
        y_axis_value = df_filter['average_probaility_error'].to_numpy()
        plt.scatter(x_axis_value,y_axis_value)
        plt.plot(x_axis_value,y_axis_value,label=name_k)


        # I plot the subtraction here
        if flag != 1:
            #1. then, plot the 1e-05 for the LDPC part ~ ndB
            coded_y_compare,coded_x_compare = find_nearest(y_axis_value,1e-05)
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
            text_str = str(unencoded_x_compare-coded_x_compare)
            text_str = text_str[:3] + "dB"
            ax.text(midpoint_text_x,midpoint_text_y, text_str, style='italic')

            #3. make a dotted line between them 
            #4. make a name on here
            #plt.scatter(coded_x_compare,coded_y_compare)
            pass


    # for label or title
    ax.set_yscale('log')
    plt.xlabel("E$_{b}$/N$_{0}$ (dB)")
    plt.ylabel("bit error rate(BER)")
    plt.ylim((10**-6,1))
    plt.grid(True,linestyle='-.')

    # show the graph
    plt.margins(x=0)
    plt.legend()
    plt.show()

# it consists the data of average_probaility_error,detected_bit_error,undetected_bit_error
def plot_SNR_graph2(name):

    warnings.filterwarnings('ignore')
    fig, ax = plt.subplots()


    df = pd.read_csv("sim_result.csv")
    filter = (df["LDPC_code"] == name)

    # encoded graph
    df1 = df[filter]
    df1 = df1[['LDPC_code','SNRbDB','average_probaility_error','noise_set']]
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
    coded_y_compare,coded_x_compare = find_nearest(average_probaility_error,1e-05)
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
    text_str = str(unencoded_x_compare-coded_x_compare)
    text_str = text_str[:3] + "dB"
    ax.text(midpoint_text_x,midpoint_text_y, text_str, style='italic')

    ax.set_yscale('log')

    plt.xlabel("E$_{b}$/N$_{0}$ (dB)")
    plt.ylabel("bit error rate(BER)")
    plt.ylim((10**-6,1))

    plt.grid(True,linestyle='-.')

    plt.title(name)
    plt.scatter(x_axis_value,average_probaility_error)
    plt.plot(x_axis_value,average_probaility_error,label='average_probaility_error')
    plt.scatter(x_axis_value,detected_bit_error,marker='s')
    plt.plot(x_axis_value,detected_bit_error,label='detected_bit_error')
    plt.scatter(x_axis_value,undetected_bit_error,marker='triangle_up')
    plt.plot(x_axis_value,undetected_bit_error,label='undetected_bit_error')

    # plt.axis('tight') 
    plt.margins(x=0)

    # Q function with ecnoding plot
    plt.legend()
    plt.show()

#plot_SNR_graph("MacKeyCodeTest","MacKeyCodeTest2")
plot_SNR_graph2()