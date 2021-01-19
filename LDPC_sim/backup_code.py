    def message_sneding_simulation2(self,noise_set,ylen,xlen,generator_matrix,parity_check_matrix,message_channel,decoding_tool,print_flag,iteration):

        # message generator
        dmin = int(xlen / 2)

        message = np.random.randint(2,size = ylen)
        message2 = np.random.randint(2,size = ylen)
        message = np.matmul(message,generator_matrix) % 2
        message2 = np.matmul(message2,generator_matrix) % 2

        print("original message pair:")
        print(message)
        print(message2)

        product_message = np.tensordot(message,message2,axes = 0)
        product_message_original = product_message.copy()
        product_message[product_message==0] = -1

        # noise
        product_message = np.add(product_message,np.random.normal(0,noise_set,(xlen,xlen)))

        # detected directly by hard-detection?
        product_message_copy = product_message.copy()
        product_message_copy = np.array([[1.0 if product_message_copy[i][k] > 0 else 0.0 for k in range(xlen)] for i in range(xlen)])
        product_message_copy_transpose = np.transpose(product_message_copy)

        print("product_code_with_decision")
        print(product_message_copy)

        # 1. average row that having 1 and it is non-zero row
        # 2. r - t and r + t is 1, otherwise is 0
        # 3. hard-detection? with row get? and done with col to get the codeword r.
        average_row = 0
        average_row_flag = 0
        average_col = 0
        average_col_flag = 0
        for y_axis in range(xlen):
            k1 = np.count_nonzero(product_message_copy[y_axis])
            if k1 != 0:
                average_row = average_row + k1
                average_row_flag = average_row_flag + 1
            k2 = np.count_nonzero(product_message_copy_transpose[y_axis])
            if k2 != 0:
                average_col = average_col + k2
                average_col_flag = average_col_flag + 1

        average_row = math.ceil(average_row / average_row_flag) 
        average_col = math.ceil(average_col / average_col_flag) 

        codeword1 = np.zeros(xlen)
        codeword2 = np.zeros(xlen)

        # make two codeword 
        for y_axis in range(xlen):
            if np.count_nonzero(product_message_copy[y_axis]) >= average_row:
                codeword1[y_axis] = 1

            if np.count_nonzero(product_message_copy_transpose[y_axis]) >= average_col:
                codeword2[y_axis] = 1

        average_message = np.tensordot(codeword1,codeword2,axes = 0)

        iteration = 5

        for rows in range(xlen):
            for iter_time in range(iteration):
                message_lam = np.clip(product_message_copy[rows], -0.99,0.99)
                message_lam = np.where(message_lam > 0,message_lam,(1 - message_lam) / (1 + message_lam)) + np.where(message_lam <= 0,message_lam,abs((message_lam - 1) / ( message_lam + 1)))
                message_lam = np.around(np.log(message_lam),4)
                codeword1 = decoding_tool.BDD(message_lam,codeword1,y_axis,xlen)

            # for col part
            message_lam = np.clip(product_message_copy_transpose[rows], -0.99,0.99)
            message_lam = np.where(message_lam > 0,message_lam,(1 - message_lam) / (1 + message_lam)) + np.where(message_lam <= 0,message_lam,abs((message_lam - 1) / ( message_lam + 1)))
            message_lam = np.around(np.log(message_lam),4)
            codeword2 = decoding_tool.BDD(message_lam,codeword2,y_axis,xlen)

        print("After BDD")
        print(codeword1)
        print(codeword2)

        final_message = np.tensordot(codeword1,codeword2,axes=0)

        final_message1 = codeword1.copy()
        final_message1[final_message1 == 0] = -1

        final_message2 = codeword2.copy()
        final_message2[final_message2 == 0] = -1

        print(final_message1)
        print(final_message2)

        message_lam1 = np.clip(final_message1, -0.99,0.99)
        message_lam1 = np.where(message_lam1 > 0,message_lam1,(1 - message_lam1) / (1 + message_lam1)) + np.where(message_lam1 <= 0,message_lam1,abs((message_lam1 - 1) / ( message_lam1 + 1)))
        message_lam1 = np.around(np.log10(message_lam1),4)
        final_message1 = decoding_tool.sumProductAlgorithm(message_lam1,iteration)

        message_lam2 = np.clip(final_message2, -0.99,0.99)
        message_lam2 = np.where(message_lam2 > 0,message_lam2,(1 - message_lam2) / (1 + message_lam2)) + np.where(message_lam2 <= 0,message_lam2,abs((message_lam2 - 1) / ( message_lam2 + 1)))
        message_lam2 = np.around(np.log10(message_lam2),4)
        final_messag2 = decoding_tool.sumProductAlgorithm(message_lam2,iteration)

        after_decoding_message = np.tensordot(final_message1,final_messag2,axes=0)

        fig,axs = plt.subplots(2,3)

        sns.heatmap(product_message_original, linewidth=0.1,ax = axs[0][0],cmap="Blues_r")
        axs[0][0].set_title("Product message original")

        sns.heatmap(product_message, linewidth=0.1,ax = axs[0][1],cmap="Blues_r")
        axs[0][1].set_title("Product message with AWGNnoise")

        sns.heatmap(product_message_copy, linewidth=0.1,ax = axs[0][2],cmap="Blues_r")
        axs[0][2].set_title("Product message with hard decision")

        sns.heatmap(average_message, linewidth=0.1,ax = axs[1][0],cmap="Blues_r")
        axs[1][0].set_title("Product message with average filter")

        sns.heatmap(final_message, linewidth=0.1,ax = axs[1][1],cmap="Blues_r")
        axs[1][1].set_title("Final message")

        sns.heatmap(after_decoding_message, linewidth=0.1,ax = axs[1][2],cmap="Blues_r")
        axs[1][2].set_title("After decoding message")

        plt.show()