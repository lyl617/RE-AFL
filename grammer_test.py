import re

def plot_learning_curve(file_path, fig_root_path):
    file_in = open(file_path, "r")
    row_data = file_in.readlines()
    row_cnt = len(row_data)

    epoch_data = []
    loss_data = []
    acc_data = []

    if row_cnt <= 0:
        print("--[Debug] No data in log file!")
    else:
        for row in row_data:
            row = row.strip()
            if ('Test set' in row) and (not ('Debug' in row) and not ('-->' in row)):
                tmp = row.split(':')
                epoch_data.append(int(re.findall(r"\d+", tmp[2].strip())[0]))
                loss_data.append(float(tmp[3].split(',')[0]))
                acc_data.append(float(re.findall(r"[1-9]\d*", tmp[4])[0]) / float(re.findall("[1-9]\d*", tmp[4])[1]))



    file_in.close()

    if not len(epoch_data) == 0:

        print("plot figure ok!")

        legend_loc = 'best'

        plt.figure(1)
        plt.plot(epoch_data, loss_data, 'b-')
        plt.xlabel('Epochs', fontdict=font1)
        plt.ylabel('Loss', fontdict=font1)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(('Loss',), loc=legend_loc, fontsize=14)

        plt.grid(True)

        fig_path = fig_root_path.replace("FIGTYPE", 'loss')
        plt.savefig(fig_path, dpi=600, bbox_inches='tight')

        plt.figure(2)
        plt.plot(epoch_data, acc_data, 'b-')
        plt.xlabel('Epochs', fontdict=font1)
        plt.ylabel('Accuracy', fontdict=font1)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(('Accuracy',), loc=legend_loc, fontsize=14)

        plt.grid(True)

        fig_path = fig_root_path.replace("FIGTYPE", 'acc')
        plt.savefig(fig_path, dpi=600, bbox_inches='tight')


plot_learning_curve('server_result/server_0/log/alpha_1.0/model-type_CNN_dataset-typeCIFAR10_tx2nums10_randomdata_lr0.01_epoch10_local1/model_acc_loss.txt', '.')