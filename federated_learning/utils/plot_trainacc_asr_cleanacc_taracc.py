#ours -- higher_configuration
from matplotlib import pyplot as plt
import numpy as np


def plot_trainacc_asr_cleanacc_taracc(training_epochs, train_ACC, test_ACC, clean_ACC, target_ACC):
    half = np.arange(0, training_epochs)
    plt.figure(figsize=(12.5,8))
    plt.plot(half, np.asarray(train_ACC)[half], label='Training ACC', linestyle="-.", marker="o", linewidth=3.0, markersize = 8)
    plt.plot(half, np.asarray(test_ACC)[half], label='Attack success rate', linestyle="-.", marker="o", linewidth=3.0, markersize = 8) # ASR
    plt.plot(half, np.asarray(clean_ACC)[half], label='Clean test ACC', linestyle="-.", marker="o", linewidth=3.0, markersize = 8) # ACC
    plt.plot(half, np.asarray(target_ACC)[half], label='Target class clean test ACC', linestyle="-", marker="o", linewidth=3.0, markersize = 8) # Tar-ACC
    # plt.plot(half, np.asarray(test_unl_ACC)[half], label='protected test ACC', linestyle="-.", marker="o", linewidth=3.0, markersize = 8)
    plt.ylabel('ACC', fontsize=24)
    plt.xticks(fontsize=20)
    plt.xlabel('Epoches', fontsize=24)
    plt.yticks(np.arange(0,1.1, 0.1),fontsize=20)
    plt.legend(fontsize=20,bbox_to_anchor=(1.016, 1.2),ncol=2)
    plt.grid(color="gray", linestyle="-")

    # save the plot as a PNG file
    plt.savefig('my_plot.png')
    
    # plt.show()

    # dis_idx = clean_ACC.index(max(clean_ACC))
    # print(train_ACC[dis_idx])
    # print('attack',test_ACC[dis_idx])
    # print(clean_ACC.index(max(clean_ACC)))
    # print('all class clean', clean_ACC[dis_idx])
    # print('target clean',target_ACC[dis_idx])