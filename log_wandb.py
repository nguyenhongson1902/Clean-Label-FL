# import argparse
# from easydict import EasyDict
# import json
# import sys


# def get_args(json_file):
# 	with open(json_file, "r") as f:
# 		args = json.load(f)
# 	return EasyDict(args)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='A test program.')

#     parser.add_argument('--foo-test', action='store_true', help='foo help')
#     parser.add_argument('--test', type=str, help='bar help')

#     args = parser.parse_args()

#     args_dict = get_args("test.json")
#     print(args_dict)
#     print(args.test)
#     print(type(args.test))
#     print(args_dict.cuda)

#     # print(type(args.test))

import wandb
import pandas as pd
import os


if __name__ == '__main__':
    wandb_name = f"cifar10__fedavg__num_workers_5__num_selected_workers_5__num_poisoned_workers_1__poison_amount_ratio_0.5__local_epochs_1__exp_31"
    wandb.init(name=wandb_name, project='clean-label-attack-fl', entity="nguyenhongsonk62hust")
    df = pd.read_csv("./4000_results.csv", header=None)
    df.columns = ['asr', 'clean_acc', 'tar_acc']
    for epoch, row in df.iterrows():
        acc = row[0]
        acc_clean = row[1]
        acc_tar = row[2]
        wandb.log({"comm_round": epoch+1, "asr": acc, "acc_clean": acc_clean, "acc_tar": acc_tar})

    df = pd.read_csv('./4000_workers_selected.csv', header=None)
    
    table =  wandb.Table(dataframe=df)
    wandb.log({"workers_selected": table})

    # Method 2
    # table = wandb.Table(columns=columns)
    # table.add_data("I love my phone", "1", "1")
    # table.add_data("My phone sucks", "0", "-1")
    # wandb.log({"examples": table})