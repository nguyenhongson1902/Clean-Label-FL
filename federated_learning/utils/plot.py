import matplotlib.pyplot as plt


# labels = {
#     'Client 0': {8: 2363, 3: 3154, 2: 409, 0: 2538, 1: 2206, 6: 387, 5: 178, 7: 128, 4: 40},
#     'Client 1': {1: 215, 4: 1777, 3: 1423, 9: 878, 5: 2276, 7: 498, 8: 446, 2: 202, 0: 43, 6: 1},
#     'Client 2': {4: 2349, 1: 1517, 2: 1244, 7: 2175, 8: 682, 5: 208, 6: 259, 3: 192, 9: 114, 0: 23},
#     'Client 3': {9: 2583, 7: 1057, 6: 897, 1: 1004, 2: 2473, 8: 1368, 4: 108, 0: 1053, 5: 512},
#     'Client 4': {6: 3456, 0: 1343, 2: 672, 9: 1425, 5: 1826, 7: 1142, 4: 726, 8: 141, 3: 231, 1: 58}
# }

def plot_data_dis_to_file(labels, num_class=10, data_file_name = './plots/data_distribution.png'):

    labels_list = list(labels.keys())
    num_labels = len(labels_list)
    label_ids = set()
    

    for client in labels:
        label_ids.update(labels[client].keys())

    label_ids = sorted(label_ids)

    # Initialize a list of zeros for each client and label
    label_counts = {client: [0] * num_class for client in labels}
    # print(label_counts)
    # Populate the count for each label and client
    for i, client in enumerate(labels_list):
        for j, label in enumerate(label_ids):
            label_counts[client][j] = labels[client].get(label, 0)

    # print(label_counts)

    # Plotting the bar chart
    fig, ax = plt.subplots(figsize=(10, 8))
    width = 1 / (num_labels + 1)
    x = [i for i in range(num_class)]
    # plot a bar chart where the x-axis represents the clients, and the y-axis represents the number of labels
    for i, (client, counts) in enumerate(label_counts.items()):
        left = [t + width * i for t in x]
        ax.bar(left, counts, width, label=client)

    ax.set_xticks(x)
    ax.set_xticklabels(label_ids)
    ax.set_xlabel('Label ID')
    ax.set_ylabel('Count')
    ax.set_title('Label distribution across clients')
    ax.legend()
    # plt.show()

    # Save the plot
    fig.savefig(data_file_name)

# write_data_dis_to_file(labels)