import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_style("whitegrid")
sns.set_palette(sns.color_palette("Paired"))
sns.set_context('notebook', font_scale=1.1, rc={"lines.linewidth": 3})


def compare_plot(iteration, data_arr, label_arr, x_label, y_label):
    plt.figure(figsize=(8, 5), dpi=120)
    for i in range(len(data_arr)):
        sns.lineplot(iteration, data_arr[i], label=label_arr[i])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


if __name__ == "__main__":
    sdf = pd.read_csv('xor-sigmoid-cross.csv')
    tdf = pd.read_csv('xor-tanh-cross.csv')
    iteration = tdf['iteration']

    # # plot cost
    compare_plot(iteration, [tdf['cost'], sdf['cost']],
                 ['tanh', 'sigmoid'], x_label='epoch', y_label='cost')

    # plot weights at layer 1 node 1
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=120)
    fig.suptitle('Hidden Layer Weight')
    sns.lineplot(iteration, tdf['l1_n1_w0'], label='w0 tanh', ax=axes[0])
    sns.lineplot(iteration, sdf['l1_n1_w0'], label='w0 sigmoid', ax=axes[0])
    sns.lineplot(iteration, tdf['l1_n1_w1'], label='w1 tanh', ax=axes[0])
    sns.lineplot(iteration, sdf['l1_n1_w1'], label='w1 sigmoid', ax=axes[0])
    sns.lineplot(iteration, tdf['l1_n1_w2'], label='w2 tanh', ax=axes[0])
    sns.lineplot(iteration, sdf['l1_n1_w2'], label='w2 sigmoid', ax=axes[0])
    axes[0].set_ylabel('layer-1 node-1 weight')
    sns.lineplot(iteration, tdf['l1_n2_w0'], label='w0 tanh', ax=axes[1])
    sns.lineplot(iteration, sdf['l1_n2_w0'], label='w0 sigmoid', ax=axes[1])
    sns.lineplot(iteration, tdf['l1_n2_w1'], label='w1 tanh', ax=axes[1])
    sns.lineplot(iteration, sdf['l1_n2_w1'], label='w1 sigmoid', ax=axes[1])
    sns.lineplot(iteration, tdf['l1_n2_w2'], label='w2 tanh', ax=axes[1])
    sns.lineplot(iteration, sdf['l1_n2_w2'], label='w2 sigmoid', ax=axes[1])
    axes[1].set_ylabel('layer-1 node-2 weight')
    plt.show()

    # plot weights at layer 2 node 1
    fig = plt.figure(figsize=(8, 5), dpi=120)
    sns.lineplot(iteration, tdf['l2_n1_w0'], label='w0 tanh')
    sns.lineplot(iteration, sdf['l2_n1_w0'], label='w0 sigmoid')
    sns.lineplot(iteration, tdf['l2_n1_w1'], label='w1 tanh')
    sns.lineplot(iteration, sdf['l2_n1_w1'], label='w1 sigmoid')
    sns.lineplot(iteration, tdf['l2_n1_w2'], label='w2 tanh')
    sns.lineplot(iteration, sdf['l2_n1_w2'], label='w2 sigmoid')
    plt.ylabel('layer-2 node-1 weight')
    plt.show()

    # plot hidden layer boundary
    x_val = np.arange(-2, 2, 0.2)
    w0, w1, w2 = np.array(
        sdf['l1_n1_w0'])[-1], np.array(sdf['l1_n1_w1'])[-1], np.array(sdf['l1_n1_w2'])[-1]
    y_val = -(w1*x_val + w0)/w2
    sns.lineplot(x_val, y_val, label='node 1')
    w0, w1, w2 = np.array(
        sdf['l1_n2_w0'])[-1], np.array(sdf['l1_n2_w1'])[-1], np.array(sdf['l1_n2_w2'])[-1]
    y_val = -(w1*x_val + w0)/w2
    sns.lineplot(x_val, y_val, label='node 2')
    sns.scatterplot([0, 1], [1, 0], s=100, marker='o', color='red')
    sns.scatterplot([0, 1], [0, 1], s=100, marker='X', color='red')
    plt.axis([-0.5, 1.5, -0.5, 1.5])
    plt.show()
