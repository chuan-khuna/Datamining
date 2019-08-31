import tensorflow as tf
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns

import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

sns.set_style("whitegrid")
sns.set_palette(sns.color_palette("Set2"))
sns.set_context('notebook', font_scale=1.25, rc={"lines.linewidth": 3})


# setting hidden layer activation function
## tanh or sigmoid
hidden_layer_activation = 'sigmoid'

# setting cost function
## cross-entropy or mean-square
cost_function_type = 'cross-entropy'


class XORNN:
    def __init__(self, max_iteration, learning_rate, cost_function_type, hidden_node_act):
        self.max_iteration = max_iteration
        self.learning_rate = learning_rate
        self.cost_function_type = cost_function_type
        self.hidden_node_act = hidden_node_act
        self.costs = []
        self.weight1 = []
        self.weight2 = []
        self.bias1 = []
        self.bias2 = []
        self.create_model()
        self.start_learning()

    def create_model(self):

        # tensorflow input
        self.x_ = tf.placeholder(tf.float32, shape=[4, 2], name="x-input")
        self.y_ = tf.placeholder(tf.float32, shape=[4, 1], name="y-output")

        # tensorflow weights & biases
        self.w1 = tf.Variable(tf.random_uniform([2, 2], -1, 1), name="w1")
        self.w2 = tf.Variable(tf.random_uniform([2, 1], -1, 1), name="w2")
        self.b1 = tf.Variable(tf.zeros([2]), name="b1")
        self.b2 = tf.Variable(tf.zeros([1]), name="b2")

        # setting hidden layer node activation function
        if self.hidden_node_act == 'sigmoid':
            self.layer1 = tf.sigmoid(tf.matmul(self.x_, self.w1) + self.b1)
        elif self.hidden_node_act == 'tanh':
            self.layer1 = tf.tanh(tf.matmul(self.x_, self.w1) + self.b1)
        else:
            self.layer1 = tf.tanh(tf.matmul(self.x_, self.w1) + self.b1)

        # setting output layer node activation function
        self.layer2 = tf.sigmoid(tf.matmul(self.layer1, self.w2) + self.b2)

        # setting cost function type
        if self.cost_function_type == 'cross-entropy':
            self.cost_function = tf.reduce_mean(
                -1*((self.y_ * tf.log(self.layer2)) +
                    ((1 - self.y_) * tf.log(1 - self.layer2)))
            )
        elif self.cost_function_type == 'mean-square':
            self.cost_function = tf.reduce_mean((self.y_ - self.layer2) ** 2)

    def learning_step(self):
        return tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost_function)

    def start_learning(self):
        XOR_X = [[0, 0], [0, 1], [1, 0], [1, 1]]
        XOR_Y = [[0], [1], [1], [0]]
        learning_algor = self.learning_step()
        init_model = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init_model)
        start_time = time.time()
        cost = float(
            sess.run(self.cost_function, feed_dict={self.x_: XOR_X, self.y_: XOR_Y}))
        self.costs.append(round(cost, 4))
        self.weight1.append(sess.run(self.w1))
        self.weight2.append(sess.run(self.w2))
        self.bias1.append(sess.run(self.b1))
        self.bias2.append(sess.run(self.b2))
        for i in range(self.max_iteration):
            error = sess.run(learning_algor, feed_dict={
                             self.x_: XOR_X, self.y_: XOR_Y})
            cost = float(
                sess.run(self.cost_function, feed_dict={self.x_: XOR_X, self.y_: XOR_Y}))
            self.costs.append(round(cost, 4))
            self.weight1.append(sess.run(self.w1))
            self.weight2.append(sess.run(self.w2))
            self.bias1.append(sess.run(self.b1))
            self.bias2.append(sess.run(self.b2))

        # convert to np array
        self.costs = np.array(self.costs)
        self.weight1 = np.array(self.weight1)
        self.weight2 = np.array(self.weight2)
        self.bias1 = np.array(self.bias1)
        self.bias2 = np.array(self.bias2)

        # print details
        l1 = sess.run(self.layer1, feed_dict={self.x_: XOR_X, self.y_: XOR_Y})
        predicted = sess.run(self.layer2, feed_dict={
                             self.layer1: l1, self.y_: XOR_Y})
        predicted = [round(i[0], 2) for i in predicted]
        print('\n\n-----------------------------------------------------------')
        print(f"max iteration \t {self.max_iteration}")
        print(f"learning rate \t {self.learning_rate}")
        print(f"hidden node \t {self.hidden_node_act}")
        print(f"cost function \t {self.cost_function_type}")
        print(f"time elapesed \t {round(time.time() - start_time, 3)} s")
        print(f"cost \t {self.costs[0]} -> {self.costs[-1]}")
        print(f"predicted \t {predicted}")
        print('-----------------------------------------------------------')


if __name__ == "__main__":

    max_iter = 10000
    learning_rate = 0.1
    tanh = XORNN(max_iteration=max_iter, learning_rate=learning_rate,
                 cost_function_type='mean-square', hidden_node_act='tanh')
    sigmoid = XORNN(max_iteration=max_iter, learning_rate=learning_rate,
                    cost_function_type='mean-square', hidden_node_act='sigmoid')

    x_iter = np.arange(max_iter+1)
    fig = plt.figure(figsize=(8, 5), dpi=120)
    sns.lineplot(x_iter, tanh.costs, label='tanh')
    sns.lineplot(x_iter, sigmoid.costs, label='sigmoid')
    plt.show()

    tdf = pd.DataFrame({
        'iteration': x_iter,
        'cost': tanh.costs,
        'l1_n1_w0': tanh.bias1[:, 0],
        'l1_n1_w1': tanh.weight1[:, 0, 0],
        'l1_n1_w2': tanh.weight1[:, 0, 1],
        'l1_n2_w0': tanh.bias1[:, 1],
        'l1_n2_w1': tanh.weight1[:, 1, 0],
        'l1_n2_w2': tanh.weight1[:, 1, 1],
        'l2_n1_w0': tanh.bias1[:, 0],
        'l2_n1_w1': tanh.weight2[:, 0, 0],
        'l2_n1_w2': tanh.weight2[:, 1, 0]
    })

    sdf = pd.DataFrame({
        'iteration': x_iter,
        'cost': sigmoid.costs,
        'l1_n1_w0': sigmoid.bias1[:, 0],
        'l1_n1_w1': sigmoid.weight1[:, 0, 0],
        'l1_n1_w2': sigmoid.weight1[:, 0, 1],
        'l1_n2_w0': sigmoid.bias1[:, 1],
        'l1_n2_w1': sigmoid.weight1[:, 1, 0],
        'l1_n2_w2': sigmoid.weight1[:, 1, 1],
        'l2_n1_w0': sigmoid.bias1[:, 0],
        'l2_n1_w1': sigmoid.weight2[:, 0, 0],
        'l2_n1_w2': sigmoid.weight2[:, 1, 0]
    })

    tdf.to_csv('xor-tanh-mse.csv', index=False)
    sdf.to_csv('xor-sigmoid-mse.csv', index=False)
