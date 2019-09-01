from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np
from keras.utils import plot_model
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
sns.set_palette(sns.color_palette("Set2"))
sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 3})

x = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ]
)
y = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ]
)


def create_keras_autoencoder(hidden_act="sigmoid", output_act="sigmoid", lr=0.1):
    # create model
    model = Sequential()
    model.add(Dense(3, input_dim=8, activation=hidden_act))
    model.add(Dense(8, input_dim=3, activation=output_act))
    # gradient descent
    sgd = SGD(lr=lr)
    model.compile(loss="binary_crossentropy", optimizer=sgd)

    return model


if __name__ == "__main__":
    max_epochs = 20
    lr = 0.1
    epochs = np.arange(max_epochs)

    m1_act = 'sigmoid'
    m2_act = 'softmax'

    m1 = create_keras_autoencoder(output_act=m1_act, lr=lr)
    m1_res = m1.fit(x, y, batch_size=8, epochs=max_epochs, verbose=0)
    m1_presdict = m1.predict_proba(x)
    m1_loss = m1_res.history["loss"]

    m2 = create_keras_autoencoder(output_act=m2_act, lr=lr)
    m2_res = m2.fit(x, y, batch_size=8, epochs=max_epochs, verbose=0)
    m2_presdict = m2.predict_proba(x)
    m2_loss = m2_res.history["loss"]

    # plot loss
    fig = plt.figure(figsize=(8, 5), dpi=100)
    sns.lineplot(epochs, m1_loss, label=m1_act)
    sns.lineplot(epochs, m2_loss, label=m2_act)
    plt.legend()
    plt.show()

    # plot results
    output_dim = 8
    output_sample = 8
    fig, ax = plt.subplots(int(output_sample / 2), 2, dpi=100)
    for i in range(len(m1_presdict)):
        sns.lineplot(
            range(output_dim),
            m1_presdict[i],
            drawstyle="steps-post",
            ax=ax[i // 2][i % 2],
            label=m1_act
        )
        sns.lineplot(
            range(output_dim),
            m2_presdict[i],
            drawstyle="steps-post",
            ax=ax[i // 2][i % 2],
            label=m2_act
        )
        sns.lineplot(
            range(output_dim),
            y[i],
            drawstyle="steps-post",
            ax=ax[i // 2][i % 2],
            alpha=0.5,
            label='y'
        )
    plt.show()
