import numpy as np

import random

from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Activation

from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from tensorflow import keras
from keras.utils.np_utils import to_categorical
from keras.layers.convolutional import Conv2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten
from tensorflow.keras import initializers
import scipy.io
from scipy.stats import bernoulli
import math
import os.path
from os import path

# defining user infos
class UserInfos:
    # The class "constructor" - It's actually an initializer
    def __init__(self, name,  errors):
        self.name = name
        self.errors = errors


def split_train_data(x_train, y_train):
    zip_data = list(zip(x_train, y_train))
    zip_data_sorted = [[], [], [], [], [], [], [], [], [], []];
    sort_el = 0
    sort_itr = 0
    N = 60000
    done = False

    while (done == False):
        dt, l = zip_data[sort_itr]
        if l[sort_el] == 1:
            zip_data_sorted[sort_el].append(zip_data[sort_itr])

        sort_itr += 1
        if sort_itr == N and sort_el <= 8:
            sort_el += 1
            sort_itr = 0
        if sort_itr == N and sort_el == 9:
            done = True

    return zip_data_sorted


# shares = [[5498, 350, 25, 25, 25], [6300, 367, 25, 25, 25], [5000, 700, 100, 100, 58],
#           [2750, 1453, 1165, 663, 100], [200, 25, 280, 3433, 1904], [52, 25, 280, 3160, 1904],
#           [50, 25, 280, 3659, 1904], [50, 25, 280, 4006, 1904], [50, 15, 280, 3602, 1904],
#           [50, 15, 285, 3327, 2272]]
shares = [[494, 10, 10, 5, 10], [492, 10, 10, 5, 10], [492, 10, 10, 5, 10],
          [492, 10, 10, 5, 10], [5, 43, 43, 196, 193], [5, 43, 43, 196, 193],
          [5, 43, 43, 196, 193], [5, 43, 43, 196, 193], [5, 43, 43, 196, 193],
          [5, 45, 45, 200, 195]]
# shares = [[200,30,30,30,30],[200,30,30,30,30],[200,30,30,30,30],[200,30,30,30,30],[200,30,30,180,180],[200,30,30,180,180],[200,30,30,180,180],[200,30,30,180,180],[200,30,30,180,180],[200,30,30,180,180]];
PSNR1 = [-5, 5, 5, 5, 5 ]
PSNR2 = [0, 5, 5, 5, 5 ]
PSNR3 = [5, 5, 5, 5, 5 ]
PSNR4 = [30, 5, 5, 5, 5 ]
T_max = 150;wt = 10;
Wait = T_max/wt;
D_k=[2000 ,300, 300, 1200, 1200];


def AddNoise(data, psnr):
    noisyData = [];
    var_noise = (1) / (10 ** (psnr / 10))
    for el in data:
        x, y = el
        x_n = x + np.random.normal(loc=0.0, scale=var_noise, size=x.shape);
        noisyData.append((x_n, y))
    return noisyData


# functions
def load():
    # (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    # x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    x_train = np.reshape(x_train, (-1, 784))
    x_test = np.reshape(x_test, (-1, 784))
    x_train = x_train / 255
    x_test = x_test / 255
    y_train_one_hot = to_categorical(y_train)
    y_test_one_hot = to_categorical(y_test)
    return x_train, y_train_one_hot, x_test, y_test_one_hot


def create_clients_iid(image_list, label_list, name, run, num_clients, t, initial='clients'):
    # finding d_k
    D_k = [];
    for u in users_details:
        if u.name == name:
            D_k = u.D_ks[run][0]
            break
    # create a list of client names
    client_names = ['{}_{}'.format(initial, i + 1) for i in range(num_clients)]

    # randomize the data
    data = list(zip(image_list, label_list))
    random.shuffle(data)
    # if t=='wc':
    #   data = data[::-1]
    # data = random.sample(data, len(data))
    # shard data and place at each client
    shards = {}

    for i, v in enumerate(D_k):
        shards[i] = data[len(shards):len(shards) + v]

    # number of clients must equal number of shards
    assert (len(shards) == len(client_names))
    return {client_names[i]: shards[i] for i in range(len(client_names))}

def create_clients_Niid(zip_data_sorted, name, num_clients, shares, initial='clients'):
    # finding d_k

    # create a list of client names
    client_names = ['{}_{}'.format(initial, i + 1) for i in range(num_clients)]

    # randomize the data
    # data = list(zip(image_list, label_list))
    # random.shuffle(data)
    # if t=='wc':
    #   data = data[::-1]
    # data = random.sample(data, len(data))
    # shard data and place at each client
    shards = [[], [], [], [], []]

    for j in range(10):
        zip_data_sorted[j] = random.sample(zip_data_sorted[j], len(zip_data_sorted[j]))
        for i, v in enumerate(D_k):
            if i >= 1:
                shards[i].extend(zip_data_sorted[j][shares[j][i - 1]:shares[j][i - 1] + shares[j][i]])
            else:
                shards[i].extend(zip_data_sorted[j][0: shares[j][i]])

    # number of clients must equal number of shards
    assert (len(shards) == len(client_names))
    return {client_names[i]: shards[i] for i in range(len(client_names))}


def batch_data(data_shard, bs):
    '''Takes in a clients data shard and create a tfds object off it
    args:
        shard: a data, label constituting a client's data shard
        bs:batch size
    return:
        tfds object'''
    # seperate shard into data and labels lists
    data, label = zip(*data_shard)
    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
    return dataset.batch(bs)


def model(shape, classes):
    model = keras.Sequential()
    # model = tf.keras.models.Sequential([
    # tf.keras.layers.Conv2D(32, 3, padding='same', input_shape=(32, 32, 3), activation='relu'),
    # # tf.keras.layers.Conv2D(32, 3, activation='relu'),
    # tf.keras.layers.MaxPooling2D(),
    # tf.keras.layers.Dropout(0.25),

    # # tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    # # tf.keras.layers.Conv2D(64, 3, activation='relu'),
    # # tf.keras.layers.MaxPooling2D(),
    # # tf.keras.layers.Dropout(0.25),

    # tf.keras.layers.Flatten(),
    # tf.keras.layers.Dense(64, activation='relu'),
    # tf.keras.layers.Dropout(0.5),
    # tf.keras.layers.Dense(classes, activation='softmax'),
    # ])

    # model.add(
    #     Conv2D(10, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(28, 28, 1)))
    # # model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Flatten())
    # model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
    # model.add(Dense(classes, activation='softmax'))
    # model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(200, input_shape=(shape,)))
    model.add(Activation("relu"))
    # model.add(Dense(200))
    # model.add(Activation("relu"))
    model.add(Dense(classes))
    model.add(Activation("softmax"))
    return model


def test_model(X_test, Y_test, model, run, type):
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    # logits = model.predict(X_test, batch_size=100)
    logits = model.predict(X_test)
    loss = cce(Y_test, logits)
    acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(Y_test, axis=1))
    print(
        'run: {} related to:{} | global_acc: {:.3%} | global_test_loss: {}'.format(run, type,acc, loss))
    return acc, loss


def get_errors(name, round):
    # finding error
    errors = [];
    round = math.floor(round/Wait)
    errors = users_details.errors[round]
    return errors

# number of users
users = [5]
# number of runs
r_max = 100

training_epochs = 1
accuracies = {}

# apply our function
x_train, y_train, x_test, y_test = load()
# read data from .mat file and filling user details
zip_data_sorted = split_train_data(x_train, y_train)
mat = [];
k=5;

# filename = 'C:/Users/mm03263/OneDrive - University of Surrey/Desktop/data/results/path-planning/' + str(
#     k) + '/U_' + str(wt) + '.mat'
filename = 'C:/Users/mm03263/OneDrive - University of Surrey/Desktop/data/results/path-planning/moving/150/different_noise_levels/errors.mat'
mat = scipy.io.loadmat(filename)
users_details = UserInfos(k, mat['errors_op']);
# Initializing learning parameters
# create optimizer

lr = 0.1
loss = 'categorical_crossentropy'
metrics = ['accuracy']
optimizer = optimizer = SGD(lr=lr,
                decay=lr / T_max,
                momentum=0.9
                )


def sum_scaled_weights(scaled_weight_list, CWK, D_k):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''

    avg_grad = list()
    av_up: np.ndarray = []
    av_down = 0
    # get the average grad accross all client gradients
    for i, v in enumerate(scaled_weight_list):
        if CWK[i] != 0:
            av_down += CWK[i] * D_k[i]
            v_array = np.array(v, dtype=object)
            var = v_array * (CWK[i] * D_k[i])
            if len(av_up) == 0:
                av_up = var
            else:
                av_up = var + av_up
    avg_grad = np.multiply(av_up, 1 / av_down).tolist()
    return avg_grad


test_batched = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(len(y_test))

baseUrl = 'C:/Users/mm03263/OneDrive - University of Surrey/Desktop/data/results/path-planning/moving/150/different_noise_levels/non-iid/';

clients = create_clients_Niid(zip_data_sorted, k, k, shares, initial='clients')
for run in range(r_max):
    k = users[0];
    if path.exists(baseUrl + str(run)) == False:
        os.mkdir(baseUrl + str(run))
    accuracy1 = np.zeros(T_max)

    accuracy2 = np.zeros(T_max)

    accuracy3 = np.zeros(T_max)

    accuracy4 = np.zeros(T_max)

    global_model = model(784, 10)
    global_model1 = model(784, 10)
    global_model2 = model(784, 10)
    global_model3 = model(784, 10)
    global_model4 = model(784, 10)

    clients_batched = dict()

    bs_iterator = 0;
    for (client_name, data) in clients.items():
        bs = D_k[bs_iterator]

        clients_batched[client_name] = batch_data(data, bs)
        bs_iterator += 1

    global_model1.set_weights(global_model.get_weights())
    global_model2.set_weights(global_model.get_weights())
    global_model3.set_weights(global_model.get_weights())
    global_model4.set_weights(global_model.get_weights())

    for round in range(T_max):
        print('aggregation raound' + str(round) + 'started')
        errors = get_errors(k, round)
        print(errors)

        print('##########################')
        global_weights1 = global_model1.get_weights()
        global_weights2 = global_model2.get_weights()
        global_weights3 = global_model3.get_weights()
        global_weights4 = global_model4.get_weights()
        # initial list to collect local model weights after scalling
        scaled_local_weight_list1 = list()
        scaled_local_weight_list2 = list()
        scaled_local_weight_list3 = list()
        scaled_local_weight_list4 = list()
        # randomize client data - using keys
        client_names = list(clients_batched.keys())

        CW = [];
        for index, client in enumerate(client_names):
            error_client = errors[index]
            cw1 = bernoulli.rvs(1-error_client)
            CW.append(cw1)

            local_model1 = model(784, 10)
            local_model1.compile(loss=loss,
                                 optimizer=optimizer,
                                 metrics=metrics)
            local_model2 = model(784, 10)
            local_model2.compile(loss=loss,
                                 optimizer=optimizer,
                                 metrics=metrics)

            local_model3 = model(784, 10)
            local_model3.compile(loss=loss,
                                 optimizer=optimizer,
                                 metrics=metrics)
            local_model4 = model(784, 10)
            local_model4.compile(loss=loss,
                                 optimizer=optimizer,
                                 metrics=metrics)

            # fit local model with client's data
            # clients_batched[client]
            data_noisy1 = AddNoise(clients[client], PSNR1[index])
            data_noisy_batched1 = batch_data(data_noisy1, D_k[index])

            data_noisy2 = AddNoise(clients[client], PSNR2[index])
            data_noisy_batched2 = batch_data(data_noisy2, D_k[index])

            data_noisy3 = AddNoise(clients[client], PSNR3[index])
            data_noisy_batched3 = batch_data(data_noisy3, D_k[index])

            data_noisy4 = AddNoise(clients[client], PSNR4[index])
            data_noisy_batched4 = batch_data(data_noisy4, D_k[index])

            local_model2.set_weights(global_weights2)
            h2 = local_model2.fit(data_noisy_batched2, epochs=training_epochs, verbose=0)

            # set local model weight to the weight of the global model
            local_model1.set_weights(global_weights1)
            # fit local model with client's data
            h1 = local_model1.fit(data_noisy_batched1, epochs=training_epochs, verbose=0)

            # set local model weight to the weight of the global model
            local_model3.set_weights(global_weights3)
            # fit local model with client's data
            h3 = local_model3.fit(data_noisy_batched3, epochs=training_epochs, verbose=0)

            # set local model weight to the weight of the global model
            local_model4.set_weights(global_weights4)
            # fit local model with client's data
            h4 = local_model4.fit(data_noisy_batched4, epochs=training_epochs, verbose=0)

            scaled_local_weight_list1.append(local_model1.get_weights())
            scaled_local_weight_list2.append(local_model2.get_weights())
            scaled_local_weight_list3.append(local_model3.get_weights())
            scaled_local_weight_list4.append(local_model4.get_weights())
            # clear session to free memory after each communication round
            K.clear_session()
        print('aggregation raound' + str(round) + 'ended')
        all_zero = False

        temp = np.zeros(k)
        if (CW == temp).all():
            all_zero = True


        print(CW)

        if all_zero == False:
            average_weights1 = sum_scaled_weights(scaled_local_weight_list1, CW, D_k)
            global_model1.set_weights(average_weights1)

        if all_zero== False:
            average_weights2 = sum_scaled_weights(scaled_local_weight_list2, CW, D_k)
            global_model2.set_weights(average_weights2)

        if all_zero == False:
            average_weights3 = sum_scaled_weights(scaled_local_weight_list3, CW, D_k)
            global_model3.set_weights(average_weights3)

        if all_zero == False:
            average_weights4 = sum_scaled_weights(scaled_local_weight_list4, CW, D_k)
            global_model4.set_weights(average_weights4)

        print('############################################')
        for (x_test, y_test) in test_batched:
            global_acc_op, global_loss_op = test_model(x_test, y_test, global_model1, run, 'op1')
            accuracy1[round] = global_acc_op


        for (x_test, y_test) in test_batched:
            global_acc_wc, global_loss_wc = test_model(x_test, y_test, global_model2, run, 'op2')
            accuracy2[round] = global_acc_wc


        for (x_test, y_test) in test_batched:
            global_acc_op_comm, global_loss_op_comm = test_model(x_test, y_test, global_model3, run, 'op3')
            accuracy3[round] = global_acc_op_comm

        for (x_test, y_test) in test_batched:
            global_acc_op_comm, global_loss_op_comm = test_model(x_test, y_test, global_model4, run, 'op3')
            accuracy4[round] = global_acc_op_comm
    # out of round
    open(baseUrl + str(run) +'/-5.csv',
         'a').close()
    np.savetxt(
        fname=baseUrl + str(run)+ '/-5.csv',
        delimiter=",", X=accuracy1)

    open(baseUrl + str(run)+ '/0.csv',
         'a').close()
    np.savetxt(
        fname=baseUrl + str(run)+ '/0.csv',
        delimiter=",", X=accuracy2)


    open(baseUrl + str(run)+ '/5.csv',
         'a').close()
    np.savetxt(
        fname=baseUrl + str(run)+ '/5.csv',
        delimiter=",", X=accuracy3)

    open(baseUrl + str(run)+ '/30.csv',
         'a').close()
    np.savetxt(
        fname=baseUrl + str(run)+ '/30.csv',
        delimiter=",", X=accuracy4)