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
from tensorflow_examples.models.pix2pix import pix2pix
import os.path
from os import path
import tensorflow_datasets as tfds
from IPython.display import clear_output
import matplotlib.pyplot as plt
import pandas as pd

# defining user infos
class UserInfos:
    # The class "constructor" - It's actually an initializer
    def __init__(self, name, errors_wcs, errors_ops, errors_op_comms):
        self.name = name
        self.errors_wcs = errors_wcs
        self.errors_ops = errors_ops
        self.errors_op_comms = errors_op_comms


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask


def load_image(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))
    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


PSNR = [30, 5, 5, 5, 5]
T_max = 150;
wt = 1;
Wait = T_max / wt;
D_k = [1200 ,330,330, 870, 870];
OUTPUT_CLASSES = 3

dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 128
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
train_images = dataset['train'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
test_images = dataset['test'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
species = []
zero = 0
one = 0
for item in dataset['train']:
    species.append((item['species']).numpy().tolist())

class Augment(tf.keras.layers.Layer):
  def __init__(self, seed=42):
    super().__init__()
    # both use the same seed, so they'll make the same random changes.
    self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
    self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)

  def call(self, inputs, labels):
    inputs = self.augment_inputs(inputs)
    labels = self.augment_labels(labels)
    return inputs, labels


# train_batches = (
#     train_images
#     .cache()
#     .shuffle(BUFFER_SIZE)
#     .batch(BATCH_SIZE)
#     .repeat()
#     .map(Augment())
#     .prefetch(buffer_size=tf.data.AUTOTUNE))

test_batches = test_images.batch(BATCH_SIZE)


def AddNoise(data, psnr, data_amount):

    var_noise = (1) / (10 ** (psnr / 10))
    l_i = []
    l_m = []
    for index_data in range(data_amount):
        x_n = data[0][index_data] + tf.random.normal(mean=0.0, stddev=var_noise, shape=np.array(data[0][index_data]).shape,dtype=tf.dtypes.float32);
        l_i.append(x_n)
        l_m.append(data[1][index_data])
    temp = tf.data.Dataset.from_tensor_slices((l_i, l_m))
    noisyData = (
        temp
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE)
            .repeat()
            .map(Augment())
            .prefetch(buffer_size=tf.data.AUTOTUNE))
    return noisyData


# functions


def create_clients(train_images, num_clients, species, initial='clients'):
    # create a list of client names
    client_names = ['{}_{}'.format(initial, i + 1) for i in range(num_clients)]
    shards = [[[],[]],[[],[]],[[],[]],[[],[]],[[],[]],[[],[]]]
    index = 1
    index_species = 0
    for images, masks in train_images:
        if index<num_clients or len(shards[0][0]) != D_k[0] :
            if species[index_species] == 0:
                if (len(shards[0][0]) < D_k[0]):
                    shards[0][0].append(images)
                    shards[0][1].append(masks)
            if species[index_species] == 1 and index<num_clients:
                if (len(shards[index][0]) < D_k[index]):
                    shards[index][0].append(images)
                    shards[index][1].append(masks)
                else:
                    index += 1
        else:
            break
        index_species += 1
    # for images, masks in train_images:
    #     if start<stop:
    #         start+=1
    #         if index == 0
    #             shards[index][0].append(images)
    #             shards[index][1].append(masks)
    #     elif index < 4:
    #         start = 0
    #         index += 1
    #         stop = D_k[index]
    #     else:
    #         break
    # x = tf.data.Dataset.from_tensor_slices((shards[0][0], shards[0][1]))
    # number of clients must equal number of shards
    # assert (len(shards) == len(client_names))
    return {client_names[i]: shards[i] for i in range(len(client_names))}


def model(output_channels:int):
  inputs = tf.keras.layers.Input(shape=[128, 128, 3])

  # Downsampling through the model
  skips = down_stack(inputs)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  # This is the last layer of the model
  last = tf.keras.layers.Conv2DTranspose(
      filters=output_channels, kernel_size=3, strides=2,
      padding='same')  #64x64 -> 128x128

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

def create_mask(pred_mask):
  pred_mask = tf.math.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

def test_model(X_test, Y_test, model, run, type):
    cce  = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                               reduction=tf.keras.losses.Reduction.NONE)
    logits = model.predict(X_test)
    predicted_mask = create_mask(logits)
    # loss = cce(Y_test, logits)
    acc = accuracy_score(tf.argmax(predicted_mask, axis=1), tf.argmax(Y_test[0], axis=1))

    return acc


def get_errors_wc(name, round):
    # finding error
    errors_wc = [];
    round = math.floor(round / Wait)
    errors_wc = users_details.errors_wcs[round]
    return errors_wc


def get_errors_op(name, round):
    # finding error
    errors_op = [];
    round = math.floor(round / Wait)
    errors_op = users_details.errors_ops[round]
    return errors_op


def get_errors_op_comm(name, round):
    # finding error
    errors_op_comm = [];
    round = math.floor(round / Wait)
    errors_op_comm = users_details.errors_op_comms[round]
    return errors_op_comm


def get_DK(name):
    # finding error
    D_K = [];
    D_K = users_details.D_ks
    return D_K


# number of users
users = [5]
# number of runs
r_max = 100

training_epochs = 1
accuracies = {}


mat = [];
users_details = [];
for i in range(len(users)):
    k = users[i]
    u = UserInfos(k, [], [], [])
    # filename = 'C:/Users/mm03263/OneDrive - University of Surrey/Desktop/data/results/path-planning/' + str(
    #     k) + '/U_' + str(wt) + '.mat'
    filename = 'C:/Users/mm03263/OneDrive - University of Surrey/Desktop/data/results/new-dataset/errors.mat'
    mat = scipy.io.loadmat(filename)
    error_wc = mat['errors_wc']
    error_op = mat['errors_op']
    error_op_comm = mat['errors_op_comm']
    users_details = UserInfos(k, error_wc, error_op, error_op_comm);
# Initializing learning parameters
# create optimizer

lr = 0.001
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = ['sparse_categorical_accuracy']
# optimizer  = SGD(lr=lr,
#                             decay=lr / T_max,
#                             momentum=0.9
#                             )
optimizer = 'adam'

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




baseUrl = 'C:/Users/mm03263/OneDrive - University of Surrey/Desktop/data/results/new-dataset/iid/'+str(k)+'/';

clients = create_clients((train_images), k, species,  initial='clients')
base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

down_stack.trainable = False
up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

for run in range(r_max):
    k = users[0];
    if path.exists(baseUrl + str(k) + '_op') == False:
        os.mkdir(baseUrl + str(k) + '_op')
    if path.exists(baseUrl + str(k) + '_wc') == False:
        os.mkdir(baseUrl + str(k) + '_wc')
    if path.exists(baseUrl + str(k) + '_op_comm') == False:
        os.mkdir(baseUrl + str(k) + '_op_comm')

    if path.exists(baseUrl + str(k) + '_op' + '/' + str(run)) == False:
        os.mkdir(baseUrl + str(k) + '_op' + '/' + str(run))
    if path.exists(baseUrl + str(k) + '_wc' + '/' + str(run)) == False:
        os.mkdir(baseUrl + str(k) + '_wc' + '/' + str(run))
    if path.exists(baseUrl + str(k) + '_op_comm' + '/' + str(run)) == False:
        os.mkdir(baseUrl + str(k) + '_op_comm' + '/' + str(run))

    accuracy_op = np.zeros(T_max)
    loss_op = np.zeros(T_max)

    accuracy_wc = np.zeros(T_max)
    loss_wc = np.zeros(T_max)

    accuracy_op_comm = np.zeros(T_max)
    loss_op_comm = np.zeros(T_max)

    global_model = model(output_channels=OUTPUT_CLASSES)
    global_model1 = model(output_channels=OUTPUT_CLASSES)
    global_model2 = model(output_channels=OUTPUT_CLASSES)
    global_model3 = model(output_channels=OUTPUT_CLASSES)
    global_model4 = model(output_channels=OUTPUT_CLASSES)

    global_model1.compile(loss=loss,
                                 optimizer=optimizer,
                                 metrics=metrics)
    global_model2.compile(loss=loss,
                          optimizer=optimizer,
                          metrics=metrics)
    global_model3.compile(loss=loss,
                          optimizer=optimizer,
                          metrics=metrics)
    global_model4.compile(loss=loss,
                          optimizer=optimizer,
                          metrics=metrics)
    clients_batched = dict()


    global_model1.set_weights(global_model.get_weights())
    global_model2.set_weights(global_model.get_weights())
    global_model3.set_weights(global_model.get_weights())
    global_model4.set_weights(global_model.get_weights())
    open(baseUrl + str(k) + '_op' + '/' + str(run) + '/' + 'acc_op' + '.csv',
         'a').close()
    open(baseUrl + str(k) + '_wc' + '/' + str(run) + '/' + 'acc_wc' + '.csv',
         'a').close()
    open(baseUrl + str(k) + '_op_comm' + '/' + str(run) + '/' + 'acc_op_comm' + '.csv',
         'a').close()
    for round in range(T_max):
        print('aggregation raound' + str(round) + 'started')
        error_op = get_errors_op(k, round)
        error_op_comm = get_errors_op_comm(k, round)
        error_wc = get_errors_wc(k, round)

        print(error_op)
        print(error_wc)
        print(error_op_comm)
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
        client_names = ['clients_1','clients_2','clients_3','clients_4','clients_5']

        CW_op = [];
        CW_wc = [];
        CW_op_comm = [];
        for index, client in enumerate(client_names):
            error_client_op = error_op[index]
            cw1 = bernoulli.rvs(1 - error_client_op)
            CW_op.append(cw1)

            error_client_wc = error_wc[index]
            cw2 = bernoulli.rvs(1 - error_client_wc)
            CW_wc.append(cw2)

            error_client_op_comm = error_op_comm[index]
            cw3 = bernoulli.rvs(1 - error_client_op_comm)
            CW_op_comm.append(cw3)

            local_model1 = model(output_channels=OUTPUT_CLASSES)
            local_model1.compile(loss=loss,
                                 optimizer=optimizer,
                                 metrics=metrics)
            local_model2 = model(output_channels=OUTPUT_CLASSES)
            local_model2.compile(loss=loss,
                                 optimizer=optimizer,
                                 metrics=metrics)

            local_model3 = model(output_channels=OUTPUT_CLASSES)
            local_model3.compile(loss=loss,
                                 optimizer=optimizer,
                                 metrics=metrics)
            local_model4 = model(output_channels=OUTPUT_CLASSES)
            local_model4.compile(loss=loss,
                                 optimizer=optimizer,
                                 metrics=metrics)

            local_model2.set_weights(global_weights2)
            # fit local model with client's data
            # clients_batched[client]

            data_noisy = AddNoise(clients[client], PSNR[index], len(clients[client][0]))


            h2 = local_model2.fit(data_noisy, epochs=training_epochs,
                          steps_per_epoch=D_k[index]//D_k[index])

            # set local model weight to the weight of the global model
            local_model1.set_weights(global_weights1)
            # fit local model with client's data
            h1 = local_model1.fit(data_noisy, epochs=training_epochs,
                          steps_per_epoch=D_k[index]//D_k[index])

            # set local model weight to the weight of the global model
            local_model3.set_weights(global_weights3)
            # fit local model with client's data
            h3 = local_model3.fit(data_noisy, epochs=training_epochs,
                          steps_per_epoch=D_k[index]//D_k[index])

            # set local model weight to the weight of the global model
            local_model4.set_weights(global_weights4)
            # fit local model with client's data
            h4 = local_model4.fit(data_noisy, epochs=training_epochs,
                          steps_per_epoch=D_k[index]//D_k[index])

            scaled_local_weight_list1.append(local_model1.get_weights())
            scaled_local_weight_list2.append(local_model2.get_weights())
            scaled_local_weight_list3.append(local_model3.get_weights())
            scaled_local_weight_list4.append(local_model4.get_weights())

            # clear session to free memory after each communication round
            K.clear_session()
        print('aggregation raound' + str(round) + 'ended')
        all_zero_op = False
        all_zero_wc = False
        all_zero_c = False
        all_zero_op_comm = False
        temp = np.zeros(k)
        if (CW_op == temp).all():
            all_zero_op = True
        if (CW_wc == temp).all():
            all_zero_wc = True

        if (CW_op_comm == temp).all():
            all_zero_op_comm = True

        print(CW_op)
        print(CW_wc)
        print(CW_op_comm)

        if all_zero_op == False:
            average_weights1 = sum_scaled_weights(scaled_local_weight_list1, CW_op, D_k)
            global_model1.set_weights(average_weights1)

        if all_zero_wc == False:
            average_weights2 = sum_scaled_weights(scaled_local_weight_list2, CW_wc, D_k)
            global_model2.set_weights(average_weights2)

        if all_zero_op_comm == False:
            average_weights4 = sum_scaled_weights(scaled_local_weight_list4, CW_op_comm, D_k)
            global_model4.set_weights(average_weights4)

        print('############################################')
        # test_model(test_batches, global_model1, 0, 'op')

        global_loss_op, global_acc_op = global_model1.evaluate(test_batches)
        global_loss_wc, global_acc_wc = global_model2.evaluate(test_batches)
        global_loss_op_comm, global_acc_op_comm = global_model4.evaluate(test_batches)

        accuracy_op[round] = global_acc_op
        accuracy_wc[round] = global_acc_wc
        accuracy_op_comm[round] = global_acc_op_comm
        # total = 0
        # for (x_test, y_test) in test_batches:
        #     total+=1
        #     global_acc_op = test_model(x_test, y_test, global_model1, run, 'op')
        #     accuracy_op[round]+= global_acc_op
        #
        #     global_acc_wc = test_model(x_test, y_test, global_model2, run, 'wc')
        #     accuracy_wc[round] += global_acc_wc
        #
        #     global_acc_op_comm = test_model(x_test, y_test, global_model4, run, 'op_comm')
        #     accuracy_op_comm[round] += global_acc_op_comm
        #
        # accuracy_op[round] = accuracy_op[round]/total
        # accuracy_wc[round] = accuracy_wc[round]/total
        # accuracy_op_comm[round] = accuracy_op_comm[round]/total
        #
        print('run: {} round: {} related to:{} | global_acc: {:.3%} '.format(run, round, 'op', accuracy_op[round]))
        print('run: {} round: {} related to:{} | global_acc: {:.3%} '.format(run, round, 'wc', accuracy_wc[round]))
        print('run: {} round: {} related to:{} | global_acc: {:.3%} '.format(run, round, 'op_comm', accuracy_op_comm[round]))
        with open(baseUrl + str(k) + '_op' + '/' + str(run) + '/' + 'acc_op' + '.csv', "a") as f:
            f.write(str(accuracy_op[round]))
            f.write("\n")
        with open(baseUrl + str(k) + '_wc' + '/' + str(run) + '/' + 'acc_wc' + '.csv', "a") as f:
            f.write(str(accuracy_wc[round]))
            f.write("\n")
        with open(baseUrl + str(k) + '_op_comm' + '/' + str(run) + '/' + 'acc_op_comm'+ '.csv', "a") as f:
            f.write(str(accuracy_op_comm[round]))
            f.write("\n")
    # out of round


