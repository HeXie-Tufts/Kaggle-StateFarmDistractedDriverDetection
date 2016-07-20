# He Xie, Tufts University, July 2016
# Data reading and saving methods, modified from ZFTurbo's keras sample kernel on kaggle.com
# https://www.kaggle.com/zfturbo/state-farm-distracted-driver-detection/keras-sample/run/202460
# Neural Network model structure, modified from Keras example, simple deep CNN on the CIFAR10 small images dataset

import os
import glob
import cv2
import math
import pickle
import datetime
import pandas as pd
from random import shuffle
# from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.regularizers import l2
import numpy as np
np.random.seed(2016)


batch_size = 32
nb_classes = 10
nb_epoch = 40
data_augmentation = True
# The percentage of training data leave out as validation set
test_size = 0.2

# input image dimensions
img_rows, img_cols = 48, 64
# input image is grayscale. If RGB, img_channels = 3
img_channels = 1

# Whenever cache data is available, use cache data instead of read image files again
use_cache = 1

def get_im_cv2(path, img_rows, img_cols, img_channels=1):
    # Read image and rezide to img_rows * img_cols
    if img_channels == 1:
        img = cv2.imread(path, 0)
    elif img_channels == 3:
        img = cv2.imread(path)
    # Reduce size
    resized = cv2.resize(img, (img_cols, img_rows))
    return resized

def get_driver_data():
    dr = dict()
    path = os.path.join("..", "Data", "driver_imgs_list.csv")
    print("Read drivers data")
    f = open(path, "r")
    line = f.readline()
    while (1):
        line = f.readline()
        if line == "":
            break
        arr = line.strip().split(",")
        dr[arr[2]] = arr[0]
    f.close()
    return dr

def load_train(img_rows, img_cols, img_channels = 1):
    X_train = []
    y_train = []
    driver_id = []

    driver_data = get_driver_data()

    print("Read train images")
    for j in range(10):
        print("Load folder c{}".format(j))
        path = os.path.join("..", "Data", "imgs", "train", "c" + str(j), "*.jpg")
        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            img = get_im_cv2(fl, img_rows, img_cols, img_channels)
            X_train.append(img)
            y_train.append(j)
            driver_id.append(driver_data[flbase])

    unique_drivers = sorted(list(set(driver_id)))
    print("Unique drivers: {}".format(len(unique_drivers)))
    print(unique_drivers)
    return X_train, y_train, driver_id, unique_drivers

def load_test(img_rows, img_cols, img_channels=1):
    print('Read test images')
    path = os.path.join('..', 'Data', "imgs", 'test', '*.jpg')
    files = glob.glob(path)
    X_test = []
    X_test_id = []
    total = 0
    thr = math.floor(len(files)/10)
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im_cv2(fl, img_rows, img_cols, img_channels)
        X_test.append(img)
        X_test_id.append(flbase)
        total += 1
        if total%thr == 0:
            print('Read {} images from {}'.format(total, len(files)))

    return X_test, X_test_id

def cache_data(data, path):
    if os.path.isdir(os.path.dirname(path)):
        file = open(path, "wb")
        pickle.dump(data, file)
        file.close()
    else:
        print("Directory doesn't exists")

def restore_data(path):
    data = dict()
    if os.path.isfile(path):
        file = open(path, 'rb')
        data = pickle.load(file)
    return data

def save_model(model, note):
    json_string = model.to_json()
    if not os.path.isdir("cache"):
        os.mkdir("cache")
    open(os.path.join('cache', note + 'architecture.json'), 'w').write(json_string)
    model.save_weights(os.path.join('cache', note + 'model_weights.h5'), overwrite=True)

def read_model(note):
    model = model_from_json(open(os.path.join('cache', note + 'architecture.json')).read())
    model.load_weights(os.path.join('cache', note + 'model_weights.h5'))
    return model

def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    if not os.path.isdir('subm'):
        os.mkdir('subm')
    suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join('subm', 'submission_' + suffix + '.csv')
    result1.to_csv(sub_file, index=False)

def read_and_normalize_train_data(img_rows, img_cols, img_channels=1):
    cache_path = os.path.join('cache', 'train_r_' + str(img_rows) + '_c_' + str(img_cols) + '_t_' + str(img_channels) + '.dat')
    if not os.path.isfile(cache_path) or use_cache == 0:
        train_data, train_target, driver_id, unique_drivers = load_train(img_rows, img_cols, img_channels)
        cache_data((train_data, train_target, driver_id, unique_drivers), cache_path)
    else:
        print('Restore train from cache!')
        (train_data, train_target, driver_id, unique_drivers) = restore_data(cache_path)

    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)
    train_data = train_data.reshape(train_data.shape[0], img_channels, img_rows, img_cols)
    train_target = np_utils.to_categorical(train_target, 10)
    train_data = train_data.astype('float32')
    train_data /= 255
    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target, driver_id, unique_drivers

def read_and_normalize_test_data(img_rows, img_cols, img_channels=1):
    cache_path = os.path.join('cache', 'test_r_' + str(img_rows) + '_c_' + str(img_cols) + '_t_' + str(img_channels) + '.dat')
    if not os.path.isfile(cache_path) or use_cache == 0:
        test_data, test_id = load_test(img_rows, img_cols, img_channels)
        cache_data((test_data, test_id), cache_path)
    else:
        print('Restore test from cache!')
        (test_data, test_id) = restore_data(cache_path)

    test_data = np.array(test_data, dtype=np.uint8)
    test_data = test_data.reshape(test_data.shape[0], img_channels, img_rows, img_cols)
    test_data = test_data.astype('float32')
    test_data /= 255
    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    return test_data, test_id

def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()

# modified method, select validation set by selecting images of certain drivers,  He Xie
def split_test_drivers(train_data, train_target, driver_id, unique_drivers, test_size):
    # test_size is a float between 0 and 1
    X_train, y_train, X_test, y_test = [], [], [], []
    nb_test_drivers = int(math.floor(len(unique_drivers) * test_size))
    shuffle(unique_drivers)
    test_drivers = unique_drivers[:nb_test_drivers]

    for i in range(len(driver_id)):
        if driver_id[i] in test_drivers:
            X_test.append(train_data[i])
            y_test.append(train_target[i])
        else:
            X_train.append(train_data[i])
            y_train.append(train_target[i])
    X_train =np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float64)
    X_test =np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float64)
    return X_train, y_train, X_test, y_test

def create_model_v5(img_channels, img_rows, img_cols, nb_classes, wr=0.001):
    # model structure from Keras cifar10 small image classification example
    # added weight regularization
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=(img_channels, img_rows, img_cols), W_regularizer=l2(wr)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3, W_regularizer=l2(wr)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='same', W_regularizer=l2(wr)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, W_regularizer=l2(wr)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, W_regularizer=l2(wr)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
    return model

# Data already normalized and y data turn into 10 length one hot vectors
# Load training data and split into train set and validation set
train_data, train_target, driver_id, unique_drivers = read_and_normalize_train_data(img_rows, img_cols, img_channels)
X_train, y_train, X_test, y_test = split_test_drivers(train_data, train_target, driver_id, unique_drivers, test_size)
# Create model
model = create_model_v5(img_channels, img_rows, img_cols, nb_classes, wr = 0.001)

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(X_train, y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, y_test),
              shuffle=True, verbose = 2)
else:
    print('Using real-time data augmentation.')

    # this will do preprocessing and realtime data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images, changed to False
        vertical_flip=False)  # randomly flip images

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)

    # fit the model on the batches generated by datagen.flow()
    his = model.fit_generator(datagen.flow(X_train, y_train,
                        batch_size=batch_size),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=nb_epoch,
                        validation_data=(X_test, y_test), verbose = 1)

# Generate prediction
test_data, test_id = read_and_normalize_test_data(img_rows, img_cols, img_channels)
yfull_test = []
test_prediction = model.predict(test_data, batch_size=128, verbose=1)
yfull_test.append(test_prediction)

info_string = "cifar_" +'loss_' + "crossEntropy" \
                     + '_r_' + str(img_rows) \
                     + '_c_' + str(img_cols) \
                     + '_c_'  + str(img_channels)\
                     + '_ep_' + str(nb_epoch)

test_res = merge_several_folds_mean(yfull_test, 1)
create_submission(test_res, test_id, info_string)
