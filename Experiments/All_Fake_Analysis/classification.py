import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from tensorflow.keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from tqdm import trange, tqdm

import pickle

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--Ngroups', type=int, default=1)

n_groups = parser.parse_args().Ngroups

# load eye detection data

# real data
real_0 = pd.read_csv('../../Dataset/EEG_Eye_State_ZeroOne_chop_5best_0.csv').iloc[:, :n_groups*100]
real_1 = pd.read_csv('../../Dataset/EEG_Eye_State_ZeroOne_chop_5best_1.csv').iloc[:, :n_groups*100]

real_0['label'] = 0
real_1['label'] = 1

# Fake data

# baseline

baseline_0 = pd.read_csv(f'Fake_Data/generated_samples_baseline_0_{n_groups}.csv')
baseline_1 = pd.read_csv(f'Fake_Data/generated_samples_baseline_1_{n_groups}.csv')

baseline_0['label'] = 0
baseline_1['label'] = 1

# LSTM MLP with CD

LSTM_MLP_with_CD_0 = pd.read_csv(f'Fake_Data/generated_samples_LSTM_MLP_with_CD_0_{n_groups}.csv')
LSTM_MLP_with_CD_1 = pd.read_csv(f'Fake_Data/generated_samples_LSTM_MLP_with_CD_1_{n_groups}.csv')

LSTM_MLP_with_CD_0['label'] = 0
LSTM_MLP_with_CD_1['label'] = 1

# LSTM without CD

LSTM_without_CD_0 = pd.read_csv(f'Fake_Data/generated_samples_LSTM_without_CD_0_{n_groups}.csv')
LSTM_without_CD_1 = pd.read_csv(f'Fake_Data/generated_samples_LSTM_without_CD_1_{n_groups}.csv')

LSTM_without_CD_0['label'] = 0
LSTM_without_CD_1['label'] = 1

target_real = np.hstack([np.array(real_0['label']), np.array(real_1['label'])])
target_fake = {}

target_baseline = np.hstack([np.array(baseline_0['label']), np.array(baseline_1['label'])])
target_LSTM_MLP_with_CD = np.hstack([np.array(LSTM_MLP_with_CD_0['label']), np.array(LSTM_MLP_with_CD_1['label'])])
target_LSTM_without_CD = np.hstack([np.array(LSTM_without_CD_0['label']), np.array(LSTM_without_CD_1['label'])])

target_fake['baseline'] = target_baseline
target_fake['LSTM_MLP_with_CD'] = target_LSTM_MLP_with_CD
target_fake['LSTM_without_CD'] = target_LSTM_without_CD


# real data initalization
data_real_0 = np.zeros((len(real_0), 100, n_groups))
data_real_1 = np.zeros((len(real_1), 100, n_groups))

# fake data initalization

# baseline
data_baseline_0 = np.zeros((len(baseline_0), 100, n_groups))
data_baseline_1 = np.zeros((len(baseline_1), 100, n_groups))

# LSTM MLP with CD
data_LSTM_MLP_with_CD_0 = np.zeros((len(LSTM_MLP_with_CD_0), 100, n_groups))
data_LSTM_MLP_with_CD_1 = np.zeros((len(LSTM_MLP_with_CD_1), 100, n_groups))

# LSTM without CD
data_LSTM_without_CD_0 = np.zeros((len(LSTM_without_CD_0), 100, n_groups))
data_LSTM_without_CD_1 = np.zeros((len(LSTM_without_CD_1), 100, n_groups))


# real data preprocessing
for instance in real_0.itertuples():
    for i in range(n_groups):
        data_real_0[instance[0], :, i] = instance[100*i+1:100*(i+1)+1]

for instance in real_1.itertuples():
    for i in range(n_groups):
        data_real_1[instance[0], :, i] = instance[100*i+1:100*(i+1)+1]

data_real = np.vstack([data_real_0, data_real_1])

# fake data preprocessing
fake_data = {}

# baseline
for instance in baseline_0.itertuples():
    for i in range(n_groups):
        data_baseline_0[instance[0], :, i] = instance[100*i+1:100*(i+1)+1]

for instance in baseline_1.itertuples():
    for i in range(n_groups):
        data_baseline_1[instance[0], :, i] = instance[100*i+1:100*(i+1)+1]

data_baseline = np.vstack([data_baseline_0, data_baseline_1])
fake_data['baseline'] = data_baseline

# LSTM MLP with CD
for instance in LSTM_MLP_with_CD_0.itertuples():
    for i in range(n_groups):
        data_LSTM_MLP_with_CD_0[instance[0], :, i] = instance[100*i+1:100*(i+1)+1]

for instance in LSTM_MLP_with_CD_1.itertuples():
    for i in range(n_groups):
        data_LSTM_MLP_with_CD_1[instance[0], :, i] = instance[100*i+1:100*(i+1)+1]

data_LSTM_MLP_with_CD = np.vstack([data_LSTM_MLP_with_CD_0, data_LSTM_MLP_with_CD_1])
fake_data['LSTM_MLP_with_CD'] = data_LSTM_MLP_with_CD

# LSTM without CD
for instance in LSTM_without_CD_0.itertuples():
    for i in range(n_groups):
        data_LSTM_without_CD_0[instance[0], :, i] = instance[100*i+1:100*(i+1)+1]
for instance in LSTM_without_CD_1.itertuples():
    for i in range(n_groups):
        data_LSTM_without_CD_1[instance[0], :, i] = instance[100*i+1:100*(i+1)+1]

data_LSTM_without_CD = np.vstack([data_LSTM_without_CD_0, data_LSTM_without_CD_1])
fake_data['LSTM_without_CD'] = data_LSTM_without_CD

final_scores = {}
adam = Adam(learning_rate=0.001)

for iteration in trange(5):
    final_scores[iteration] = {}
    # real data
    X_train = {}
    X_val = {}
    X_test = {}
    y_train = {}
    y_val= {}
    y_test = {}

    X_train['real'], X_test['real'], y_train['real'], y_test['real'] = train_test_split(data_real, target_real, test_size=0.2, random_state=42, stratify=target_real)
    non_test_X, non_test_y = X_train['real'], y_train['real']
    X_train['real'], X_val['real'], y_train['real'], y_val['real'] = train_test_split(X_train['real'], y_train['real'], test_size=0.2, random_state=42, stratify=y_train['real'])

    for i in tqdm([1, 2, 4, 6, 8, 10]):
        data_types = ['baseline', 'LSTM_MLP_with_CD', 'LSTM_without_CD']
        
        for fake_type in data_types:
            fake_fraction = ((len(data_real)*i)/(len(fake_data[fake_type])))
            _, X_fake_portion, _, y_fake_portion = train_test_split(fake_data[fake_type], target_fake[fake_type], test_size=fake_fraction, random_state=42, stratify=target_fake[fake_type])
            X_train[fake_type], X_test[fake_type], y_train[fake_type], y_test[fake_type] = train_test_split(X_fake_portion,y_fake_portion, test_size=0.2, random_state=42, stratify=y_fake_portion)
            X_train[fake_type], X_val[fake_type], y_train[fake_type], y_val[fake_type] = train_test_split(X_train[fake_type], y_train[fake_type], test_size=0.2, random_state=42, stratify=y_train[fake_type])

        if i == 1:
            _, real_X_portion, _, real_y_portion = train_test_split(non_test_X, non_test_y, test_size=(10/100)*0.8, random_state=42, stratify=non_test_y)
            X_train['real'], X_val['real'], y_train['real'], y_val['real'] = train_test_split(real_X_portion, real_y_portion, test_size=0.2, random_state=42, stratify=real_y_portion)
            data_types.append('real')

        models = {}
        for data_type in data_types:
            models[data_type] = Sequential()
            models[data_type].add(LSTM(256, input_shape=(100, n_groups)))
            models[data_type].add(Dense(1, activation='sigmoid'))
        
       

        chk = {}
        for data_type in data_types:
            chk[data_type] = ModelCheckpoint(f'best_model_{data_type}_{i}_{iteration}_{n_groups}.pkl', monitor='val_accuracy', save_best_only=True, mode='max', verbose=0)

        for data_type in data_types:
            models[data_type].compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
            models[data_type].fit(X_train[data_type], y_train[data_type], batch_size=128, epochs=200, validation_data=(X_val[data_type], y_val[data_type]), callbacks=[chk[data_type]], verbose=0)

        scores = {}
        for data_type in data_types:

            #loading the model and checking accuracy on the test data
            model = load_model(f'best_model_{data_type}_{i}_{iteration}_{n_groups}.pkl')
            test_preds_real = model.predict(X_test['real'])

            preds_real = np.zeros(test_preds_real.shape)
            preds_real[test_preds_real > 0.5] = 1
            scores[data_type] = accuracy_score(y_test['real'], preds_real)

        final_scores[iteration][i] = scores

        # print and save the final scores
        print(final_scores)
        with open(f'final_scores_{n_groups}.pkl', 'wb') as f:
            pickle.dump(final_scores, f)
