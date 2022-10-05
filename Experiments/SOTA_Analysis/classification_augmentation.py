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

# load eye detection data

# real data
real_0 = pd.read_csv('../../Dataset/EEG_Eye_State_ZeroOne_chop_5best_0.csv').iloc[:, :500]
real_1 = pd.read_csv('../../Dataset/EEG_Eye_State_ZeroOne_chop_5best_1.csv').iloc[:, :500]

# Fake data

# GroupGAN

groupgan_0 = np.load('generated_datasets/GroupGAN_EEG_Eye_State_ZeroOne_chop_5best_0.npy')
groupgan_1 = np.load('generated_datasets/GroupGAN_EEG_Eye_State_ZeroOne_chop_5best_1.npy')

# timeGAN

timegan_0 = np.load('generated_datasets/timeGAN_EEG_Eye_State_ZeroOne_chop_5best_0.npy')
timegan_1 = np.load('generated_datasets/timeGAN_EEG_Eye_State_ZeroOne_chop_5best_1.npy')


## FF

ff_0 = np.load('generated_datasets/FF_EEG_Eye_State_ZeroOne_chop_5best_0.npy')
ff_1 = np.load('generated_datasets/FF_EEG_Eye_State_ZeroOne_chop_5best_1.npy')


real_label_0 = np.zeros(real_0.shape[0])
real_label_1 = np.ones(real_1.shape[0])    

groupgan_label_0 = np.zeros(groupgan_0.shape[0])
groupgan_label_1 = np.ones(groupgan_1.shape[0])

timegan_label_0 = np.zeros(timegan_0.shape[0])
timegan_label_1 = np.ones(timegan_1.shape[0])

ff_label_0 = np.zeros(ff_0.shape[0])
ff_label_1 = np.ones(ff_1.shape[0])

target_real = np.hstack([real_label_0, real_label_1])
target_fake = {}

target_groupgan = np.hstack([groupgan_label_0, groupgan_label_1])
target_timegan = np.hstack([timegan_label_0, timegan_label_1])
target_ff = np.hstack([ff_label_0, ff_label_1])

target_fake['groupgan'] = target_groupgan
target_fake['timegan'] = target_timegan
target_fake['ff'] = target_ff

# real data initalization
data_real_0 = np.zeros((len(real_0), 100, 5))
data_real_1 = np.zeros((len(real_1), 100, 5))

# fake data initalization

# GroupGAN
data_groupgan_0 = np.zeros((len(groupgan_0), 100, 5))
data_groupgan_1 = np.zeros((len(groupgan_1), 100, 5))

# timeGAN
data_timegan_0 = np.zeros((len(timegan_0), 100, 5))
data_timegan_1 = np.zeros((len(timegan_1), 100, 5))


# FF
data_ff_0 = np.zeros((len(ff_0), 100, 5))
data_ff_1 = np.zeros((len(ff_1), 100, 5))


# real data preprocessing
for instance in real_0.itertuples():
    for i in range(5):
        data_real_0[instance[0], :, i] = instance[100*i+1:100*(i+1)+1]

for instance in real_1.itertuples():
    for i in range(5):
        data_real_1[instance[0], :, i] = instance[100*i+1:100*(i+1)+1]

data_real = np.vstack([data_real_0, data_real_1])

# fake data preprocessing
fake_data = {}


# GroupGAN
data_groupgan_0 = groupgan_0
data_groupgan_1 = groupgan_1
data_groupgan = np.vstack([data_groupgan_0, data_groupgan_1])
fake_data['groupgan'] = data_groupgan

# timeGAN
data_timegan_0 = timegan_0
data_timegan_1 = timegan_1
data_timegan = np.vstack([data_timegan_0, data_timegan_1])
fake_data['timegan'] = data_timegan

# FF
data_ff_0 = ff_0
data_ff_1 = ff_1
data_ff = np.vstack([data_ff_0, data_ff_1])
fake_data['ff'] = data_ff

final_scores = {}
adam = Adam(learning_rate=0.001)

fraction = 0.5
for iteration in trange(50):
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

    data_types = ['groupgan', 'timegan', 'ff']
    _, X_real_portion, _, y_real_portion = train_test_split(non_test_X, non_test_y, test_size=fraction, random_state=42, stratify=non_test_y)

    for fake_type in data_types:
        # fake_fraction = ((len(X_real_portion)*i)/(len(fake_data[fake_type])))
        X_fake_portion, y_fake_portion = fake_data[fake_type], target_fake[fake_type]
        # _, X_fake_portion, _, y_fake_portion = train_test_split(fake_data[fake_type], target_fake[fake_type], test_size=1, random_state=42, stratify=target_fake[fake_type])
        X_train[fake_type], X_test[fake_type], y_train[fake_type], y_test[fake_type] = train_test_split(np.vstack([X_real_portion, X_fake_portion]), np.hstack([y_real_portion, y_fake_portion]), test_size=0.2, random_state=42, stratify=np.hstack([y_real_portion, y_fake_portion]))
        X_train[fake_type], X_val[fake_type], y_train[fake_type], y_val[fake_type] = train_test_split(X_train[fake_type], y_train[fake_type], test_size=0.2, random_state=42, stratify=y_train[fake_type])

    _, real_X_portion, _, real_y_portion = train_test_split(non_test_X, non_test_y, test_size=(fraction)*0.8, random_state=42, stratify=non_test_y)
    X_train['real'], X_val['real'], y_train['real'], y_val['real'] = train_test_split(real_X_portion, real_y_portion, test_size=0.2, random_state=42, stratify=real_y_portion)
    data_types.append('real')

    models = {}
    for data_type in data_types:
        models[data_type] = Sequential()
        models[data_type].add(LSTM(256, input_shape=(100, 5)))
        models[data_type].add(Dense(1, activation='sigmoid'))

    chk = {}
    for data_type in data_types:
        chk[data_type] = ModelCheckpoint(f'augmentation_best_model_{data_type}_{int(fraction*100)}_{i}_{iteration}_{5}.pkl', monitor='val_accuracy', save_best_only=True, mode='max', verbose=0)

    for data_type in data_types:
        models[data_type].compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
        models[data_type].fit(X_train[data_type], y_train[data_type], batch_size=128, epochs=200, validation_data=(X_val[data_type], y_val[data_type]), callbacks=[chk[data_type]], verbose=0)

    scores = {}
    for data_type in data_types:

        #loading the model and checking accuracy on the test data
        model = load_model(f'augmentation_best_model_{data_type}_{int(fraction*100)}_{i}_{iteration}_{5}.pkl')
        test_preds_real = model.predict(X_test['real'])

        preds_real = np.zeros(test_preds_real.shape)
        preds_real[test_preds_real > 0.5] = 1
        scores[data_type] = accuracy_score(y_test['real'], preds_real)

    final_scores[iteration][i] = scores

    # print and save the final scores
    print(final_scores)
    with open(f'augmentation_final_scores_{fraction}_{5}.pkl', 'wb') as f:
        pickle.dump(final_scores, f)