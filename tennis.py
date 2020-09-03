# This is the script version of the jupyter notebook in this repo.

import sys
from utils.create_features_utils import *
import pandas as pd
import numpy as np
from keras import models, layers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.text import Tokenizer

# NOTE: need to run the following (in bash) prior to running this code, so that scikit-lean will run
#    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras import metrics
from keras.models import load_model
#import seaborn as sns
#import matplotlib.pyplot as plt
#sns.set_style("darkgrid")

# Read match data with features
# The data has been prepped with: create_features.py
df = pd.read_csv('data/wimbledon_matches_with_feature.csv')
df = df.dropna()
df['diff_rank'] = df['player_0_rank'] - df['player_1_rank']

features_list = [
 'diff_rank',
 'diff_match_win_percent',
 'diff_games_win_percent',
 'diff_5_set_match_win_percent',
 'diff_close_sets_percent',
 'diff_match_win_percent_grass',
 'diff_games_win_percent_grass',
 'diff_5_set_match_win_percent_grass',
 'diff_close_sets_percent_grass',
 'diff_match_win_percent_52',
 'diff_games_win_percent_52',
 'diff_5_set_match_win_percent_52',
 'diff_close_sets_percent_52',
 'diff_match_win_percent_grass_60',
 'diff_games_win_percent_grass_60',
 'diff_5_set_match_win_percent_grass_60',
 'diff_close_sets_percent_grass_60',
 'diff_match_win_percent_hh',
 'diff_games_win_percent_hh',
 'diff_match_win_percent_grass_hh',
 'diff_games_win_percent_grass_hh']

# Define data frames for the labels - ie the expected results of each match - and the features to train on.
# The training data (ie the historical matches) is 'labeled' in that each row has the match result in the 'outcome' column.
target = df.outcome
features = df[features_list]

#print('********* target ************')
#print(target)

# Split Data intro Train (80 %) and Test (20%) data sets
# Test data sets are used to 'test' the accuracy of the model. ie, give the model data it has not yet seen, and see if it predicts
# the desired result (which is in the 'outcome' column of the data).
# So, using the train_features dataset, generate a model that predicts the train_target results.
# Then, using the test_features data, run it through the model to see if it correctly predicts the test_target results.
train_features, test_features, train_target, test_target = train_test_split(features, target, test_size=0.20, random_state=1)

#print('********* train_target  ************')
#print(train_target)
#sys.exit()

# Build the neural network.
# Details
#    - Number of Layers: 3. (2 Hidden Layers)
#    - Number of Neuros in each layer: 64->32->1
#    - Activation relu->relu->sigmoid
#    - Stop if validation loss does not improve for 500 epochs
#    - Save the best model which gives the maximum validation accuracy.
# The model has only 1 output neuron, because there is only one outcome we want (win = y/n)
# The input shape of the first layer is the number of features/columns to be used.
network = models.Sequential()
network.add(layers.Dense(units=64, activation='relu', input_shape=(len(features.columns),)))
network.add(layers.Dense(units=32, activation='relu'))
network.add(layers.Dense(units=1, activation='sigmoid'))

network.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# display the characteristics of the model
network.summary()

es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=500)
mc = ModelCheckpoint('data/best_model.h5', monitor='val_loss', mode='min', verbose=2, save_best_only=True)

history = network.fit(train_features, train_target,
            epochs=1000, verbose=0, batch_size=128,
            validation_data=(test_features, test_target), callbacks=[es, mc])

saved_model = load_model('data/best_model.h5')

# Accuracy of the best model
# So, given the model, see how accurate it was on the training data, and then how accurate on the test data.
_, train_acc = saved_model.evaluate(train_features, train_target, verbose=0)
_, test_acc = saved_model.evaluate(test_features, test_target, verbose=0)

print('Train Accuracy: %.3f, Test Accuracy: %.3f' % (train_acc, test_acc))

"""
plt.subplots(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.title('Loss after each Epoch')
plt.plot(history.epoch[::10], history.history['loss'][::10], label='Train')
plt.plot(history.epoch[::10], history.history['val_loss'][::10], label='Test')
plt.legend(['Train', 'Test'],loc='upper right', title='Sample', facecolor='white',fancybox=True)
plt.xlabel('Loss')
plt.ylabel('Epochs')

plt.subplot(1, 2, 2)
plt.title('Accuracy after each Epoch')
plt.plot(history.epoch[::10], history.history['acc'][::10], label='Train')
plt.plot(history.epoch[::10], history.history['val_acc'][::10], label='Test')
plt.xlabel('Accuracy')
plt.ylabel('Epochs')
plt.legend(['Train', 'Test'], loc='upper left', title='Sample', facecolor='white', fancybox=True)

plt.savefig('data/results/loss_acc.jpg', quality=100)
"""

# Test Data classification report and confusion matrix

print('******* clasification report for test data **********')
print(classification_report(test_target, saved_model.predict_classes(test_features)))
print('******* confusion matrix for test data **********')
print(confusion_matrix(test_target, saved_model.predict_classes(test_features)))

print('******* clasification report for test data **********')
print(classification_report(train_target, saved_model.predict_classes(train_features)))
print('******* confusion matrix for test data **********')
print(confusion_matrix(train_target, saved_model.predict_classes(train_features)))

# Load in the target data that we want to predict over
# 2019 Wimbledon Matches
df_2019 = pd.read_csv('data/wimbledon_2019.csv')
df_raw = pd.read_csv('data/mens/combined_raw_data.csv')

df_2019['Date'] = '2019/07/07'
df_2019['Surface'] = 'Grass'
df_2019['diff_rank'] = df_2019['player_0_rank'] - df_2019['player_1_rank']

# create the features/columns we need in the target data
# Note that this same process was done on the training data too.
df_2019 = create_features(df_2019, df_raw)

# just grab the columns we need for the predictions
features_16 = df_2019[features_list]

# run the predictions (ie, who will win each match) using the model we developed above
df_2019['prediction'] = saved_model.predict_classes(features_16)
# run the probabilities of the predictions
df_2019['probability'] = 1 - np.abs(df_2019.prediction - saved_model.predict_proba(features_16).flatten())

# display the results
print(df_2019[['Round', 'player_0', 'player_1', 'prediction', 'probability']])

