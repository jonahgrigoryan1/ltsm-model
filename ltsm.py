# detect and init the TPU
tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()

# instantiate a distribution strategy
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)  

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from numba import jit
from tqdm.notebook import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization, GRU, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.regularizers import l1_l2


# =====================
# Data Loading
# =====================

train_data = pd.read_csv("/kaggle/input/optiver-trading-at-the-close/train.csv")

# List of columns for which we want to fill missing values
columns_to_fill = ['imbalance_size', 'reference_price', 'matched_size', 'bid_price', 'ask_price', 'wap', 'target']

# For each stock_id, fill the missing values
for stock in train_data['stock_id'].unique():
    for col in columns_to_fill:
        train_data.loc[train_data['stock_id'] == stock, col] = train_data.loc[train_data['stock_id'] == stock, col].fillna(method='ffill').fillna(method='bfill')

train_data['far_price'] = train_data.groupby('stock_id')['far_price'].transform(lambda x: x.interpolate()).fillna(method='ffill').fillna(method='bfill')
train_data['near_price'] = train_data.groupby('stock_id')['near_price'].transform(lambda x: x.interpolate()).fillna(method='ffill').fillna(method='bfill')



# Calculate the difference in 'Ask Size' and 'Bid Size' between the current and previous ticks
train_data['Ask_Size_Diff'] = train_data['ask_size'].diff()
train_data['Bid_Size_Diff'] = train_data['bid_size'].diff()

# Compute Order Flow Imbalance (OFI)
train_data['OFI'] = (train_data['Ask_Size_Diff'] * train_data['ask_price']) - (train_data['Bid_Size_Diff'] * train_data['bid_price'])

# Handle missing values generated due to differencing
train_data['OFI'].fillna(0, inplace=True)
train_data['Liquidity_Imbalance'] = (train_data['bid_size'] - train_data['ask_size']) / (train_data['bid_size'] + train_data['ask_size'])
train_data.fillna(method='bfill', inplace=True)
train_data.fillna(method='ffill', inplace=True)



# =====================
# Data Splitting
# =====================

tscv = TimeSeriesSplit(n_splits=5)

for train_index, valid_index in tscv.split(train_data):
    train_subset = train_data.iloc[train_index]
    valid_subset = train_data.iloc[valid_index]
    
    # Normalize using MinMaxScaler
    columns_to_normalize = ['imbalance_size', 'reference_price', 'matched_size', 'far_price', 'near_price', 
                            'bid_price', 'ask_price', 'bid_size', 'ask_size', 'OFI', 'Liquidity_Imbalance']

    # Ensure that columns are float32 before normalization
    for col in columns_to_normalize:
        train_subset[col] = train_subset[col].astype('float32')
        valid_subset[col] = valid_subset[col].astype('float32')

    scaler = MinMaxScaler()
    train_subset[columns_to_normalize] = scaler.fit_transform(train_subset[columns_to_normalize])
    valid_subset[columns_to_normalize] = scaler.transform(valid_subset[columns_to_normalize])  # Use the same scaler



# Data Preprocessing: Sequence Creation
SEQUENCE_LENGTH = 30

def create_sequences_for_stock(stock_data, SEQUENCE_LENGTH):
    features = stock_data[columns_to_normalize].values
    targets = stock_data['target'].values

    num_records = features.shape[0]
    num_features = features.shape[1]

    X = np.empty((num_records - SEQUENCE_LENGTH, SEQUENCE_LENGTH, num_features))
    y = np.empty(num_records - SEQUENCE_LENGTH)

    for i in range(num_records - SEQUENCE_LENGTH):
        X[i] = features[i:i + SEQUENCE_LENGTH]
        y[i] = targets[i + SEQUENCE_LENGTH]  # assuming target is the last column

    return X, y



X_temp, y_temp = create_sequences_for_stock(train_data[train_data['stock_id'] == train_data['stock_id'].iloc[0]], SEQUENCE_LENGTH)
print(X_temp.shape, y_temp.shape)
print(X_temp[0], y_temp[0])

# Create sequences for each stock and concatenate them
X_train_list, y_train_list = [], []
for stock_id in train_subset['stock_id'].unique():
    stock_data = train_subset[train_subset['stock_id'] == stock_id]
    X_stock, y_stock = create_sequences_for_stock(stock_data, SEQUENCE_LENGTH)
    X_train_list.append(X_stock)
    y_train_list.append(y_stock)

X_train = np.concatenate(X_train_list, axis=0)
y_train = np.concatenate(y_train_list, axis=0)

X_valid_list, y_valid_list = [], []
for stock_id in valid_subset['stock_id'].unique():
    stock_data = valid_subset[valid_subset['stock_id'] == stock_id]
    X_stock, y_stock = create_sequences_for_stock(stock_data, SEQUENCE_LENGTH)
    X_valid_list.append(X_stock)
    y_valid_list.append(y_stock)

X_valid = np.concatenate(X_valid_list, axis=0)
y_valid = np.concatenate(y_valid_list, axis=0)

# After creating sequences, ensure the datatype is float32
X_train = X_train.astype('float32')
y_train = y_train.astype('float32')
X_valid = X_valid.astype('float32')
y_valid = y_valid.astype('float32')


# Define the checkpoint path and filename
checkpoint_filepath = './weights_best.h5'

# Create the callback
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,          # Set to True if you only want to save weights and not the entire model
    monitor='val_loss',              # Monitor validation loss
    mode='min',                      # Mode should be 'min' for validation loss
    save_best_only=True,             # Save only the best model weights
    verbose=1                        # Print a message every time weights are saved
)


BATCH_SIZE = 256 * tpu_strategy.num_replicas_in_sync  # This doubles the effective batch size


with tpu_strategy.scope():
    model = keras.Sequential([    
        # Initial LSTM layer
        Bidirectional(LSTM(128, kernel_initializer='glorot_normal', return_sequences=True, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4), input_shape=(X_train.shape[1], X_train.shape[2]))),
        Dropout(0.5),

        #Second LSTM layer
        Bidirectional(LSTM(128, kernel_initializer='glorot_normal', return_sequences=True, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))),
        Dropout(0.5),

        # GRU layer
        Bidirectional(GRU(128, kernel_initializer='glorot_normal', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))),
        Dropout(0.5),
    
         # Dense layer
        Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        Dropout(0.5),

        Dense(1)
    ])        
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005, epsilon=1e-08)
    model.compile(optimizer=optimizer, loss='mae')


# EarlyStopping callback
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# ReduceLROnPlateau callback
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

# Fit the model
history = model.fit(
    X_train, y_train, 
    epochs=50, 
    batch_size=BATCH_SIZE, 
    validation_data=(X_valid, y_valid), 
    verbose=1, 
    callbacks=[early_stop, reduce_lr, checkpoint_callback]
)