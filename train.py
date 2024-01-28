import argparse
import os
import numpy as np

from keras.models import Sequential, save_model
from keras.layers import GRU, Conv1D, AveragePooling1D, Dense, Flatten
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.regularizers import L1

from src.helpers import generate_input_sequencial_data, transform_time_to_time_data, get_dhn_clusters, get_dhn_from_id, step_scheduler, flatten_data

parser = argparse.ArgumentParser(description="Training cluster arguments",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument("--model", help="select model to use [gru, cnn, mlp]", type=str, default="gru")
parser.add_argument("--model_version", help="add model version to use [1,2]", type=int)
parser.add_argument("--cluster_key", help="cluster key among the selected clusters", type=str)
parser.add_argument("--cluster_dhn_id", help="id of the DHN having the cluster [1,2,3,4]", type=int)
parser.add_argument("--cluster_set_id", help="random walk set from which the cluster [1,2]", type=int)

parser.add_argument("--reg", help="if true, use l1 regularization during traing", type=int, default=0, required=False)

args = parser.parse_args()
config = vars(args)

# Find the cluster
model_architecture = config["model"]
version = int(config["model_version"])
key = config["cluster_key"]
dhn_id = int(config["cluster_dhn_id"])
set_id = int(config["cluster_set_id"])

if "reg" in config:
    use_reg = int(config["reg"])
else:
    use_reg = 0

l1_rate = 0.0001
n_epochs = 100

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
scheduler_cl = LearningRateScheduler(schedule=step_scheduler)

# Get cluster datas 
dhn = get_dhn_from_id(dhn_id, os.path.join(".", "ARTICLE_dhn_data"))
cluster = get_dhn_clusters(dhn_id, os.path.join(".", "ARTICLE_dhn_data"))[f"v{set_id}"][key]

train_x, train_y, test_x, test_y, scaller_y, scaller_x = generate_input_sequencial_data(dhn, cluster, shuffle_data=True, time_step=60)
folder_save_path = os.path.join("ARTICLE_trained_mls_data", f"Network_{dhn_id}", f"copied_cluster_set_{set_id}_files", f"cluster_{key}_folder")

if model_architecture == "gru":
    if version == 1:
        if use_reg == 0:
            model = Sequential([
                GRU(units=20, unroll=False, return_sequences=True, input_shape=(train_x.shape[1], train_x.shape[2]), reset_after=True),
                GRU(units=20, unroll=False, return_sequences=False, reset_after=True),
                Dense(60, activation='relu'),
                Dense(train_y.shape[1], activation='relu', name='output_dense')
            ])
        else:
            model = Sequential([
                GRU(units=20, unroll=False, return_sequences=True, input_shape=(train_x.shape[1], train_x.shape[2]), reset_after=True,
                    kernel_regularizer=L1(l1=l1_rate), bias_regularizer=L1(l1=l1_rate)),
                GRU(units=20, unroll=False, return_sequences=False, reset_after=True, 
                    kernel_regularizer=L1(l1=l1_rate), bias_regularizer=L1(l1=l1_rate)),
                Dense(60, activation='relu', kernel_regularizer=L1(l1=l1_rate), bias_regularizer=L1(l1=l1_rate)),
                Dense(train_y.shape[1], activation='relu', name='output_dense')
            ])
    else:
        if use_reg == 0:
            model = Sequential([
                GRU(units=30, unroll=False, return_sequences=True, input_shape=(train_x.shape[1], train_x.shape[2]), reset_after=True),
                GRU(units=30, unroll=False, return_sequences=False, reset_after=True),
                Dense(40, activation='relu'),
                Dense(train_y.shape[1], activation='relu', name='output_dense')
            ])
        else:
            model = Sequential([
                GRU(units=30, unroll=False, return_sequences=True, input_shape=(train_x.shape[1], train_x.shape[2]), reset_after=True,
                    kernel_regularizer=L1(l1=l1_rate), bias_regularizer=L1(l1=l1_rate)),
                GRU(units=30, unroll=False, return_sequences=False, reset_after=True, 
                    kernel_regularizer=L1(l1=l1_rate), bias_regularizer=L1(l1=l1_rate)),
                Dense(40, activation='relu', kernel_regularizer=L1(l1=l1_rate), bias_regularizer=L1(l1=l1_rate)),
                Dense(train_y.shape[1], activation='relu', name='output_dense')
            ])
    
    print(f'Training model gru for cluster {key}')
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mae', metrics=['mae'])
    history = model.fit(train_x, train_y, epochs=n_epochs, batch_size=100, validation_split=0.2, verbose=2, callbacks=[early_stopping, scheduler_cl])
    preds = scaller_y.inverse_transform(model.predict(test_x))
    reals = scaller_y.inverse_transform(test_y)
    mae = np.mean(np.abs(preds-reals))
    print(f"GRU model trained with mae = {mae}")
    save_model(model, os.path.join(folder_save_path, f"model_gru_v{version}_{key}.h5"), save_traces=False)
            
elif model_architecture == "cnn":
    if version == 1:
        if use_reg == 0:
            model = Sequential([
                Conv1D(filters=10, kernel_size=4, activation="selu", input_shape=(train_x.shape[1], train_x.shape[2]),),
                AveragePooling1D(pool_size=4),
                Conv1D(filters=20, kernel_size=3, activation="selu"),
                AveragePooling1D(pool_size=3),
                Conv1D(filters=30, kernel_size=2, activation="selu"),
                AveragePooling1D(pool_size=2),
                Flatten(),
                Dense(40, activation='relu'),
                Dense(train_y.shape[1], activation='relu', name='output_dense')
            ])
        else:
            model = Sequential([
                Conv1D(filters=10, kernel_size=4, input_shape=(train_x.shape[1], train_x.shape[2]), kernel_regularizer=L1(l1=l1_rate), bias_regularizer=L1(l1=l1_rate), activation="selu"),
                AveragePooling1D(pool_size=4),
                Conv1D(filters=20, kernel_size=3, kernel_regularizer=L1(l1=l1_rate), bias_regularizer=L1(l1=l1_rate), activation="selu"),
                AveragePooling1D(pool_size=3),
                Conv1D(filters=30, kernel_size=2, kernel_regularizer=L1(l1=l1_rate), bias_regularizer=L1(l1=l1_rate), activation="selu"),
                AveragePooling1D(pool_size=2),
                Flatten(),
                Dense(40, activation='relu', kernel_regularizer=L1(l1=l1_rate), bias_regularizer=L1(l1=l1_rate)),
                Dense(train_y.shape[1], activation='relu', name='output_dense')
            ])
    else:
        if use_reg == 0:
            model = Sequential([
                Conv1D(filters=10, kernel_size=3, activation="selu", input_shape=(train_x.shape[1], train_x.shape[2]),),
                AveragePooling1D(pool_size=3),
                Conv1D(filters=20, kernel_size=2, activation="selu"),
                AveragePooling1D(pool_size=2),
                Flatten(),
                Dense(40, activation='relu'),
                Dense(train_y.shape[1], activation='relu', name='output_dense')
            ])
        else:
            model = Sequential([
                Conv1D(filters=10, kernel_size=3, input_shape=(train_x.shape[1], train_x.shape[2]), kernel_regularizer=L1(l1=l1_rate), bias_regularizer=L1(l1=l1_rate), activation="selu"),
                AveragePooling1D(pool_size=3),
                Conv1D(filters=20, kernel_size=2, kernel_regularizer=L1(l1=l1_rate), bias_regularizer=L1(l1=l1_rate), activation="selu"),
                AveragePooling1D(pool_size=2),
                Flatten(),
                Dense(40, activation='relu', kernel_regularizer=L1(l1=l1_rate), bias_regularizer=L1(l1=l1_rate)),
                Dense(train_y.shape[1], activation='relu', name='output_dense')
            ])
    
    print(f'Training model cnn for cluster {key}')        
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mae', metrics=['mae'])
    history = model.fit(train_x, train_y, epochs=n_epochs, batch_size=100, validation_split=0.2, verbose=2, callbacks=[early_stopping, scheduler_cl])
    preds = scaller_y.inverse_transform(model.predict(test_x))
    reals = scaller_y.inverse_transform(test_y)
    mae = np.mean(np.abs(preds-reals))
    print(f"CNN model trained with mae = {mae}")
    save_model(model, os.path.join(folder_save_path, f"model_cnn_v{version}_{key}.h5"), save_traces=False)

elif model_architecture == "mlp":
    if version == 1:
        train_x, train_y = transform_time_to_time_data(train_x, train_y)
        test_x, test_y = transform_time_to_time_data(test_x, test_y)
    else:
        train_x, train_y = flatten_data(train_x, train_y)
        test_x, test_y = flatten_data(test_x, test_y)
        
    if use_reg == 0:
        model = Sequential([
                    Dense(units=60, input_shape=(train_x.shape[1],), activation='relu'),
                    Dense(units=60, activation='relu'),
                    Dense(units=40, activation='relu'),
                    Dense(units=train_y.shape[1], activation='relu', name='output_dense')
                ])
    else:
        model = Sequential([
                    Dense(units=60, input_shape=(train_x.shape[1],), activation='relu', kernel_regularizer=L1(l1=l1_rate), bias_regularizer=L1(l1=l1_rate)),
                    Dense(units=60, activation='relu', kernel_regularizer=L1(l1=l1_rate), bias_regularizer=L1(l1=l1_rate)),
                    Dense(units=40, activation='relu', kernel_regularizer=L1(l1=l1_rate), bias_regularizer=L1(l1=l1_rate)),
                    Dense(units=train_y.shape[1], activation='relu', name='output_dense')
                ]) # 6,xxx
    
    print(f'Training model mlp for cluster {key}')   
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mae', metrics=['mae'])
    history = model.fit(train_x, train_y, epochs=n_epochs, batch_size=100, validation_split=0.2, verbose=2, callbacks=[early_stopping, scheduler_cl])
    preds = scaller_y.inverse_transform(model.predict(test_x))
    reals = scaller_y.inverse_transform(test_y)
    mae = np.mean(np.abs(preds-reals))
    print(f"MLP model trained with mae = {mae}")
    save_model(model, os.path.join(folder_save_path, f"model_mlp_v{version}_{key}.h5"), save_traces=False)