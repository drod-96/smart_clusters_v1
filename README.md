# ML_clusters

This project contains the selected clusters, trained keras ML models corresponding saved in .h5, different figures, training processes .csv files and saved performances in .csv.

## ML architectures

Three main ML architectures have been studied, developped and trained using Keras librairy. The training is performed in 100 epochs using Adam optimizer with 20% of train-validation splits. Early stopping callback monitors the convergences with restoring the best weights based on validation losses using MAE loss. Learning rate scheduler callback changes the learning rate of Adam optimizer over the training processes.

```
model.compile(optimizer=Adam(learning_rate=0.001), loss='mae')
                early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
                scheduler_cl = LearningRateScheduler(schedule=step_scheduler)
                history = model.fit(train_x, train_y, epochs=n_epochs, batch_size=b_size, validation_split=0.2, verbose=0, callbacks=[early_stopping, scheduler_cl])
```


### FFNN models

Both FFNN models use 3 hidden layers of perceptron with respectively 60, 60 and 40 ReLU activated neurons. The first model performs time-to-time prediction but the second model uses a flatened 60 sequences.

```
model = Sequential([
            Dense(units=60, 
                    input_shape=(input_features,), 
                    activation='relu', 
                    kernel_regularizer=L1(regulazer_rate), bias_regularizer=L1(regulazer_rate), # with or without
                ),
            Dense(units=60,  
                    activation='relu', 
                    kernel_regularizer=L1(regulazer_rate), bias_regularizer=L1(regulazer_rate),
                ),
            Dense(units=40, 
                    activation='relu', 
                    kernel_regularizer=L1(regulazer_rate), bias_regularizer=L1(regulazer_rate),
                ),
            Dense(units=output_features, activation='relu', name='output_dense')
            ])
```

### CNN models

Two versions of CNN models have been considered. Both rely on 1D Convolution + Average pooling layers but differ by the number of layers, the number of filters and the size of the filers.

[-] Version 1
```
optimal_act = 'selu'
regulazer_rate = 0.0001

cnn_model_v1 = Sequential([
            Conv1D(filters=10,
                    kernel_size=4,
                    activation=optimal_act,  
                    input_shape=(60, input_features),
                    kernel_regularizer=L1(regulazer_rate), bias_regularizer=L1(regulazer_rate), # with or without
                    name='cnn_1',
            ),
            AvgPool1D(pool_size=4, name='pooling_1'),

            Conv1D(filters=20, 
                    kernel_size=3,
                    activation=optimal_act,
                    kernel_regularizer=L1(regulazer_rate), bias_regularizer=L1(regulazer_rate),
                    name='cnn_2',
            ),
            AvgPool1D(pool_size=3, name='pooling_2'),

            Conv1D(filters=30, 
                    kernel_size=2,
                    activation=optimal_act,
                    kernel_regularizer=L1(regulazer_rate), bias_regularizer=L1(regulazer_rate),
                    name='cnn_3',
            ),
            AvgPool1D(pool_size=2, name='pooling_3'),

            Flatten(),
            Dense(units=40, activation='relu', kernel_regularizer=L1(regulazer_rate), bias_regularizer=L1(regulazer_rate),),

            Dense(units=output_features, activation='relu', name='output_dense')
        ])
```

[-] Version 2
```
cnn_model_v2 = Sequential([
            Conv1D(filters=10,
                    kernel_size=3,
                    activation=optimal_act,  
                    input_shape=(60, input_features),
                    kernel_regularizer=L1(regulazer_rate), bias_regularizer=L1(regulazer_rate), # with or without
                    name='cnn_1',
            ),
            AvgPool1D(pool_size=3, name='pooling_1'),

            Conv1D(filters=20, 
                    kernel_size=2,
                    activation=optimal_act,
                    kernel_regularizer=L1(regulazer_rate), bias_regularizer=L1(regulazer_rate),
                    name='cnn_2',
            ),
            AvgPool1D(pool_size=2, name='pooling_2'),

            Flatten(),
            Dense(units=40, activation='relu', kernel_regularizer=L1(regulazer_rate), bias_regularizer=L1(regulazer_rate),),

            Dense(units=output_features, activation='relu', name='output_dense')
        ])
```


### RNN models

Two versions of RNN models have been considered. The recurrent layers are using Gated Recurrent Unit cells [https://arxiv.org/abs/1412.3555]. Both differ by the number of GRU-based layers and the number of connected dense layers.

[-] Version 1
```
rnn_model_v1 = Sequential([
            GRU(units=20, 
                unroll=False, return_sequences=True, 
                input_shape=(60, input_features),
                kernel_regularizer=L1(regulazer_rate), bias_regularizer=L1(regulazer_rate), # with or without
                name='rnn_1',
            ),

            GRU(units=20, 
                unroll=False, return_sequences=False, 
                kernel_regularizer=L1(regulazer_rate), bias_regularizer=L1(regulazer_rate), # with or without
                name='rnn_2',
            ),

            Dense(units=60, activation='relu', kernel_regularizer=L1(regulazer_rate), bias_regularizer=L1(regulazer_rate),),

            Dense(units=output_features, activation='relu', name='output_dense')
        ])
```

[-] Version 2
```
rnn_model_v2 = Sequential([
            GRU(units=20, 
                unroll=False, return_sequences=True, 
                input_shape=(60, input_features),
                kernel_regularizer=L1(regulazer_rate), bias_regularizer=L1(regulazer_rate), # with or without
                name='rnn_1',
            ),

            GRU(units=20, 
                unroll=False, return_sequences=False, 
                kernel_regularizer=L1(regulazer_rate), bias_regularizer=L1(regulazer_rate), # with or without
                name='rnn_2',
            ),

            Dense(units=60, activation='relu', kernel_regularizer=L1(regulazer_rate), bias_regularizer=L1(regulazer_rate),),

            Dense(units=60, activation='relu', kernel_regularizer=L1(regulazer_rate), bias_regularizer=L1(regulazer_rate),),

            Dense(units=output_features, activation='relu', name='output_dense')
        ])
```


## Notebook files

