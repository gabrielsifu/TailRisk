import joblib
import os
import numpy as np
import pandas as pd
from sklearn.utils import class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


class DataScientist:
    def __init__(self, sett):
        self.sett = sett
        self.xy_train = joblib.load('./data/xy_full/xy_train.joblib')
        self.xy_train_windows = joblib.load('./data/xy_full_windows/xy_train.joblib')
        self.models = {}
        self.x_train_full = pd.DataFrame()
        self.y_train_full = pd.Series(dtype='int')
        self.x_val = pd.DataFrame()
        self.y_val = pd.Series(dtype='int')
        self.x_train_full_w, self.y_train_full_w, self.x_val_w, self.y_val_w = None, None, None, None

    def fit(self):
        for seed in ["10", "09", "08", "07", "06", "05", "04", "03", "02", "01"]:
            used_indices = set()
            mlp, mlp_balanced, lstm, lstm_balanced, gru, gru_balanced = None, None, None, None, None, None

            for key in sorted(self.xy_train.keys()):
                df_w = self.xy_train_windows[key]
                df = self.xy_train[key]

                x_w, y_w, x_new_train_w, y_new_train_w, x_new_val_w, y_new_val_w = self._prepare_data(df_w, used_indices, window=True)
                x, y, x_new_train, y_new_train, x_new_val, y_new_val = self._prepare_data(df, used_indices)
                self._accumulate_training_data(x_new_train_w, y_new_train_w, x_new_val_w, y_new_val_w, w=True)
                self._accumulate_training_data(x_new_train, y_new_train, x_new_val, y_new_val)

                class_weight_dict_w, val_sample_weights_w = self._compute_weights(self.y_train_full_w, self.y_val_w)
                class_weight_dict, val_sample_weights = self._compute_weights(self.y_train_full, self.y_val)

                print("gru_balanced")
                gru_balanced = self._initialize_gru_model(gru_balanced, x_new_train_w)
                gru_balanced.fit(
                    x_new_train_w, y_new_train_w,
                    epochs=10000,
                    batch_size=10000,
                    validation_data=(self.x_val_w, self.y_val_w, val_sample_weights_w),
                    callbacks=[self._create_early_stopping()],
                    class_weight=class_weight_dict_w,
                    verbose=1
                )
                print("     gru_calibrator")
                gru_calibrator = self._train_nn_calibrator(gru_balanced, window=True)

                print("lstm_balanced")
                lstm_balanced = self._initialize_lstm_model(lstm_balanced, x_new_train_w)
                lstm_balanced.fit(
                    x_new_train_w, y_new_train_w,
                    epochs=10000,
                    batch_size=10000,
                    validation_data=(self.x_val_w, self.y_val_w, val_sample_weights_w),
                    callbacks=[self._create_early_stopping()],
                    class_weight=class_weight_dict_w,
                    verbose=1
                )
                print("     lstm_calibrator")
                lstm_calibrator = self._train_nn_calibrator(lstm_balanced, window=True)

                print("lstm")
                lstm = self._initialize_lstm_model(lstm, x_new_train_w)
                lstm.fit(
                    x_new_train_w, y_new_train_w,
                    epochs=10000,
                    batch_size=10000,
                    validation_data=(self.x_val_w, self.y_val_w),
                    callbacks=[self._create_early_stopping()],
                    verbose=1
                )

                print("gru")
                gru = self._initialize_gru_model(gru, x_new_train_w)
                gru.fit(
                    x_new_train_w, y_new_train_w,
                    epochs=10000,
                    batch_size=10000,
                    validation_data=(self.x_val_w, self.y_val_w),
                    callbacks=[self._create_early_stopping()],
                    verbose=1
                )

                print("mlp_balanced")
                mlp_balanced = self._initialize_mlp_model(mlp_balanced, x_new_train)
                mlp_balanced.fit(
                    x_new_train, y_new_train,
                    epochs=10000,
                    batch_size=10000,
                    validation_data=(self.x_val, self.y_val, val_sample_weights),
                    callbacks=[self._create_early_stopping()],
                    class_weight=class_weight_dict,
                    verbose=1
                )
                print("     mlp_calibrator")
                mlp_calibrator = self._train_nn_calibrator(mlp_balanced)

                print("mlp")
                mlp = self._initialize_mlp_model(mlp, x_new_train)
                mlp.fit(
                    x_new_train, y_new_train,
                    epochs=10000,
                    batch_size=10000,
                    validation_data=(self.x_val, self.y_val),
                    callbacks=[self._create_early_stopping()],
                    verbose=1
                )

                print("log_reg")
                log_reg = LogisticRegression(max_iter=10000)
                log_reg.fit(self.x_train_full, self.y_train_full)

                print("log_reg_balanced")
                log_reg_balanced = LogisticRegression(class_weight='balanced', max_iter=10000)
                log_reg_balanced.fit(self.x_train_full, self.y_train_full)
                print("     log_reg_calibrator")
                log_reg_calibrator = self._train_lr_calibrator(log_reg_balanced)

                self.models[key] = {
                    'lstm_balanced': lstm_balanced,
                    'lstm_calibrator': lstm_calibrator,
                    'lstm': lstm,
                    'gru_balanced': gru_balanced,
                    'gru_calibrator': gru_calibrator,
                    'gru': gru,
                    'mlp': mlp,
                    'mlp_balanced': mlp_balanced,
                    'mlp_calibrator': mlp_calibrator,
                    'log_reg': log_reg,
                    'log_reg_balanced': log_reg_balanced,
                    'log_reg_calibrator': log_reg_calibrator,
                }

            self._save_models(seed)

    @staticmethod
    def _prepare_data(df, used_indices, window=False):
        if not window:
            x = df.drop(columns=['y', 'LogReturn'])
            y = df['y']
            new_indices = df.index.difference(used_indices)
            used_indices.update(df.index)

            x_new = x.loc[new_indices]
            y_new = y.loc[new_indices]

            x_new_train, x_new_val, y_new_train, y_new_val = train_test_split(
                x_new, y_new, test_size=0.2
            )
            return x, y, x_new_train, y_new_train, x_new_val, y_new_val
        else:
            x = df[0][:, :, 1]
            # y = df[0][:, -1, 2]
            y = df[1][:, 2]

            x_new = df[0][len(used_indices):, :, 1]
            y_new = df[1][len(used_indices):, 2]

            x_new_train, x_new_val, y_new_train, y_new_val = train_test_split(
                x_new, y_new, test_size=0.2
            )

            x = x.reshape((x.shape[0], x.shape[1], 1))
            x_new_train = x_new_train.reshape((x_new_train.shape[0], x_new_train.shape[1], 1))
            x_new_val = x_new_val.reshape((x_new_val.shape[0], x_new_val.shape[1], 1))
            return x, y, x_new_train, y_new_train, x_new_val, y_new_val

    def _accumulate_training_data(self, x_train, y_train, x_val, y_val, w=False):
        if not w:
            self.x_train_full = pd.concat([self.x_train_full, x_train])
            self.y_train_full = pd.concat([self.y_train_full, y_train])
            self.x_val = pd.concat([self.x_val, x_val])
            self.y_val = pd.concat([self.y_val, y_val])
        else:
            if self.x_train_full_w is None:
                self.x_train_full_w = x_train
                self.y_train_full_w = y_train
                self.x_val_w = x_val
                self.y_val_w = y_val
            else:
                self.x_train_full_w = np.concatenate([self.x_train_full_w, x_train], axis=0)
                self.y_train_full_w = np.concatenate([self.y_train_full_w, y_train], axis=0)
                self.x_val_w = np.concatenate([self.x_val_w, x_val], axis=0)
                self.y_val_w = np.concatenate([self.y_val_w, y_val], axis=0)

    @staticmethod
    def _compute_weights(y_train_full, y_val):
        unique_classes = np.unique(y_train_full)
        class_weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=y_train_full
        )
        class_weight_dict = dict(zip(unique_classes, class_weights))
        val_sample_weights = pd.Series(y_val).map(class_weight_dict).values
        return class_weight_dict, val_sample_weights

    @staticmethod
    def _create_early_stopping():
        return EarlyStopping(
            monitor='val_auc',
            patience=10,
            mode='max',
            restore_best_weights=True
        )

    @staticmethod
    def _initialize_gru_model(model, x_train):
        if model is None:
            model = Sequential([
                Input(shape=(x_train.shape[1], x_train.shape[2])),  # (timesteps, features)
                GRU(128),
                Dense(1, activation='sigmoid')
            ])
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=[tf.keras.metrics.AUC(name='auc')]
            )
        return model

    @staticmethod
    def _initialize_lstm_model(model, x_train):
        if model is None:
            model = Sequential([
                Input(shape=(x_train.shape[1], x_train.shape[2])),  # (timesteps, features)
                LSTM(128),
                Dense(1, activation='sigmoid')
            ])
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=[tf.keras.metrics.AUC(name='auc')]
            )
        return model

    @staticmethod
    def _initialize_mlp_model(model, x_train):
        if model is None:
            model = Sequential([
                Input(shape=(x_train.shape[1],)),
                Dense(128, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=[tf.keras.metrics.AUC(name='auc')]
            )
        return model

    def _train_nn_calibrator(self, model, window=False):
        if window:
            probs = model.predict(self.x_train_full_w).flatten()
        else:
            probs = model.predict(self.x_train_full).flatten()
        calibrator = LogisticRegression(max_iter=10000)
        if window:
            calibrator.fit(probs.reshape(-1, 1), self.y_train_full_w)
        else:
            calibrator.fit(probs.reshape(-1, 1), self.y_train_full)
        return calibrator

    def _train_lr_calibrator(self, model_lr_balanced):
        probs = model_lr_balanced.predict_proba(self.x_train_full)[:, 1]
        calibrator = LogisticRegression(max_iter=10000)
        calibrator.fit(probs.reshape(-1, 1), self.y_train_full)
        return calibrator

    def _save_models(self, seed):
        os.makedirs('./data/models/', exist_ok=True)
        for key, model_dict in self.models.items():
            model_dict['lstm'].save(f'./data/models/lstm_{key}_{seed}.keras')
            model_dict['lstm_balanced'].save(f'./data/models/lstm_balanced_{key}_{seed}.keras')
            joblib.dump(model_dict['lstm_calibrator'], f'./data/models/lstm_calibrator_{key}_{seed}.joblib')

            model_dict['gru'].save(f'./data/models/gru_{key}_{seed}.keras')
            model_dict['gru_balanced'].save(f'./data/models/gru_balanced_{key}_{seed}.keras')
            joblib.dump(model_dict['gru_calibrator'], f'./data/models/gru_calibrator_{key}_{seed}.joblib')

            model_dict['mlp'].save(f'./data/models/mlp_{key}_{seed}.keras')
            model_dict['mlp_balanced'].save(f'./data/models/mlp_balanced_{key}_{seed}.keras')
            joblib.dump(model_dict['mlp_calibrator'], f'./data/models/mlp_calibrator_{key}_{seed}.joblib')

            joblib.dump(model_dict['log_reg'], f'./data/models/log_reg_{key}_{seed}.joblib')
            joblib.dump(model_dict['log_reg_balanced'], f'./data/models/log_reg_balanced_{key}_{seed}.joblib')
            joblib.dump(model_dict['log_reg_calibrator'], f'./data/models/log_reg_calibrator_{key}_{seed}.joblib')
