import os
from tabnanny import verbose

import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model


class MLOps:
    def __init__(self, sett):
        self.sett = sett
        # Load test data
        self.xy_test = joblib.load('./data/xy_full/xy_test.joblib')
        self.xy_test_w = joblib.load('./data/xy_full_windows/xy_test.joblib')
        self.models = {}

        model_dir = './data/models/'
        for filename in os.listdir(model_dir):
            file_path = os.path.join(model_dir, filename)
            # Process Keras models
            if filename.endswith('.keras'):
                # LSTM models
                if filename.startswith('lstm_balanced_'):
                    key = filename.replace('lstm_balanced_', '').replace('.keras', '')
                    seed = key[-2:]
                    key = key[:-3]
                    self.models.setdefault(key, {})[f'lstm_balanced_{seed}'] = load_model(file_path)
                elif filename.startswith('lstm_'):
                    # Exclude the balanced case
                    if not filename.startswith('lstm_balanced_'):
                        key = filename.replace('lstm_', '').replace('.keras', '')
                        seed = key[-2:]
                        key = key[:-3]
                        self.models.setdefault(key, {})[f'lstm_{seed}'] = load_model(file_path)
                # GRU models
                elif filename.startswith('gru_balanced_'):
                    key = filename.replace('gru_balanced_', '').replace('.keras', '')
                    seed = key[-2:]
                    key = key[:-3]
                    self.models.setdefault(key, {})[f'gru_balanced_{seed}'] = load_model(file_path)
                elif filename.startswith('gru_'):
                    if not filename.startswith('gru_balanced_'):
                        key = filename.replace('gru_', '').replace('.keras', '')
                        seed = key[-2:]
                        key = key[:-3]
                        self.models.setdefault(key, {})[f'gru_{seed}'] = load_model(file_path)
                # MLP models
                elif filename.startswith('mlp_balanced_'):
                    key = filename.replace('mlp_balanced_', '').replace('.keras', '')
                    seed = key[-2:]
                    key = key[:-3]
                    self.models.setdefault(key, {})[f'mlp_balanced_{seed}'] = load_model(file_path)
                elif filename.startswith('mlp_'):
                    if not filename.startswith('mlp_balanced_'):
                        key = filename.replace('mlp_', '').replace('.keras', '')
                        seed = key[-2:]
                        key = key[:-3]
                        self.models.setdefault(key, {})[f'mlp_{seed}'] = load_model(file_path)

            # Process joblib models (calibrators and logistic regression models)
            elif filename.endswith('.joblib'):
                # LSTM calibrator
                if filename.startswith('lstm_calibrator_'):
                    key = filename.replace('lstm_calibrator_', '').replace('.joblib', '')
                    seed = key[-2:]
                    key = key[:-3]
                    self.models.setdefault(key, {})[f'lstm_calibrator_{seed}'] = joblib.load(file_path)
                # GRU calibrator
                elif filename.startswith('gru_calibrator_'):
                    key = filename.replace('gru_calibrator_', '').replace('.joblib', '')
                    seed = key[-2:]
                    key = key[:-3]
                    self.models.setdefault(key, {})[f'gru_calibrator_{seed}'] = joblib.load(file_path)
                # MLP calibrator
                elif filename.startswith('mlp_calibrator_'):
                    key = filename.replace('mlp_calibrator_', '').replace('.joblib', '')
                    seed = key[-2:]
                    key = key[:-3]
                    self.models.setdefault(key, {})[f'mlp_calibrator_{seed}'] = joblib.load(file_path)
                # Logistic Regression models and calibrators
                elif filename.startswith('log_reg_calibrator_'):
                    key = filename.replace('log_reg_calibrator_', '').replace('.joblib', '')
                    seed = key[-2:]
                    key = key[:-3]
                    self.models.setdefault(key, {})[f'log_reg_calibrator_{seed}'] = joblib.load(file_path)
                elif filename.startswith('log_reg_balanced_'):
                    key = filename.replace('log_reg_balanced_', '').replace('.joblib', '')
                    seed = key[-2:]
                    key = key[:-3]
                    self.models.setdefault(key, {})[f'log_reg_balanced_{seed}'] = joblib.load(file_path)
                elif filename.startswith('log_reg_'):
                    # Exclude the balanced and calibrator cases
                    if not (filename.startswith('log_reg_balanced_') or filename.startswith('log_reg_calibrator_')):
                        key = filename.replace('log_reg_', '').replace('.joblib', '')
                        seed = key[-2:]
                        key = key[:-3]
                        self.models.setdefault(key, {})[f'log_reg_{seed}'] = joblib.load(file_path)

    def predict(self):
        predictions = {}

        # Ensure the predictions directory exists
        predictions_dir = './data/predictions/'
        os.makedirs(predictions_dir, exist_ok=True)

        for date_key, data in self.xy_test.items():
            if date_key in self.models:
                data_w = self.xy_test_w[date_key]
                model_dict = self.models[date_key]

                # Separate features and target variables
                x_test = data.drop(columns=['y', 'LogReturn'])
                x_test_w = data_w[0][:, :, 1]
                x_test_w = x_test_w.reshape((x_test_w.shape[0], x_test_w.shape[1], 1))
                y_true = data['y']
                log_returns = data['LogReturn']

                # Initialize DataFrame to store predictions
                predictions_df = pd.DataFrame({
                    'y_true': y_true,
                    'log_returns': log_returns
                })

                for seed in ["10", "09", "08", "07", "06", "05", "04", "03", "02", "01"]:
                    # Define the required models for this key
                    required_models = [
                        f'lstm_balanced_{seed}', f'lstm_calibrator_{seed}', f'lstm_{seed}',
                        f'gru_balanced_{seed}', f'gru_calibrator_{seed}', f'gru_{seed}',
                        f'mlp_balanced_{seed}', f'mlp_calibrator_{seed}', f'mlp_{seed}',
                        f'log_reg_balanced_{seed}', f'log_reg_calibrator_{seed}', f'log_reg_{seed}'
                    ]
                    missing_models = [m for m in required_models if m not in model_dict]
                    if missing_models:
                        print(f"Missing models for date {date_key}: {missing_models}. Skipping prediction for this date.")
                        continue

                    # LSTM Predictions
                    lstm_balanced_model = model_dict[f'lstm_balanced_{seed}']
                    lstm_calibrator = model_dict[f'lstm_calibrator_{seed}']
                    lstm_balanced_probs = lstm_balanced_model.predict(x_test_w).flatten()
                    lstm_balanced_probs_calibrated = lstm_calibrator.predict_proba(lstm_balanced_probs.reshape(-1, 1),
                                                                                   verbose=0)[:, 1]
                    predictions_df[f'y_pred_lstm_balanced_{seed}'] = lstm_balanced_probs_calibrated

                    lstm_model = model_dict[f'lstm_{seed}']
                    lstm_probs = lstm_model.predict(x_test_w, verbose=0).flatten()
                    predictions_df[f'y_pred_lstm_{seed}'] = lstm_probs

                    # GRU Predictions
                    gru_balanced_model = model_dict[f'gru_balanced_{seed}']
                    gru_calibrator = model_dict[f'gru_calibrator_{seed}']
                    gru_balanced_probs = gru_balanced_model.predict(x_test_w).flatten()
                    gru_balanced_probs_calibrated = gru_calibrator.predict_proba(gru_balanced_probs.reshape(-1, 1),
                                                                                 verbose=0)[:, 1]
                    predictions_df[f'y_pred_gru_balanced_{seed}'] = gru_balanced_probs_calibrated

                    gru_model = model_dict[f'gru_{seed}']
                    gru_probs = gru_model.predict(x_test_w, verbose=0).flatten()
                    predictions_df[f'y_pred_gru_{seed}'] = gru_probs

                    # MLP Predictions
                    mlp_balanced_model = model_dict[f'mlp_balanced_{seed}']
                    mlp_calibrator = model_dict[f'mlp_calibrator_{seed}']
                    mlp_balanced_probs = mlp_balanced_model.predict(x_test).flatten()
                    mlp_balanced_probs_calibrated = mlp_calibrator.predict_proba(mlp_balanced_probs.reshape(-1, 1),
                                                                                 verbose=0)[:, 1]
                    predictions_df[f'y_pred_mlp_balanced_{seed}'] = mlp_balanced_probs_calibrated

                    mlp_model = model_dict[f'mlp_{seed}']
                    mlp_probs = mlp_model.predict(x_test, verbose=0).flatten()
                    predictions_df[f'y_pred_mlp_{seed}'] = mlp_probs

                    # Logistic Regression Predictions
                    log_reg_balanced_model = model_dict[f'log_reg_balanced_{seed}']
                    log_reg_calibrator = model_dict[f'log_reg_calibrator_{seed}']
                    log_reg_balanced_probs = log_reg_balanced_model.predict_proba(x_test)[:, 1]
                    log_reg_balanced_probs_calibrated = log_reg_calibrator.predict_proba(
                        log_reg_balanced_probs.reshape(-1, 1), verbose=0)[:, 1]
                    predictions_df[f'y_pred_log_reg_balanced_{seed}'] = log_reg_balanced_probs_calibrated

                    log_reg_model = model_dict[f'log_reg_{seed}']
                    log_reg_probs = log_reg_model.predict_proba(x_test, verbose=0)[:, 1]
                    predictions_df[f'y_pred_log_reg_{seed}'] = log_reg_probs

                    # Save predictions for the current date
                    predictions[date_key] = predictions_df
            else:
                print(f"No model found for date {date_key}. Skipping prediction for this date.")

        # Save the predictions dictionary using joblib
        joblib.dump(predictions, os.path.join(predictions_dir, 'predictions.joblib'))
