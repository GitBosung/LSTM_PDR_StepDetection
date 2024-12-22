import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import MinMaxScaler

# Butterworth 필터 함수
def butterworth_filter(data, cutoff_frequency=3, fs=50, order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_frequency / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# 데이터 필터링 함수
def filter_data(df):
    for axis in ['x', 'y', 'z']:
        df[f'Filtered_Acc_{axis}'] = butterworth_filter(df[f'Accelerometer {axis}'])
        df[f'Filtered_Gyro_{axis}'] = butterworth_filter(df[f'Gyroscope {axis}'])
    df['Filtered_AccNorm'] = np.sqrt(df['Accelerometer x']**2 + df['Accelerometer y']**2 + df['Accelerometer z']**2)
    return df

# 학습 데이터 준비 함수
def prepare_training_data(dfs, features, label_column, time_steps):
    combined_df = pd.concat(dfs, ignore_index=True).sort_values(by='Elapsed Time').reset_index(drop=True)
    X = combined_df[features].values
    y = combined_df[label_column].values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_lstm, y_lstm = [], []
    for i in range(time_steps, len(X_scaled)):
        X_lstm.append(X_scaled[i - time_steps:i])
        y_lstm.append(y[i])
    return np.array(X_lstm), np.array(y_lstm)
