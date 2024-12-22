import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# 데이터 준비 함수
def prepare_lstm_data(df, features, time_steps=100):
    X = df[features].values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_lstm = [X_scaled[i-time_steps:i] for i in range(time_steps, len(X_scaled))]
    return np.array(X_lstm)

# 스텝 병합 함수
def merge_steps(indices, values, window_size=10):
    if len(indices) == 0:
        return np.array([])

    max_index = len(values) - 1
    merged_steps = []
    current_group = [indices[0]]

    for i in range(1, len(indices)):
        if indices[i] <= current_group[-1] + window_size:
            current_group.append(indices[i])
        else:
            valid_group = [idx for idx in current_group if idx <= max_index]
            if valid_group:
                max_norm_index = valid_group[np.argmax(values[valid_group])]
                merged_steps.append(max_norm_index)
            current_group = [indices[i]]

    valid_group = [idx for idx in current_group if idx <= max_index]
    if valid_group:
        max_norm_index = valid_group[np.argmax(values[valid_group])]
        merged_steps.append(max_norm_index)

    return np.array(merged_steps)

# 걸음 검출 함수
def detect_steps_with_lstm(model_path, df, features, time_steps=100, window_size=10):
    # LSTM 모델 로드
    model = load_model(model_path)

    # 데이터 준비
    X_lstm = prepare_lstm_data(df, features, time_steps)

    # 예측 수행
    predictions = model.predict(X_lstm)

    # 스텝 인덱스 추출
    step_indices = np.where(np.argmax(predictions > 0.5, axis=1) == 1)[0] + time_steps

    # 스텝 병합 처리 (중복 제거)
    merged_step_indices = merge_steps(step_indices, df['Filtered_AccNorm'].values, window_size=window_size)

    # 총 걸음 수 출력
    print(f"Total Detected Steps: {len(merged_step_indices)}")

    return merged_step_indices
