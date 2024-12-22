import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from src.preprocessing import filter_data, prepare_training_data
from src.model_training import LSTMModel

# 데이터 로드 및 전처리
dfs = [
    filter_data(pd.read_csv('data/train/labeled_looking.csv')),
    filter_data(pd.read_csv('data/train/labeled_swing.csv')),
    filter_data(pd.read_csv('data/train/labeled_pocket.csv'))
]
features = ['Filtered_Acc_x', 'Filtered_Acc_y', 'Filtered_Acc_z', 'Filtered_Gyro_x', 'Filtered_Gyro_y', 'Filtered_Gyro_z']
label_column = 'labels'
time_steps = 100

# 학습 데이터 준비
X, y = prepare_training_data(dfs, features, label_column, time_steps)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 생성 및 학습
model_instance = LSTMModel()
model = model_instance.build(input_shape=(X_train.shape[1], X_train.shape[2]))
early_stopping = EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True)
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=32, callbacks=[early_stopping])

# 모델 저장
model.save('models/lstm_model.h5')
print("모델이 저장되었습니다: models/lstm_model.h5")
