import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

def prepare_new_data(df_new, features, time_steps=100):
    X_new = df_new[features].values
    scaler = MinMaxScaler()
    X_new = scaler.fit_transform(X_new)
    X_new_lstm = [X_new[i-time_steps:i] for i in range(time_steps, len(X_new))]
    return np.array(X_new_lstm)

def predict_steps(model_path, df_new, features, time_steps=100):
    model = load_model(model_path)
    X_new = prepare_new_data(df_new, features, time_steps)
    predictions = model.predict(X_new)
    return np.argmax(predictions, axis=1)
