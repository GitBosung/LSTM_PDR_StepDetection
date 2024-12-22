from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, BatchNormalization, Dropout, Dense
from tensorflow.keras.optimizers import Adam

class LSTMModel:
    def build(self, input_shape):
        lstm_input = Input(shape=input_shape, name='lstm_input')
        lstm_layer = LSTM(128, return_sequences=True)(lstm_input)
        lstm_layer = BatchNormalization()(lstm_layer)
        lstm_layer = Dropout(0.5)(lstm_layer)
        lstm_layer = LSTM(64, return_sequences=False)(lstm_layer)
        dense_layer = Dense(32, activation='relu')(lstm_layer)
        output = Dense(3, activation='softmax')(dense_layer)
        model = Model(inputs=lstm_input, outputs=output)
        model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
        return model
