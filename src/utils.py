from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def calculate_class_weights(y_train):
    y_train_flat = np.argmax(y_train, axis=1).flatten()
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_flat), y=y_train_flat)
    return dict(enumerate(class_weights))
