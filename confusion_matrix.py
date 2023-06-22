import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

actions = [
    'fall','stand','walking','lie','sit'
]
model = load_model('models/model.h5')

test_data = np.load("dataset/confusion_matrix/seq_fall-2023-1.npy")

test_data = np.nan_to_num(test_data)
test_x = test_data[:, :, :-1]
test_y = test_data[:, 0, -1]


test_y = to_categorical(test_y, num_classes=len(actions))

y_pred = model.predict(test_x)
print(np.argmax(y_pred, axis=1))
print(np.argmax(test_y, axis=1))

print(test_y.shape)
print(y_pred.shape)

print(confusion_matrix(np.argmax(test_y, axis=1), np.argmax(y_pred, axis=1)))
