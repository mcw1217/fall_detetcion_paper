import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix ,classification_report,accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

actions = [
    'fall','stand','walking','lie','sit'
]
# model = load_model('cd_model/model.h5')
model = load_model('vec_model/model.h5')

# #관절좌표
# test_data = np.load("dataset/cd_confusion_matrix/seq_fall-2023-1.npy")
# for i in range(2,30):
#     test_data = np.concatenate([
#         test_data,
#         np.load(f'./dataset/cd_confusion_matrix/seq_fall-2023-{i}.npy')
#     ], axis=0) 
# print(test_data.shape)

# 속도,방향
test_data = np.load("dataset/confusion_matrix/seq_fall-2023-1.npy")
for i in range(2,30):
    test_data = np.concatenate([
        test_data,
        np.load(f'./dataset/confusion_matrix/seq_fall-2023-{i}.npy')
    ], axis=0) 
print(test_data.shape)

test_data = np.nan_to_num(test_data)
test_x = test_data[:, :, :-1]
test_y = test_data[:, 0, -1]


test_y = to_categorical(test_y, num_classes=len(actions))

y_pred = model.predict(test_x)
print(np.argmax(y_pred, axis=1))
# print(np.argmax(np.argmax(test_y,axis=1), axis=1))

print(test_y.shape)
print(y_pred.shape)

print(confusion_matrix(np.argmax(test_y, axis=1), np.argmax(y_pred, axis=1)))


print('\nAccuracy: {:.2f}\n'.format(accuracy_score(np.argmax(test_y,axis=1), np.argmax(y_pred,axis=1))))

print('Micro Precision: {:.2f}'.format(precision_score(np.argmax(test_y,axis=1), np.argmax(y_pred,axis=1), average='micro')))
print('Micro Recall: {:.2f}'.format(recall_score(np.argmax(test_y,axis=1), np.argmax(y_pred,axis=1), average='micro')))
print('Micro F1-score: {:.2f}\n'.format(f1_score(np.argmax(test_y,axis=1), np.argmax(y_pred,axis=1), average='micro')))

print('Weighted Precision: {:.2f}'.format(precision_score(np.argmax(test_y,axis=1), np.argmax(y_pred,axis=1), average='weighted')))
print('Weighted Recall: {:.2f}'.format(recall_score(np.argmax(test_y,axis=1), np.argmax(y_pred,axis=1), average='weighted')))
print('Weighted F1-score: {:.2f}'.format(f1_score(np.argmax(test_y,axis=1), np.argmax(y_pred,axis=1), average='weighted')))

from sklearn.metrics import classification_report
print('\nClassification Report\n')
print(classification_report(np.argmax(test_y,axis=1), np.argmax(y_pred,axis=1), target_names=['0', '3']))
