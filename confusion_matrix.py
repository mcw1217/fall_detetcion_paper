import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix ,classification_report,accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

actions = [
    'fall','stand','walking','lie','sit'
]
action_seq = []
# model = load_model('cd_model/model.h5')
model = load_model('vec_model/model.h5')
# model = load_model('models/model.h5')
y_test = []
x_test = []
result_y = []
result_pre = []
for i in range(1,30):
    test_data = np.load(f'./dataset/confusion_matrix/seq_fall-2023-{i}.npy')
    first = True
    if test_data.shape[0] > 2:
        for i in range(test_data.shape[0]):
            # #관절좌표
            # test_data = np.load("dataset/cd_confusion_matrix/seq_fall-2023-1.npy")
            # for i in range(2,30):
            #     test_data = np.concatenate([
            #         test_data,
            #         np.load(f'./dataset/cd_confusion_matrix/seq_fall-2023-{i}.npy')
            #     ], axis=0) 
            # print(test_data.shape)

            # 속도,방향
            test_data = np.nan_to_num(test_data)
            test_x = test_data[:, :, :-1]
            test_y = test_data[:, 0, -1]
            # print(test_x.shape, test_y.shape)

            test_y = to_categorical(test_y, num_classes=len(actions))
            if first is True:
                result_y.extend(np.argmax(test_y, axis=1))
                first = False
            y_pred = model.predict(test_x)
            # print(y_pred,'y_pred')

            i_pred = (np.argmax(y_pred, axis=1))
            # print(i_pred,"i_pred")
            # print(y_pred.shape)
            conf_s = y_pred[[q for q in range(y_pred.shape[0])],i_pred]
            # print(conf_s, "conf")
            for j in range(conf_s.shape[0]):
                # print(j,"jj")
                conf = conf_s[j]
                # print(conf)
                if conf < 0.8:
                    i_pred[j] = 5
                action_seq.append(i_pred[j])
            # print(action_seq,"action_seq")

            if len(action_seq) < 3:
                print("Error : action_seq length < 3")
            
            this_action = 7
            if len(action_seq) > 2: 
                if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                    this_action = i_pred[-1]
                    
                    i_pred = i_pred[:-1]
                    action_seq = action_seq[:-1]
                    
                x_test.append(this_action)                
            
        result_pre.extend(x_test)
        x_test = []
        action_seq = []
    else:
        print(f"Error : {i}번째가 2개 이하입니다.")    






# 평가
print(result_pre,"result_pre")
print(result_y,"result_y")
print(confusion_matrix(result_y, result_pre))


print('\nAccuracy: {:.2f}\n'.format(accuracy_score(result_y, result_pre)))

print('Micro Precision: {:.2f}'.format(precision_score(result_y, result_pre, average='micro')))
print('Micro Recall: {:.2f}'.format(recall_score(result_y, result_pre, average='micro')))
print('Micro F1-score: {:.2f}\n'.format(f1_score(result_y, result_pre, average='micro')))

print('Weighted Precision: {:.2f}'.format(precision_score(result_y, result_pre, average='weighted')))
print('Weighted Recall: {:.2f}'.format(recall_score(result_y, result_pre, average='weighted')))
print('Weighted F1-score: {:.2f}'.format(f1_score(result_y, result_pre, average='weighted')))

from sklearn.metrics import classification_report
print('\nClassification Report\n')
print(classification_report(result_y, result_pre))


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

y_test = result_y  # Actual classes
y_pred = result_pre  # Predicted classes

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Compute metrics
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred)

# Print metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
