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
    # test_data = np.load(f'./dataset/cd_confusion_matrix/seq_fall-2023-{i}.npy')
    test_data = np.load(f'./dataset/confusion_matrix/seq_fall-2023-{i}.npy')
    first = True
    if test_data.shape[0] > 2:
        # for i in range(test_data.shape[0]):
        test_data = np.nan_to_num(test_data)
        test_x = test_data[:, :, :-1]
        test_y = test_data[:-2, 0, -1]
        print(test_y,'test_y')


        test_y = to_categorical(test_y, num_classes=len(actions))
        if first is True:
            result_y.extend(np.argmax(test_y, axis=1))
            first = False
        y_pred = model.predict(test_x) # 모든 프레임에 대한 predict가 나옴

        i_pred = (np.argmax(y_pred, axis=1)) # 행동의 index가 저장
        conf_s = y_pred[[q for q in range(y_pred.shape[0])],i_pred] 
        # y_pred에서 가장 신뢰도가 높은 인덱스를 뽑아내 신뢰도만 남게 한다.
        #[q for q in range(y_pred.shape[0])]는 인덱스의 첫번째부터 지정하기 위해 넣은 코드

        for j in range(conf_s.shape[0]):
            conf = conf_s[j]
            if conf < 0.8:
                i_pred[j] = 7
            action_seq.append(i_pred[j])
        action_seq = np.sort(action_seq)[::-1] # 54번 코드에서 반대부터 행동의 신뢰도를 판단하기때문에 역정렬을 하여 영상의 순서를 맞춰준다.
        print(action_seq,'action_seq')

        if len(action_seq) < 3:
            print("Error : action_seq length < 3")

        
        while len(action_seq) > 2: #predict된 행동들이 2개 이상일때까지만 작동
            this_action = 7 # 연속되지 않은 행위가 나올 경우의 라벨
            if action_seq[-1] == action_seq[-2] == action_seq[-3]: #연속된 3번의 같은 행위가 나올때만 행위로 인정
                this_action = i_pred[-1]
                i_pred = i_pred[:-1]
                action_seq = action_seq[:-1]
                x_test.append(this_action)
            else: # 연속되지 않은 행위가 나오면 Denied 결과를 내보냄
                i_pred = i_pred[:-1]
                action_seq = action_seq[:-1]
                x_test.append(this_action)

        result_pre.extend(x_test)
        print(result_pre,'result_pre')
        x_test = []
        action_seq = []
    else:
        print(f"Error : {i}번째가 2개 이하입니다.")


from sklearn.metrics import classification_report
print('\nClassification Report\n')
print(classification_report(result_y, result_pre))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

y_test = result_y  # Actual classes
y_pred = result_pre  # Predicted classes

print('y_test : ',y_test)
print('y_pred : ',y_pred)


label = []
for number in np.unique(y_pred):
    if number == 7:
        label.append('Denied')
    else:
        label.append(actions[number])

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
print(confusion_matrix)

# Confusion matrix 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='g',xticklabels=label,yticklabels=label)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# accuracy = metrics.accuracy_score(y_test, y_pred)
# precision = metrics.precision_score(y_test, y_pred)
# recall = metrics.recall_score(y_test, y_pred)
# f1 = metrics.f1_score(y_test, y_pred)
#
# # Print metrics
# print("Accuracy:", accuracy)
# print("Precision:", precision)
# print("Recall:", recall)
# print("F1 Score:", f1)
#
#
#
#