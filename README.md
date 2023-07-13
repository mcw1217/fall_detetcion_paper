# fall_detetcion_paper

## 관절좌표 모델의 Confusion Matrix
![관절좌표 모델](https://github.com/mcw1217/fall_detetcion_paper/assets/87608623/788e25df-458c-4e6b-90c6-78b740a5ac52)


## 관절좌표,속도,시간 모델의 Confusion Matrix
![속도,방향 모델](https://github.com/mcw1217/fall_detetcion_paper/assets/87608623/8eac431d-6789-4a59-b48b-ce45ef9c63cd)

- 테스트 결과 '관절좌표 모델'에 비해 '관절좌표,속도,시간 모델' 의 Fall의 감지율이 더 높으며, 오류율 역시 더 적게 나타난다.
- Confusion Matrix의 Denied는 행동 감지 실패를 의미한다.
- 테스트는 불특정 다수의 낙상 영상 30개를 기준으로 함

- 관절좌표 모델의 Recall: 88%
- 관절좌표, 속도, 시간 모델의 Recall: 95%



--

[COMMIT] :  MODIFY CONFUSION MATRIX
1. confusion_matrix.py 코드에서 51번째 줄의 confidence를 확인하는 코드에 오류가 발생하여 모든 confidence가 첫번째 prediction의 confidence로 고정되는 오류를 해결
2. testdata폴더의 30개의 테스트 데이터의 Sliding window를 적용하여 데이터를 predict할 수 있도록 변경 / 10번째 데이터는 사용 불가능
3. cd_confusion_matrix(관절좌표 matrix용 데이터셋)의 정확도보다 confusion_matrix(관절좌표,속도,방향 martix용 데이터셋)이 더 높은 정확도를 보임

[COMMIT] : MODIFY CONFUSION MATRIX 2 - 2023-07-13
1. confusion_matrix.py 코드를 multi_label_confusion_matrix_create.py 코드로 변경 - multi_label_confusion_matrix 시각화 기능 추가, 시각화 label 오류 수정,
2. multi_label_confusion_matrix.py의 평가기능 오류 수정( 수정으로 인하여 정확한 confusion_matrix 평가가 가능해짐 )
3. multi_label_confusion_matrix.py의 45번째 줄의 action_seq 리스트의 값을 역정렬하여 행동의 Confidence 평가가 제대로 되도록 수정( 수정을 통하여 정확도가 미세하게 달라짐 )
