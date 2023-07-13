# fall_detetcion_paper

[COMMIT] :  MODIFY CONFUSION MATRIX
1. confusion_matrix.py 코드에서 51번째 줄의 confidence를 확인하는 코드에 오류가 발생하여 모든 confidence가 첫번째 prediction의 confidence로 고정되는 오류를 해결
2. testdata폴더의 30개의 테스트 데이터의 Sliding window를 적용하여 데이터를 predict할 수 있도록 변경 / 10번째 데이터는 사용 불가능
3. cd_confusion_matrix(관절좌표 matrix용 데이터셋)의 정확도보다 confusion_matrix(관절좌표,속도,방향 martix용 데이터셋)이 더 높은 정확도를 보임

## 관절좌표 모델의 Confusion Matrix
![관절좌표 모델](https://github.com/mcw1217/fall_detetcion_paper/assets/87608623/788e25df-458c-4e6b-90c6-78b740a5ac52)


## 관절좌표,속도,시간 모델의 Confusion Matrix
![속도,방향 모델](https://github.com/mcw1217/fall_detetcion_paper/assets/87608623/8eac431d-6789-4a59-b48b-ce45ef9c63cd)

- 테스트 결과 '관절좌표 모델'에 비해 '관절좌표,속도,시간 모델' 의 Fall의 감지율이 더 높으며, 오류율 역시 더 적게 나타난다.
- Confusion Matrix의 Denied는 행동 감지 실패를 의미한다.
- 테스트는 불특정 다수의 낙상 영상 30개를 기준으로 함

- 관절좌표 모델의 Recall: 88%
- 관절좌표, 속도, 시간 모델의 Recall: 95%
