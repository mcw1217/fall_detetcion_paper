# fall_detetcion_paper

[COMMIT] :  MODIFY CONFUSION MATRIX
1. confusion_matrix.py 코드에서 51번째 줄의 confidence를 확인하는 코드에 오류가 발생하여 모든 confidence가 첫번째 prediction의 confidence로 고정되는 오류를 해결
2. testdata폴더의 30개의 테스트 데이터의 Sliding window를 적용하여 데이터를 predict할 수 있도록 변경 / 10번째 데이터는 사용 불가능
3. cd_confusion_matrix(관절좌표 matrix용 데이터셋)의 정확도보다 confusion_matrix(관절좌표,속도,방향 martix용 데이터셋)이 더 높은 정확도를 보임
![관절좌표 Confusion](https://github.com/mcw1217/fall_detetcion_paper/assets/87608623/3f138a77-72a0-4925-b078-cf2c6c1b404a)
![속도,시간 Confusion](https://github.com/mcw1217/fall_detetcion_paper/assets/87608623/191905bf-02f3-4ba4-9a66-f70040e06255)
