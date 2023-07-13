import sys
import time
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime
now = datetime.now()


actions = ['fall','stand','walking','lie','sit']
seq_length = 30
queue= list()
pre_time=0
model = load_model('vec_model/model.h5')
# model = load_model('vec_model/model.h5') 
# model = load_model('cd_model/model.h5') 


# MediaPipe hands model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture("dataset/4-sit/sit-1.mp4")
_,img = cap.read()
width = int(img.shape[1] / 2)
height = int(img.shape[0] /2)
print(width)
cv2.namedWindow('img', 0) 
cv2.resizeWindow("img",width,height )

seq = []
action_seq = []
last_action = None
count=0

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break
    # img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = pose.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.pose_landmarks is not None:
            res = result.pose_landmarks
            joint = np.zeros((33, 4))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

            # Compute angles between joints
            v1 = joint[[26,24,25,23,14,16,13,15], :3] # Parent joint
            v2 = joint[[28,26,27,25,12,14,11,13], :3] # Child joint
            v = v2 - v1 # [20, 3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                    v[[0,2,5,7],:], 
                    v[[1,3,4,6],:] # [4,]15,]
                    )
                )

            angle = np.degrees(angle) # Convert radian to degree
            
            # 벡터의 방향성
            if count == 0:
                queue.append(joint[:,:3])
            pre_v = queue.pop()
            queue.append(joint[:, :3])
            new_v = joint[:,:3] - pre_v
            # new_v = new_v / np.linalg.norm(new_v, axis=1)[:, np.newaxis]
            
            #벡터의 가속도 
            cur_time = float(datetime.now().strftime('%Y%m%d%H%M%S.%f'))
            if count== 0:
                pre_time = cur_time-0.001
            speed = (joint[:,:3] - pre_v) / (cur_time-pre_time) # 현재 프레임의 속도
            pre_time = cur_time
            
            
            d = np.concatenate([joint[:].flatten(), new_v.flatten(),speed.flatten(),angle])
            # d = np.concatenate([new_v.flatten(),speed.flatten(),angle])
            # d = np.concatenate([joint[:].flatten(),angle])
            d = np.nan_to_num(d) 
            
            seq.append(d)
        
            

            mp_drawing.draw_landmarks(img, res, mp_pose.POSE_CONNECTIONS)

            if len(seq) < seq_length:
                continue

            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

            y_pred = model.predict(input_data).squeeze()

            i_pred = int(np.argmax(y_pred))
            conf = y_pred[i_pred]

            if conf < 0.8:   # 넘어지는 순간 정확도가 애매해지기 때문에 계속해서 continue 처리가 된다. continue 처리가 되면 새로운 이미지가 들어오기때문에 송출시에는 끊어지는 것 처럼 보인다.
                continue

            action = actions[i_pred]
            action_seq.append(action)

            if len(action_seq) < 3:
                continue

            this_action = '?'
            if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                this_action = action

            cv2.putText(img, f'{this_action.upper()}', org=(int(res.landmark[ 0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break


