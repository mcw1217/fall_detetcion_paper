import sys
import time
# sys.path.append('pingpong')
# from pingpong.pingpongthread import PingPongThread
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# PingPongThreadInstance = PingPongThread(number=2)
# PingPongThreadInstance.start()
# PingPongThreadInstance.wait_until_full_connect()

actions = ['fall','stand']
seq_length = 30

model = load_model('models/model.h5')


# MediaPipe hands model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture('dataset/test5.mp4')

seq = []
action_seq = []
last_action = None

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.flip(img, 1)
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

            d = np.concatenate([joint[[11,12,13,14,15,23,24,25,26,27,28],:].flatten(), angle])

            seq.append(d)

            mp_drawing.draw_landmarks(img, res, mp_pose.POSE_CONNECTIONS)

            if len(seq) < seq_length:
                continue

            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

            y_pred = model.predict(input_data).squeeze()

            i_pred = int(np.argmax(y_pred))
            conf = y_pred[i_pred]

            if conf < 0.8:   # 넘어지는 순간 정확도가 애매해지기 때문에 계속해서 continue 처리가 된다. continue 처리가 되면 새로운 이미지가 들어오기때문에 송출시에는 끊어지는 것 처럼 보인다.
                cv2.imshow('img',img)
                continue

            action = actions[i_pred]
            action_seq.append(action)

            if len(action_seq) < 3:
                cv2.imshow('img',img)
                continue

            this_action = '?'
            if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                this_action = action

            cv2.putText(img, f'{this_action.upper()}', org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break


