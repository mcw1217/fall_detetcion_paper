import cv2
import mediapipe as mp
import numpy as np
import time, os
from datetime import datetime

#설정 파라미터
actions = ["fall"]
indexing = 0
media_size= 30
flip_option = True #처음 생성할때는 False / 반전데이터 생성시에만 True 

plus_size= 0
if flip_option == True:
    plus_size = media_size
    

seq_length = 30
secs_for_action = 30 
queue = list()
pre_time = 0


# MediaPipe hands model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)
for countdown in range(1,media_size+1):
    # cap = cv2.VideoCapture(f"./dataset/new_testdata/test{countdown}.mp4")
    cap = cv2.VideoCapture(f"./dataset/new_testdata/test{countdown}.mp4")
    _,img = cap.read()
    width = int(img.shape[1] / 2)
    height = int(img.shape[0] /2)
    cv2.namedWindow('img', 0)
    cv2.resizeWindow("img",width,height )

    created_time = int(time.time())
    os.makedirs("dataset", exist_ok=True)
    count = 0

    while cap.isOpened():
        for idx, action in enumerate(actions):
            idx = indexing
            data = []
            
            while True:
                ret, img = cap.read()
                
                if not ret:
                    print("Ignoring empty camera frame.")
                    break
                
                if flip_option == True:
                    img = cv2.flip(img, 1)            
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = pose.process(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                if result.pose_landmarks is not None:
                        res = result.pose_landmarks
                        joint = np.zeros((33, 4))
                        for j, lm in enumerate(result.pose_landmarks.landmark):
                            joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                        # Compute angles between joints
                        v1 = joint[[26,24,25,23,14,16,13,15], :3] # Parent joint
                        v2 = joint[[28,26,27,25,12,14,11,13], :3] # Child joint
                        v = v2 - v1 # [20, 3]
                        # Normalize v
                        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                        # Get angle using arcos of dot product
                        angle = np.arccos(
                            np.einsum(
                                "nt,nt->n",
                                v[[0,2,5,7],:], 
                                v[[1,3,4,6],:] # [4,]
                            )
                        )  # [15,]

                        angle = np.degrees(angle)  # Convert radian to degree

                        angle_label = np.array([angle], dtype=np.float32)
                        angle_label = np.append(angle_label, idx)
                        
                        # 첫 프레임일 경우만 queue에 현재 좌표를 넣어 0을 만듦
                        # 벡터의 방향성 구하는 코드
                        if count == 0:
                            queue.append(joint[:,:3])
                            print("[NOTICE] : First data appended queue!")
                            print("[NOTICE] : Input data shape - ", queue[0].shape)
                        pre_v = queue.pop()
                        queue.append(joint[:, :3])
                        
                        new_v = joint[:,:3] - pre_v
                        new_v = new_v / np.linalg.norm(new_v, axis=1)[:, np.newaxis]
                        
                        #벡터의 속도
                        cur_time = float(datetime.now().strftime('%Y%m%d%H%M%S.%f'))
                        if count== 0:
                            pre_time = cur_time-0.001
                        speed = (joint[:,:3] - pre_v) / (cur_time-pre_time) # 현재 프레임의 속도
                        pre_time = cur_time
                        
                        
                        #데이터 결합
                        d = np.concatenate([joint[:].flatten(), new_v.flatten(), speed.flatten(), angle_label])
                        # d = np.concatenate([new_v.flatten(), speed.flatten(), angle_label])
                        # d = np.concatenate([joint[:].flatten(), angle_label])

                        data.append(d)

                        mp_drawing.draw_landmarks(img, res, mp_pose.POSE_CONNECTIONS)
                        count += 1

                cv2.imshow("img", img)
                if cv2.waitKey(1) == ord("q"):
                    break

            data = np.array(data)
            print(action, data.shape)
            # np.save(os.path.join('dataset', f'raw_{action}_2023-2'), data)

            # Create sequence data
            full_seq_data = []
            for seq in range(len(data) - seq_length):
                full_seq_data.append(data[seq:seq + seq_length])    

            full_seq_data = np.array(full_seq_data)
            print(action, full_seq_data.shape)
            np.save(os.path.join('./dataset/confusion_matrix', f'seq_{action}-2023-{countdown+plus_size}'), full_seq_data)
            cv2.destroyAllWindows()
        break

