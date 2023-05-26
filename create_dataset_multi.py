import cv2
import mediapipe as mp
import numpy as np
import time, os

actions = ["walking"]
seq_length = 30
secs_for_action = 30 
queue = list()
count = 0

# MediaPipe hands model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)
for countdown in range(1,2):
    cap = cv2.VideoCapture(f"./dataset/test/walking-{countdown}.mp4")
    _,img = cap.read()
    width = int(img.shape[1] / 2)
    height = int(img.shape[0] /2)
    cv2.namedWindow('img', 0)
    cv2.resizeWindow("img",width,height )

    created_time = int(time.time())
    os.makedirs("dataset", exist_ok=True)

    while cap.isOpened():
        for idx, action in enumerate(actions):
            idx = 2
            data = []

            ret, img = cap.read()

            # img = cv2.flip(img, 1)
            

            while True:
                ret, img = cap.read()
                
                if not ret:
                    print("Ignoring empty camera frame.")
                    break

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

                        if count == 0:
                            queue.append(joint[:,:3])
                            print("[NOTICE] : First data appended queue!")
                            print("[NOTICE] : Input data shape - ", queue[0].shape)
                        pre_v = queue.pop()
                        print("[NOTICE] : 큐에서 이전 프레임을 추출")
                        print("[NOTICE] : 현재 큐에 남은 데이터 - ", queue)
                        queue.append(joint[:, :3])
                        print("[NOTICE] : 큐에 현재 프레임 입력")
                        print("[NOTICE] : 현재 큐에 남은 데이터 - ", queue)          
                        
                        new_v = joint[:,:3] - pre_v
                        new_v = new_v / np.linalg.norm(new_v, axis=1)[:, np.newaxis]
                        print("[NOTICE] : 벡터의 방향 계산 완료!")
                        print("[NOTICE] : 현재 벡터의 방향값 - ",new_v)

                        # 기존의 관절좌표 데이터와 라벨 데이터 생성
                        d = np.concatenate([joint[:].flatten(), angle_label])
                        # 생성된 데이터에 이전 프레임과 현재 프레임의 벡터의 방향을 추가
                        d = np.concatenate([d.flatten(), new_v.flatten()])
                        print("[NOTICE] : 생성된 데이터의 Shape - ",d.shape)

                        data.append(d)

                        mp_drawing.draw_landmarks(img, res, mp_pose.POSE_CONNECTIONS)

                cv2.imshow("img", img)
                count += 1
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
            np.save(os.path.join('dataset/sampledata', f'seq_{action}-2023-{countdown}'), full_seq_data)
            cv2.destroyAllWindows()
        break

