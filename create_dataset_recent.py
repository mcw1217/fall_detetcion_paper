import cv2
import mediapipe as mp
import numpy as np
import time, os

actions = ["lie"]
seq_length = 30
secs_for_action = 30

# MediaPipe hands model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

cap = cv2.VideoCapture("./dataset/3-lie/liedata.mp4")
_,img = cap.read()
width = int(img.shape[1] / 2)
height = int(img.shape[0] /2)
cv2.namedWindow('img', 0)
cv2.resizeWindow("img",width,height )

created_time = int(time.time())
os.makedirs("dataset", exist_ok=True)

while cap.isOpened():
    for idx, action in enumerate(actions):
        idx = 3
        data = []

        ret, img = cap.read()

        # img = cv2.flip(img, 1)
        

        while True:
            ret, img = cap.read()
            
            if not ret:
                print("Ignoring empty camera frame.")
                break

            img = cv2.flip(img, 0)            
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

                    d = np.concatenate([joint[[11,12,13,14,15,23,24,25,26,27,28],:].flatten(), angle_label])

                    data.append(d)

                    mp_drawing.draw_landmarks(img, res, mp_pose.POSE_CONNECTIONS)

            cv2.imshow("img", img)
            if cv2.waitKey(1) == ord("q"):
                break

        data = np.array(data)
        print(action, data.shape)
        # np.save(os.path.join('dataset', f'raw_{action}_2023-2'), data)

        # Create sequence data
        full_seq_data = []
        for seq in range(len(data) - seq_length):
            vector_l = abs(data[seq] - data[seq + seq_length])
            full_seq_data.append(data[seq:seq + seq_length])

        full_seq_data = np.array(full_seq_data)
        print(action, full_seq_data.shape)
        np.save(os.path.join('dataset', f'seq_{action}-2023-6'), full_seq_data)
    break



