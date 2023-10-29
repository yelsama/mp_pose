import cv2
import mediapipe as mp
import numpy as np

def calculate_angle(a, b, c):
	a = np.array(a) # First
	b = np.array(b) # Mid
	c = np.array(c) # End

	radian = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
	angle = np.abs(radian*180.0/np.pi)
	if angle >180.0:
		angle = 360-angle
	return angle

cap = cv2.VideoCapture(0)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose =  mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

while True:
    ret, image = cap.read()

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = pose.process(image)

    try:
        landmarks = results.pose_landmarks.landmark
        l_sholder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        print(l_sholder)
    except:
        pass
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
    )

    cv2.imshow('Mediapipe Feed', image)	
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
