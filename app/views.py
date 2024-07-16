# exercise/views.py

from django.shortcuts import render
from django.http import StreamingHttpResponse
import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def gen_frames():  # generate frame by frame from camera
    cap = cv2.VideoCapture(0)
    
    # Initialize counters and stages for five exercises
    counters = [0] * 5
    stages = ['down'] * 5

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Exercise 1: Biceps Curl (left arm)
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                
                # Visualize left angle
                cv2.putText(image, str(left_angle), 
                            tuple(np.multiply(left_elbow, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Exercise 1 stage detection
                if left_angle > 160:
                    stages[0] = "down"
                if left_angle < 30 and stages[0] == 'down':
                    stages[0] = "up"
                    counters[0] += 1
                    print(f'Exercise 1 counter: {counters[0]}')
                
                # Repeat similar steps for the other four exercises
                # Replace the joint coordinates and angle thresholds according to the specific exercises
                
                # Exercise 2: (e.g., Right Arm Curl)
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                
                right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                
                cv2.putText(image, str(right_angle), 
                            tuple(np.multiply(right_elbow, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                
                if right_angle > 160:
                    stages[1] = "down"
                if right_angle < 30 and stages[1] == 'down':
                    stages[1] = "up"
                    counters[1] += 1
                    print(f'Exercise 2 counter: {counters[1]}')
                
                # Exercise 3: (e.g., Left Leg Stretch)
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                
                left_leg_angle = calculate_angle(left_hip, left_knee, left_ankle)
                
                cv2.putText(image, str(left_leg_angle), 
                            tuple(np.multiply(left_knee, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                
                if left_leg_angle > 160:
                    stages[2] = "down"
                if left_leg_angle < 30 and stages[2] == 'down':
                    stages[2] = "up"
                    counters[2] += 1
                    print(f'Exercise 3 counter: {counters[2]}')
                
                # Exercise 4: (e.g., Right Leg Stretch)
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                
                right_leg_angle = calculate_angle(right_hip, right_knee, right_ankle)
                
                cv2.putText(image, str(right_leg_angle), 
                            tuple(np.multiply(right_knee, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                
                if right_leg_angle > 160:
                    stages[3] = "down"
                if right_leg_angle < 30 and stages[3] == 'down':
                    stages[3] = "up"
                    counters[3] += 1
                    print(f'Exercise 4 counter: {counters[3]}')
                
                # Exercise 5: (e.g., Core Exercise)
                # Define the joints and angle calculation specific to this exercise
                
                core_point1 = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                core_point2 = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                core_point3 = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                            landmarks[mp_pose.PoseLandmark.NOSE.value].y]
                
                core_angle = calculate_angle(core_point1, core_point2, core_point3)
                
                cv2.putText(image, str(core_angle), 
                            tuple(np.multiply(core_point2, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                
                if core_angle > 160:
                    stages[4] = "down"
                if core_angle < 30 and stages[4] == 'down':
                    stages[4] = "up"
                    counters[4] += 1
                    print(f'Exercise 5 counter: {counters[4]}')
                
            except:
                pass

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
            
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

def video_feed(request):
    print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def home(request):
    return render(request, 'home.html')


from django.shortcuts import render
from django.http import StreamingHttpResponse, HttpResponseServerError
import cv2
import mediapipe as mp
import numpy as np
import base64

# Initialize Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to process frame with Mediapipe and return annotated image
def process_frame(frame):
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

        # Encode image to JPEG format
        _, encoded_image = cv2.imencode('.jpg', image)
        return encoded_image.tobytes()

# View function to stream video and detect landmarks
def detect_landmarks(request):
    # Function to capture video frames
    def video_stream():
        cap = cv2.VideoCapture(0)  # Use 0 for webcam, or use a video file path
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_image = process_frame(frame)

            # Encode image to base64 for HTML display
            encoded_img = base64.b64encode(processed_image).decode('utf-8')
            img_tag = f'<img src="data:image/jpeg;base64,{encoded_img}" />'

            # Yield HTML content with processed image embedded
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + processed_image + b'\r\n')

    # HTTP Response headers for Multipart Streaming
    return StreamingHttpResponse(video_stream(), content_type='multipart/x-mixed-replace; boundary=frame')

