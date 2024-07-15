# exercise/consumers.py

import cv2
import numpy as np
import mediapipe as mp
from channels.generic.websocket import AsyncWebsocketConsumer
import base64
import json
from channels.exceptions import StopConsumer

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

class VideoConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    async def disconnect(self, close_code):
        self.pose.close()
        raise StopConsumer

    async def receive(self, text_data):
        frame_data = json.loads(text_data)['frame']
        frame = self.decode_base64(frame_data)
        
        image = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)
        
        # Draw landmarks on the frame
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
        
        ret, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        await self.send(text_data=json.dumps({'frame': image_base64}))

    def decode_base64(self, data):
        return base64.b64decode(data)
